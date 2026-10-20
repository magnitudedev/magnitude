use std::collections::{BTreeMap, BTreeSet};
use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use axum::extract::{DefaultBodyLimit, Path, Query, Request, State};
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, put};
use axum::{Json, Router};
use futures_util::{future::BoxFuture, stream::BoxStream};
use magnitude_service_contracts::bootstrap_protocol::{
    IcnBinaryIdentity, IcnInstallationDeclaration, IcnParentCommand, IcnParentCommandType,
    IcnStartupBackend, IcnStartupProgressRecord, IcnStartupProgressRecordType, IcnStartupRecord,
    IcnStartupRecordType,
};
use magnitude_service_contracts::models::{
    CandidateInspection, CandidateInspectionEvent, CandidateInspectionProgress,
    CandidateInspectionProgressSink, CandidateInspectionRequest, CatalogComponent,
    CatalogInstallationAdmission, CatalogInstallationOperation, CatalogInstallationOperationId,
    CatalogInstallationRemoval, CatalogInstallations, CatalogInstallationsResponse, CatalogModel,
    CatalogModelState, CatalogModels, CatalogModelsResponse, CustomModel, CustomModelOrigin,
    CustomModelState, DrafterCandidatesRequest, DrafterCandidatesResponse, DrafterPruning,
    EffectiveModel, FoundSource, LocalModelRemoval, LocalModels, LocalModelsConfiguration,
    LocalModelsResponse, ModelAssessment, ModelAssessmentDomainSnapshot, ModelAssessmentEntryState,
    ModelAssessmentSubject, ModelAssessments, ModelAssessmentsSnapshot, ModelCandidate,
    ModelCandidates, ModelCapabilities, ModelDownload, ModelDownloadId, ModelDownloads,
    ModelDownloadsResponse, ModelId, ModelInstance, ModelInstanceId, ModelInstancesInvalidation,
    ModelInstancesSnapshot, ModelLoadDevice, ModelLoadPlan, ParsedModelId, ServableModelBundle,
    StartModelDownloadRequest, StartModelDownloadResponse,
};
use magnitude_service_contracts::{
    ExecutionBackend, HardwareProvider, HardwareSnapshot, HuggingFaceCredential,
    HuggingFaceModelCatalog,
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
    local_models: Option<Arc<dyn LocalModels>>,
    catalog_installations: Option<Arc<dyn CatalogInstallations>>,
    model_assessments: Option<Arc<dyn ModelAssessments>>,
    model_downloads: Option<Arc<dyn ModelDownloads>>,
    model_candidates: Option<Arc<dyn ModelCandidates>>,
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
    LocalModels,
    ModelAssessments,
    CatalogInstallations,
    Downloads,
    Instances,
}

impl InferenceResourceTopic {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Hardware => "hardware",
            Self::Catalog => "catalog",
            Self::LocalModels => "local-models",
            Self::ModelAssessments => "model-assessments",
            Self::CatalogInstallations => "catalog-installations",
            Self::Downloads => "downloads",
            Self::Instances => "instances",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        match value {
            "hardware" => Some(Self::Hardware),
            "catalog" => Some(Self::Catalog),
            "local-models" => Some(Self::LocalModels),
            "model-assessments" => Some(Self::ModelAssessments),
            "catalog-installations" => Some(Self::CatalogInstallations),
            "downloads" => Some(Self::Downloads),
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
            local_models: None,
            catalog_installations: None,
            model_assessments: None,
            model_downloads: None,
            model_candidates: None,
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
        local_models: Arc<dyn LocalModels>,
        catalog_installations: Arc<dyn CatalogInstallations>,
    ) -> Self {
        self.catalog_models = Some(catalog_models);
        self.local_models = Some(local_models);
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

    pub fn with_model_candidates(mut self, model_candidates: Arc<dyn ModelCandidates>) -> Self {
        self.model_candidates = Some(model_candidates);
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
            "/api/v1/catalog/models/{model_id}/components/{component}/install",
            post(install_catalog_component),
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
        .route(
            "/api/v1/local-models/configuration",
            get(local_models_configuration).put(configure_local_models),
        )
        .route("/api/v1/local-models", get(local_models))
        .route("/api/v1/local-models/refresh", post(refresh_local_models))
        .route(
            "/api/v1/local-models/{model_id}/files",
            axum::routing::delete(remove_local_model_files),
        )
        .route(
            "/api/v1/local-models/drafters/prune",
            post(prune_local_drafters),
        )
        .route("/api/v1/candidates/inspect", post(inspect_model_candidate))
        .route(
            "/api/v1/candidates/inspect/stream",
            post(stream_model_candidate_inspection),
        )
        .route("/api/v1/candidates/drafters", post(drafter_candidates))
        .route("/api/v1/candidates/bundle", post(candidate_bundle))
        .route("/api/v1/model-assessments", get(model_assessments))
        .route(
            "/api/v1/model-assessments/candidates",
            post(assess_model_candidate),
        )
        .route(
            "/api/v1/downloads",
            get(model_downloads).post(start_model_download),
        )
        .route(
            "/api/v1/downloads/{download_id}/cancel",
            post(cancel_model_download),
        )
        .route(
            "/api/v1/downloads/{download_id}/acknowledge-failure",
            post(acknowledge_model_download_failure),
        )
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
        .route(
            "/api/v1/sources/hugging-face/credential",
            put(configure_hugging_face_credential),
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
        .layer(DefaultBodyLimit::max(
            magnitude_serving::MAX_HTTP_BODY_BYTES,
        ))
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
            InventoryError::DoesNotFit(_) => (
                StatusCode::UNPROCESSABLE_ENTITY,
                "invalid_request_error",
                "model_does_not_fit",
                false,
            ),
            InventoryError::ModelUnsupported { .. } => (
                StatusCode::UNPROCESSABLE_ENTITY,
                "invalid_request_error",
                "model_unsupported",
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
        (status = 422, description = "Installed model failed integrity validation, does not fit, or the engine cannot run it (`model_unsupported`)", body = ErrorResponse),
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
    let local = state
        .local_models
        .as_ref()
        .ok_or_else(|| ApiError::server("local models are not configured"))?;
    let downloads = state
        .model_downloads
        .as_ref()
        .ok_or_else(|| ApiError::server("model downloads are not configured"))?;
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
    let installation_events =
        futures_util::StreamExt::map(installations.watch_catalog_installations(), |event| {
            InferenceResourceInvalidation {
                topic: InferenceResourceTopic::CatalogInstallations,
                revision: event.revision,
            }
        });
    let catalog_events = futures_util::StreamExt::map(catalog.watch_catalog(), |event| {
        InferenceResourceInvalidation {
            topic: InferenceResourceTopic::Catalog,
            revision: event.revision,
        }
    });
    let local_events =
        futures_util::StreamExt::map(local.watch_local(), |event| InferenceResourceInvalidation {
            topic: InferenceResourceTopic::LocalModels,
            revision: event.revision,
        });
    let download_events =
        futures_util::StreamExt::map(downloads.watch(), |event| InferenceResourceInvalidation {
            topic: InferenceResourceTopic::Downloads,
            revision: event.revision,
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
            futures_util::stream::select(local_events, download_events),
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
        (status = 404, description = "Repository or revision not found", body = ErrorResponse),
        (status = 409, description = "The operation conflicts with the model's current state", body = ErrorResponse),
        (status = 422, description = "The repository's metadata or a file's header is invalid", body = ErrorResponse),
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
        (status = 404, description = "Repository or revision not found", body = ErrorResponse),
        (status = 409, description = "The operation conflicts with the model's current state", body = ErrorResponse),
        (status = 422, description = "The repository's metadata or a file's header is invalid", body = ErrorResponse),
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

#[utoipa::path(put, path = "/api/v1/sources/hugging-face/credential", operation_id = "configureHuggingFaceCredential", tag = "hugging-face",
    request_body(content = HuggingFaceCredential, content_type = "application/json"),
    responses(
        (status = 204, description = "Later Hub requests present this token, or none when it is absent. The token is write-only: no response returns it"),
        (status = 400, description = "Empty token", body = ErrorResponse),
        (status = 500, description = "The credential could not be applied", body = ErrorResponse)
    )
)]
#[tracing::instrument(name = "icn.hugging_face.configure_credential", skip_all, err(Debug))]
async fn configure_hugging_face_credential(
    State(state): State<AppState>,
    Json(credential): Json<HuggingFaceCredential>,
) -> Result<StatusCode, ApiError> {
    let catalog = state
        .hugging_face_catalog
        .as_ref()
        .ok_or_else(|| ApiError::server("Hugging Face discovery is not configured"))?;
    if credential
        .token
        .as_deref()
        .is_some_and(|token| token.trim().is_empty())
    {
        return Err(ApiError::invalid("a Hugging Face token must not be empty"));
    }
    catalog
        .configure_credential(credential)
        .await
        .map(|()| StatusCode::NO_CONTENT)
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

#[utoipa::path(post, path = "/api/v1/catalog/models/{model_id}/components/{component}/install", operation_id = "installCatalogComponent", tag = "catalog",
    params(
        ("model_id" = String, Path, description = "Canonical catalog model ID"),
        ("component" = CatalogComponent, Path, description = "The one component to fetch")
    ),
    responses(
        (status = 200, description = "Catalog installation admission", body = CatalogInstallationAdmission),
        (status = 404, description = "Catalog model not found", body = ErrorResponse),
        (status = 409, description = "Installation cannot be admitted", body = ErrorResponse),
        (status = 500, description = "Installation failed", body = ErrorResponse)
    )
)]
async fn install_catalog_component(
    State(state): State<AppState>,
    Path((model_id, component)): Path<(String, CatalogComponent)>,
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
        .install_catalog_component(&parsed_id, component)
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

fn local_models_domain(state: &AppState) -> Result<&Arc<dyn LocalModels>, ApiError> {
    state
        .local_models
        .as_ref()
        .ok_or_else(|| ApiError::server("local models are not configured"))
}

fn model_candidates_domain(state: &AppState) -> Result<&Arc<dyn ModelCandidates>, ApiError> {
    state
        .model_candidates
        .as_ref()
        .ok_or_else(|| ApiError::server("model candidates are not configured"))
}

fn model_downloads_domain(state: &AppState) -> Result<&Arc<dyn ModelDownloads>, ApiError> {
    state
        .model_downloads
        .as_ref()
        .ok_or_else(|| ApiError::server("model downloads are not configured"))
}

#[utoipa::path(get, path = "/api/v1/local-models/configuration", operation_id = "getLocalModelsConfiguration", tag = "local-models",
    responses(
        (status = 200, description = "The active local-models configuration", body = LocalModelsConfiguration),
        (status = 500, description = "Local models unavailable", body = ErrorResponse)
    )
)]
async fn local_models_configuration(
    State(state): State<AppState>,
) -> Result<Json<LocalModelsConfiguration>, ApiError> {
    local_models_domain(&state)?
        .configuration()
        .await
        .map(Json)
        .map_err(ApiError::from_inventory)
}

#[utoipa::path(put, path = "/api/v1/local-models/configuration", operation_id = "configureLocalModels", tag = "local-models",
    request_body(content = LocalModelsConfiguration, content_type = "application/json"),
    responses(
        (status = 200, description = "The applied local-models configuration; applying the same snapshot again changes nothing", body = LocalModelsConfiguration),
        (status = 400, description = "Invalid configuration", body = ErrorResponse),
        (status = 500, description = "Configuration could not be applied", body = ErrorResponse)
    )
)]
#[tracing::instrument(name = "icn.local_models.configure", skip_all, err(Debug))]
async fn configure_local_models(
    State(state): State<AppState>,
    Json(configuration): Json<LocalModelsConfiguration>,
) -> Result<Json<LocalModelsConfiguration>, ApiError> {
    let result = local_models_domain(&state)?
        .configure(configuration)
        .await
        .map_err(ApiError::from_inventory)?;
    state.invalidate_resources([InferenceResourceTopic::LocalModels]);
    Ok(Json(result))
}

#[utoipa::path(get, path = "/api/v1/local-models", operation_id = "listLocalModels", tag = "local-models",
    responses(
        (status = 200, description = "Custom models on this computer, and the ones the engine cannot run", body = LocalModelsResponse),
        (status = 500, description = "Local models unavailable", body = ErrorResponse)
    )
)]
async fn local_models(
    State(state): State<AppState>,
) -> Result<Json<LocalModelsResponse>, ApiError> {
    local_models_domain(&state)?
        .list_local()
        .await
        .map(Json)
        .map_err(ApiError::from_inventory)
}

#[utoipa::path(post, path = "/api/v1/local-models/refresh", operation_id = "refreshLocalModels", tag = "local-models",
    responses(
        (status = 200, description = "Local models after a rescan of every enabled root", body = LocalModelsResponse),
        (status = 500, description = "Rescan failed", body = ErrorResponse)
    )
)]
async fn refresh_local_models(
    State(state): State<AppState>,
) -> Result<Json<LocalModelsResponse>, ApiError> {
    let result = local_models_domain(&state)?
        .refresh_local()
        .await
        .map_err(ApiError::from_inventory)?;
    state.invalidate_resources([InferenceResourceTopic::LocalModels]);
    Ok(Json(result))
}

#[utoipa::path(delete, path = "/api/v1/local-models/{model_id}/files", operation_id = "removeLocalModelFiles", tag = "local-models",
    params(("model_id" = String, Path, description = "Canonical custom model ID of an added model")),
    responses(
        (status = 200, description = "Magnitude's copy of the added model was removed; files in other locations are never touched", body = LocalModelRemoval),
        (status = 400, description = "Not a custom model ID", body = ErrorResponse),
        (status = 404, description = "Custom model not found", body = ErrorResponse),
        (status = 409, description = "The model is not an added model, so its files are never removed", body = ErrorResponse),
        (status = 500, description = "Removal failed", body = ErrorResponse)
    )
)]
async fn remove_local_model_files(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> Result<Json<LocalModelRemoval>, ApiError> {
    let model_id = model_id
        .parse::<ModelId>()
        .map_err(|error| ApiError::invalid(error.to_string()))?;
    if matches!(model_id.parsed(), ParsedModelId::Catalog { .. }) {
        return Err(ApiError::invalid(
            "local-model removal requires a custom model ID",
        ));
    }
    let result = local_models_domain(&state)?
        .remove_local(&model_id)
        .await
        .map_err(ApiError::from_inventory)?;
    state.invalidate_resources([InferenceResourceTopic::LocalModels]);
    Ok(Json(result))
}

#[utoipa::path(post, path = "/api/v1/local-models/drafters/prune", operation_id = "pruneLocalDrafters", tag = "local-models",
    responses(
        (status = 200, description = "Magnitude-owned drafter packages no model uses were removed; drafters in other locations are never touched", body = DrafterPruning),
        (status = 500, description = "Removal failed", body = ErrorResponse)
    )
)]
#[tracing::instrument(name = "icn.local_models.prune_drafters", skip_all, err(Debug))]
async fn prune_local_drafters(
    State(state): State<AppState>,
) -> Result<Json<DrafterPruning>, ApiError> {
    let result = local_models_domain(&state)?
        .prune_drafters()
        .await
        .map_err(ApiError::from_inventory)?;
    if !result.removed.is_empty() {
        state.invalidate_resources([InferenceResourceTopic::LocalModels]);
    }
    Ok(Json(result))
}

#[utoipa::path(post, path = "/api/v1/candidates/inspect", operation_id = "inspectModelCandidate", tag = "candidates",
    request_body(content = CandidateInspectionRequest, content_type = "application/json"),
    responses(
        (status = 200, description = "Every version of the candidate with the engine's verdict", body = CandidateInspection),
        (status = 400, description = "Invalid candidate", body = ErrorResponse),
        (status = 404, description = "Repository, revision or file not found", body = ErrorResponse),
        (status = 409, description = "The operation conflicts with the model's current state", body = ErrorResponse),
        (status = 422, description = "The repository's metadata or a file's header is invalid", body = ErrorResponse),
        (status = 500, description = "Inspection failed", body = ErrorResponse)
    )
)]
#[tracing::instrument(name = "icn.candidates.inspect", skip_all, err(Debug))]
async fn inspect_model_candidate(
    State(state): State<AppState>,
    Json(request): Json<CandidateInspectionRequest>,
) -> Result<Json<CandidateInspection>, ApiError> {
    model_candidates_domain(&state)?
        .inspect(request)
        .await
        .map(Json)
        .map_err(ApiError::from_inventory)
}

#[utoipa::path(post, path = "/api/v1/candidates/inspect/stream", operation_id = "streamModelCandidateInspection", tag = "candidates",
    request_body(content = CandidateInspectionRequest, content_type = "application/json"),
    responses(
        (status = 200, description = "The inspection's progress in order, then the inspection; a failure after the first event is an `error` event", body = String, content_type = "text/event-stream"),
        (status = 400, description = "Invalid candidate", body = ErrorResponse),
        (status = 404, description = "Repository, revision or file not found", body = ErrorResponse),
        (status = 409, description = "The operation conflicts with the model's current state", body = ErrorResponse),
        (status = 422, description = "The repository's metadata or a file's header is invalid", body = ErrorResponse),
        (status = 500, description = "Inspection failed", body = ErrorResponse)
    )
)]
#[tracing::instrument(name = "icn.candidates.inspect_stream", skip_all, err(Debug))]
async fn stream_model_candidate_inspection(
    State(state): State<AppState>,
    Json(request): Json<CandidateInspectionRequest>,
) -> Result<Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>>, ApiError> {
    let candidates = Arc::clone(model_candidates_domain(&state)?);
    let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel();
    let progress = CandidateInspectionProgressSink::new(move |progress| {
        let _ = sender.send(progress);
    });
    let mut inspection: BoxFuture<'static, Result<CandidateInspection, InventoryError>> =
        Box::pin(async move { candidates.inspect_with_progress(request, progress).await });
    // A failure before the first progress (a repository or revision that can't be resolved) is
    // the response's own status, as the request/response inspection reports it.
    let stage = tokio::select! {
        biased;
        Some(progress) = receiver.recv() => InspectionStage::Running { first: Some(progress), receiver, inspection },
        result = &mut inspection => match receiver.try_recv() {
            Ok(progress) => InspectionStage::Draining { first: Some(progress), receiver, result },
            Err(_) => match result {
                Ok(inspection) => InspectionStage::Draining { first: None, receiver, result: Ok(inspection) },
                Err(error) => return Err(ApiError::from_inventory(error)),
            },
        },
    };
    let events = futures_util::stream::unfold(stage, |stage| async move {
        match stage {
            InspectionStage::Running {
                first: Some(progress),
                receiver,
                inspection,
            } => Some((
                progress_event(progress),
                InspectionStage::Running {
                    first: None,
                    receiver,
                    inspection,
                },
            )),
            InspectionStage::Running {
                first: None,
                mut receiver,
                mut inspection,
            } => tokio::select! {
                biased;
                Some(progress) = receiver.recv() => Some((
                    progress_event(progress),
                    InspectionStage::Running { first: None, receiver, inspection },
                )),
                result = &mut inspection => {
                    // Progress reported in the same poll that finished the inspection comes first.
                    let first = receiver.try_recv().ok();
                    match first {
                        Some(progress) => Some((
                            progress_event(progress),
                            InspectionStage::Draining { first: None, receiver, result },
                        )),
                        None => Some((inspection_outcome_event(result), InspectionStage::Done)),
                    }
                },
            },
            InspectionStage::Draining {
                first,
                mut receiver,
                result,
            } => match first.or_else(|| receiver.try_recv().ok()) {
                Some(progress) => Some((
                    progress_event(progress),
                    InspectionStage::Draining {
                        first: None,
                        receiver,
                        result,
                    },
                )),
                None => Some((inspection_outcome_event(result), InspectionStage::Done)),
            },
            InspectionStage::Done => None,
        }
    });
    Ok(Sse::new(events).keep_alive(KeepAlive::default()))
}

/// The stream's progress through an inspection: running, then any progress still buffered once
/// it finished, then its outcome.
enum InspectionStage {
    Running {
        first: Option<CandidateInspectionProgress>,
        receiver: tokio::sync::mpsc::UnboundedReceiver<CandidateInspectionProgress>,
        inspection: BoxFuture<'static, Result<CandidateInspection, InventoryError>>,
    },
    Draining {
        first: Option<CandidateInspectionProgress>,
        receiver: tokio::sync::mpsc::UnboundedReceiver<CandidateInspectionProgress>,
        result: Result<CandidateInspection, InventoryError>,
    },
    Done,
}

fn progress_event(progress: CandidateInspectionProgress) -> Result<Event, Infallible> {
    Ok(Event::default().data(
        serde_json::to_string(&CandidateInspectionEvent::Progress { progress })
            .expect("inspection progress is serializable"),
    ))
}

/// The inspection, or its failure as an `error` event carrying the error the request/response
/// inspection would have returned.
fn inspection_outcome_event(
    result: Result<CandidateInspection, InventoryError>,
) -> Result<Event, Infallible> {
    Ok(match result {
        Ok(inspection) => Event::default().data(
            serde_json::to_string(&CandidateInspectionEvent::Inspected { inspection })
                .expect("inspection is serializable"),
        ),
        Err(error) => Event::default().event("error").data(
            serde_json::to_string(&ApiError::from_inventory(error).body)
                .expect("error response is serializable"),
        ),
    })
}

#[utoipa::path(post, path = "/api/v1/candidates/drafters", operation_id = "listDrafterCandidates", tag = "candidates",
    request_body(content = DrafterCandidatesRequest, content_type = "application/json"),
    responses(
        (status = 200, description = "Drafter candidates for the target, in category order", body = DrafterCandidatesResponse),
        (status = 400, description = "Invalid target", body = ErrorResponse),
        (status = 404, description = "Target not found", body = ErrorResponse),
        (status = 409, description = "The operation conflicts with the model's current state", body = ErrorResponse),
        (status = 422, description = "The repository's metadata or a file's header is invalid", body = ErrorResponse),
        (status = 500, description = "Enumeration failed", body = ErrorResponse)
    )
)]
#[tracing::instrument(name = "icn.candidates.drafters", skip_all, err(Debug))]
async fn drafter_candidates(
    State(state): State<AppState>,
    Json(request): Json<DrafterCandidatesRequest>,
) -> Result<Json<DrafterCandidatesResponse>, ApiError> {
    model_candidates_domain(&state)?
        .drafters(request)
        .await
        .map(Json)
        .map_err(ApiError::from_inventory)
}

#[utoipa::path(post, path = "/api/v1/candidates/bundle", operation_id = "buildCandidateBundle", tag = "candidates",
    request_body(content = ModelCandidate, content_type = "application/json"),
    responses(
        (status = 200, description = "The exact bundle a download of the candidate fetches; a candidate whose target is local yields its drafter alone", body = ServableModelBundle),
        (status = 400, description = "Invalid candidate", body = ErrorResponse),
        (status = 404, description = "Candidate file not found", body = ErrorResponse),
        (status = 409, description = "The operation conflicts with the model's current state", body = ErrorResponse),
        (status = 422, description = "The repository's metadata or a file's header is invalid", body = ErrorResponse),
        (status = 500, description = "Bundle construction failed", body = ErrorResponse)
    )
)]
#[tracing::instrument(name = "icn.candidates.bundle", skip_all, err(Debug))]
async fn candidate_bundle(
    State(state): State<AppState>,
    Json(candidate): Json<ModelCandidate>,
) -> Result<Json<ServableModelBundle>, ApiError> {
    model_candidates_domain(&state)?
        .bundle(candidate)
        .await
        .map(Json)
        .map_err(ApiError::from_inventory)
}

#[utoipa::path(post, path = "/api/v1/model-assessments/candidates", operation_id = "assessModelCandidate", tag = "models",
    request_body(content = ModelCandidate, content_type = "application/json"),
    responses(
        (status = 200, description = "The engine's header-only verdict for the candidate", body = ModelAssessment),
        (status = 400, description = "Invalid candidate", body = ErrorResponse),
        (status = 404, description = "Candidate file not found", body = ErrorResponse),
        (status = 409, description = "The operation conflicts with the model's current state", body = ErrorResponse),
        (status = 422, description = "The repository's metadata or a file's header is invalid", body = ErrorResponse),
        (status = 500, description = "Assessment failed", body = ErrorResponse)
    )
)]
#[tracing::instrument(name = "icn.candidates.assess", skip_all, err(Debug))]
async fn assess_model_candidate(
    State(state): State<AppState>,
    Json(candidate): Json<ModelCandidate>,
) -> Result<Json<ModelAssessment>, ApiError> {
    model_candidates_domain(&state)?
        .assess_candidate(candidate)
        .await
        .map(Json)
        .map_err(ApiError::from_inventory)
}

#[utoipa::path(get, path = "/api/v1/downloads", operation_id = "listModelDownloads", tag = "downloads",
    responses(
        (status = 200, description = "Model bundle downloads", body = ModelDownloadsResponse),
        (status = 500, description = "Downloads unavailable", body = ErrorResponse)
    )
)]
async fn model_downloads(
    State(state): State<AppState>,
) -> Result<Json<ModelDownloadsResponse>, ApiError> {
    model_downloads_domain(&state)?
        .list()
        .await
        .map(Json)
        .map_err(ApiError::from_inventory)
}

#[utoipa::path(post, path = "/api/v1/downloads", operation_id = "startModelDownload", tag = "downloads",
    request_body(content = StartModelDownloadRequest, content_type = "application/json"),
    responses(
        (status = 200, description = "The admitted download, absent when the bundle is already installed", body = StartModelDownloadResponse),
        (status = 400, description = "Invalid bundle", body = ErrorResponse),
        (status = 409, description = "Download cannot be admitted", body = ErrorResponse),
        (status = 500, description = "Download failed to start", body = ErrorResponse)
    )
)]
#[tracing::instrument(name = "icn.downloads.start", skip_all, err(Debug))]
async fn start_model_download(
    State(state): State<AppState>,
    Json(request): Json<StartModelDownloadRequest>,
) -> Result<Json<StartModelDownloadResponse>, ApiError> {
    let result = model_downloads_domain(&state)?
        .start(request)
        .await
        .map_err(ApiError::from_inventory)?;
    state.invalidate_resources([InferenceResourceTopic::Downloads]);
    Ok(Json(result))
}

#[utoipa::path(post, path = "/api/v1/downloads/{download_id}/cancel", operation_id = "cancelModelDownload", tag = "downloads",
    params(("download_id" = String, Path, description = "Model download ID")),
    responses(
        (status = 200, description = "The cancelled download", body = ModelDownload),
        (status = 404, description = "Download not found", body = ErrorResponse),
        (status = 500, description = "Cancellation failed", body = ErrorResponse)
    )
)]
async fn cancel_model_download(
    State(state): State<AppState>,
    Path(download_id): Path<String>,
) -> Result<Json<ModelDownload>, ApiError> {
    let result = model_downloads_domain(&state)?
        .cancel(&ModelDownloadId(download_id))
        .await
        .map_err(ApiError::from_inventory)?;
    state.invalidate_resources([InferenceResourceTopic::Downloads]);
    Ok(Json(result))
}

#[utoipa::path(post, path = "/api/v1/downloads/{download_id}/acknowledge-failure", operation_id = "acknowledgeModelDownloadFailure", tag = "downloads",
    params(("download_id" = String, Path, description = "Model download ID")),
    responses(
        (status = 200, description = "The download with its failure acknowledged", body = ModelDownload),
        (status = 404, description = "Download not found", body = ErrorResponse),
        (status = 409, description = "Download has not failed", body = ErrorResponse),
        (status = 500, description = "Acknowledgement failed", body = ErrorResponse)
    )
)]
async fn acknowledge_model_download_failure(
    State(state): State<AppState>,
    Path(download_id): Path<String>,
) -> Result<Json<ModelDownload>, ApiError> {
    let result = model_downloads_domain(&state)?
        .acknowledge_failure(&ModelDownloadId(download_id))
        .await
        .map_err(ApiError::from_inventory)?;
    state.invalidate_resources([InferenceResourceTopic::Downloads]);
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
    let local = local_models_domain(&state)?
        .list_local()
        .await
        .map_err(ApiError::from_inventory)?;
    let assessment_snapshot = state
        .model_assessments
        .as_ref()
        .ok_or_else(|| ApiError::server("model assessments are not configured"))?
        .snapshot()
        .await
        .map_err(ApiError::from_inventory)?;
    let offered_capabilities = offered_capabilities(&assessment_snapshot);
    let catalog_models = catalog.models.into_iter().filter_map(|model| {
        let CatalogModelState::Installed { effective, .. } = model.local_state else {
            return None;
        };
        let EffectiveModel::Ready { model: target } = effective else {
            return None;
        };
        let capabilities = offered_capabilities.get(&ModelAssessmentSubject::Catalog {
            model_id: model.id.clone(),
            selection: magnitude_service_contracts::models::CatalogModelSelection::Effective,
        })?;
        let name = format!("{} ({})", model.display_name, model.variant_label);
        let id = model.id.to_string();
        open_ai_model(
            id,
            "magnitude",
            name,
            model.description,
            target.profile.context_length,
            capabilities.clone(),
        )
    });
    let custom_models = local.models.into_iter().filter_map(|model| {
        let CustomModelState::Ready { model: target, .. } = &model.state else {
            return None;
        };
        let capabilities = offered_capabilities.get(&ModelAssessmentSubject::Custom {
            model_id: model.model_id.clone(),
        })?;
        let (owned_by, description) = custom_model_source(&model);
        open_ai_model(
            model.model_id.to_string(),
            owned_by,
            model.display_name,
            description,
            target.profile.context_length,
            capabilities.clone(),
        )
    });
    let data = catalog_models.chain(custom_models).collect();
    Ok(Json(OpenAiModelsResponse {
        object: "list",
        data,
    }))
}

/// The `owned_by` label and description of a custom model, which name where its files came from.
fn custom_model_source(model: &CustomModel) -> (&'static str, String) {
    let repository = match model.model_id.parsed() {
        ParsedModelId::HuggingFace { repository_id, .. } => Some(repository_id),
        ParsedModelId::Catalog { .. } | ParsedModelId::Local { .. } => None,
    };
    let from = |label: &str| match &repository {
        Some(repository) => format!("{label} ({repository})", repository = repository.as_str()),
        None => label.to_owned(),
    };
    match &model.origin {
        CustomModelOrigin::Added { repository, .. } => (
            "huggingface",
            format!("Added from Hugging Face ({})", repository.as_str()),
        ),
        CustomModelOrigin::Imported { path } => {
            ("local-file", format!("Added from {}", path.display()))
        }
        CustomModelOrigin::Found { source } => match source {
            FoundSource::HuggingFaceCache => {
                ("huggingface-cache", from("Found in the Hugging Face cache"))
            }
            FoundSource::LmStudio => ("lm-studio", from("Found in LM Studio")),
            FoundSource::LlamaCpp => ("llama-cpp", from("Found in the llama.cpp cache")),
            FoundSource::Ollama => ("ollama", from("Found in Ollama")),
            FoundSource::Folder { path } => ("folder", format!("Found in {}", path.display())),
        },
    }
}

/// The capabilities of every offered model: one whose assessment is `Fits`. A model that does not
/// fit or that the engine cannot run is never offered for inference.
fn offered_capabilities(
    snapshot: &ModelAssessmentsSnapshot,
) -> BTreeMap<ModelAssessmentSubject, ModelCapabilities> {
    [&snapshot.catalog, &snapshot.custom]
        .into_iter()
        .filter_map(|domain| match domain {
            ModelAssessmentDomainSnapshot::Available { entries, .. } => Some(entries),
            ModelAssessmentDomainSnapshot::Pending { .. } => None,
        })
        .flatten()
        .filter_map(|entry| match &entry.state {
            ModelAssessmentEntryState::Assessed {
                capabilities,
                profiles,
                ..
            } if matches!(profiles.first(), Some(ModelAssessment::Fits { .. })) => {
                Some((entry.subject.clone(), capabilities.clone()))
            }
            ModelAssessmentEntryState::Assessed { .. }
            | ModelAssessmentEntryState::Assessing
            | ModelAssessmentEntryState::Dropped => None,
        })
        .collect()
}

/// An offered model's OpenAI listing entry. A model that supports reasoning but has no default
/// effort is an incomplete assessment: it is left out of the listing rather than taking down every
/// other model's entry.
fn open_ai_model(
    id: String,
    owned_by: &'static str,
    name: String,
    description: String,
    context_length: u32,
    capabilities: magnitude_service_contracts::models::ModelCapabilities,
) -> Option<OpenAiModel> {
    let mut supported_parameters = vec!["max_tokens"];
    if capabilities.tools {
        supported_parameters.extend(["tools", "tool_choice"]);
    }
    if capabilities.structured_output {
        supported_parameters.extend(["structured_outputs", "response_format"]);
    }
    let reasoning = match (
        capabilities.reasoning.supported,
        capabilities.reasoning.default_effort,
    ) {
        (false, _) => None,
        (true, Some(default_effort)) => {
            supported_parameters.push("reasoning");
            Some(OpenAiModelReasoning {
                default_enabled: default_effort != "none",
                mandatory: !capabilities
                    .reasoning
                    .efforts
                    .iter()
                    .any(|effort| effort == "none"),
                supported_efforts: capabilities.reasoning.efforts,
                default_effort,
            })
        }
        (true, None) => return None,
    };
    let mut input_modalities = vec!["text"];
    if capabilities.vision {
        input_modalities.push("image");
    }
    Some(OpenAiModel {
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
    })
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
        install_catalog_component,
        remove_catalog_model_installation,
        catalog_installations,
        catalog_installation,
        cancel_catalog_installation,
        acknowledge_catalog_installation_failure,
        local_models_configuration,
        configure_local_models,
        local_models,
        refresh_local_models,
        remove_local_model_files,
        prune_local_drafters,
        inspect_model_candidate,
        stream_model_candidate_inspection,
        drafter_candidates,
        candidate_bundle,
        model_assessments,
        assess_model_candidate,
        model_downloads,
        start_model_download,
        cancel_model_download,
        acknowledge_model_download_failure,
        standard_models,
        search_hugging_face_models,
        resolve_hugging_face_repository,
        configure_hugging_face_credential,
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
        LocalModelsConfiguration,
        LocalModelsResponse,
        LocalModelRemoval,
        DrafterPruning,
        CandidateInspectionRequest,
        CandidateInspection,
        DrafterCandidatesRequest,
        DrafterCandidatesResponse,
        ModelCandidate,
        ModelAssessment,
        ModelDownloadsResponse,
        StartModelDownloadRequest,
        StartModelDownloadResponse,
        ModelDownload,
        ModelAssessmentsSnapshot,
        CatalogInstallationsResponse,
        CatalogInstallationAdmission,
        CatalogInstallationRemoval,
        CatalogComponent,
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
        HuggingFaceCredential,
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

struct CandidateInspectionStream;

impl StreamContract for CandidateInspectionStream {
    type Event = CandidateInspectionEvent;
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
    attach_stream_contract::<CandidateInspectionStream>(
        &mut document,
        "streamModelCandidateInspection",
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
        )
        .expect("listed model");

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

    #[derive(Default)]
    struct StubHuggingFaceCatalog {
        credentials: std::sync::Mutex<Vec<Option<String>>>,
    }

    impl HuggingFaceModelCatalog for StubHuggingFaceCatalog {
        fn configure_credential(
            &self,
            credential: HuggingFaceCredential,
        ) -> BoxFuture<'_, Result<(), InventoryError>> {
            self.credentials.lock().unwrap().push(credential.token);
            Box::pin(async { Ok(()) })
        }

        fn search(
            &self,
            request: HuggingFaceModelSearchRequest,
        ) -> BoxFuture<'_, Result<HuggingFaceModelSearchResults, InventoryError>> {
            Box::pin(async move {
                Ok(HuggingFaceModelSearchResults {
                    results: vec![magnitude_service_contracts::HuggingFaceModelSearchResult {
                        repository: format!("owner/{}", request.query),
                        commit: "a".repeat(40),
                        last_modified: None,
                        downloads: Some(10),
                        likes: Some(2),
                        gated: false,
                        private: false,
                        tags: vec!["gguf".to_owned()],
                        architecture: Some("qwen35moe".to_owned()),
                        variant_of: Some("qwen3.6-35b-a3b:gguf:q4".parse().unwrap()),
                        version_count: Some(3),
                        unsupported: Vec::new(),
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
        let state = AppState::new().with_hugging_face_catalog(Arc::new(StubHuggingFaceCatalog::default()));
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
        assert_eq!(search_body["results"][0]["repository"], "owner/model");
        assert_eq!(
            search_body["results"][0]["variantOf"],
            "qwen3.6-35b-a3b:gguf:q4"
        );
        assert!(search_body["results"][0].get("lastModified").is_none());

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
        assert_eq!(resolve_body["ggufFiles"][0]["sizeBytes"], 123);
    }

    #[tokio::test]
    async fn the_hugging_face_credential_is_write_only_and_absent_means_anonymous() {
        let catalog = Arc::new(StubHuggingFaceCatalog::default());
        let state = AppState::new().with_hugging_face_catalog(catalog.clone());
        for body in [json!({ "token": "hf_secret" }), json!({})] {
            let response = test_app(state.clone())
                .oneshot(json_request(
                    "PUT",
                    "/api/v1/sources/hugging-face/credential",
                    body,
                ))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::NO_CONTENT);
            let bytes = response.into_body().collect().await.unwrap().to_bytes();
            assert!(bytes.is_empty());
        }
        assert_eq!(
            *catalog.credentials.lock().unwrap(),
            vec![Some("hf_secret".to_owned()), None]
        );
        let document = serde_json::to_string(&openapi().unwrap()).unwrap();
        assert!(!document.contains("hf_secret"));
        let value: Value = serde_json::from_str(&document).unwrap();
        let operation = &value["paths"]["/api/v1/sources/hugging-face/credential"]["put"];
        assert_eq!(operation["operationId"], "configureHuggingFaceCredential");
        assert!(operation["responses"]["204"]["content"].is_null());
        let schema = &value["components"]["schemas"]["HuggingFaceCredential"];
        assert_eq!(schema["properties"]["token"]["type"], "string");
        assert!(schema.get("required").is_none_or(|required| required
            .as_array()
            .is_some_and(Vec::is_empty)));
    }

    #[tokio::test]
    async fn an_empty_hugging_face_token_is_rejected() {
        let catalog = Arc::new(StubHuggingFaceCatalog::default());
        let state = AppState::new().with_hugging_face_catalog(catalog.clone());
        for token in ["", "   "] {
            let (status, body) = send(
                test_app(state.clone()),
                json_request(
                    "PUT",
                    "/api/v1/sources/hugging-face/credential",
                    json!({ "token": token }),
                ),
            )
            .await;
            assert_eq!(status, StatusCode::BAD_REQUEST, "{token:?}");
            assert_eq!(body["error"]["code"], "invalid_request");
        }
        assert!(catalog.credentials.lock().unwrap().is_empty());
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
            Box::pin(async move { Ok(self.instances().await?.instances.remove(0)) })
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
        let response =
            test_app(AppState::new().with_model_controller(Arc::new(StubModelInstanceController)))
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
            environment_id: magnitude_service_contracts::models::AssessmentEnvironmentId(
                "environment".to_owned(),
            ),
            catalog: ModelAssessmentDomainSnapshot::Pending { source_revision: 0 },
            custom: ModelAssessmentDomainSnapshot::Pending { source_revision: 0 },
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
        assert_eq!(body["environmentId"], "environment");
        assert_eq!(body["catalog"]["_tag"], "Pending");
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
    fn exported_candidate_operations_declare_every_inventory_error_status() {
        let value = serde_json::to_value(openapi().unwrap()).unwrap();
        for path in [
            "/api/v1/sources/hugging-face/search",
            "/api/v1/sources/hugging-face/resolve",
            "/api/v1/candidates/inspect",
            "/api/v1/candidates/drafters",
            "/api/v1/candidates/bundle",
            "/api/v1/model-assessments/candidates",
        ] {
            let responses = &value["paths"][path]["post"]["responses"];
            for status in ["400", "404", "409", "422", "500"] {
                assert!(
                    responses[status].is_object(),
                    "post {path} must declare inventory error status {status}: an undeclared one reaches clients as an invalid response, not the service's message",
                );
            }
        }
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

    use magnitude_service_contracts::models::{
        CandidateFile, CandidateSource, CatalogInstallationAdmission, CatalogInstallationRemoval,
        CustomModelCatalogAttribution, CustomModelFile, DrafterCandidate, DrafterCategory,
        DrafterLineage, DrafterReference, DrafterVerdict, KnownUnsupportedReason,
        ModelDomainInvalidation, ModelDownloadState, ModelDownloadsInvalidation, ModelMetadata,
        ModelPackage, ModelPackageId, ModelPackageProperties, ModelPackageSource, ReadyModel,
        ResolvedModelInstallation, ServableModelBundle, ServingProfile, SpeculativeMethod,
        UnsupportedLocalModel, UnsupportedReason,
    };

    fn local_model_id() -> ModelId {
        format!("local:{}", "ab".repeat(32)).parse().unwrap()
    }

    fn custom_model(model_id: ModelId, origin: CustomModelOrigin) -> CustomModel {
        CustomModel {
            model_id,
            origin,
            display_name: "Fine Tune".to_owned(),
            variant_label: Some("Q4_K_M".to_owned()),
            variant_of: Some("qwen3.6-35b-a3b:gguf:q4".parse().unwrap()),
            files: vec![CustomModelFile {
                path: "/models/fine-tune.gguf".into(),
                size_bytes: 100,
            }],
            size_bytes: 100,
            projector: None,
            drafter: None,
            drafter_pending: false,
            recommended_drafter: None,
            license: None,
            profile: ServingProfile {
                context_length: 8192,
            },
            update_commit: None,
            state: CustomModelState::Ready {
                installation: ResolvedModelInstallation::Resolved {
                    installed_bytes: 100,
                    primary_path: "/models/fine-tune.gguf".into(),
                    ownership:
                        magnitude_service_contracts::models::ModelInstallationOwnership::External,
                },
                model: ReadyModel {
                    metadata: ModelMetadata {
                        format: "gguf".to_owned(),
                        architecture: "qwen35moe".to_owned(),
                        quantization: "Q4_K_M".to_owned(),
                        quantization_name: "4-bit".to_owned(),
                        storage_bytes: 100,
                        maximum_context_length: Some(8192),
                    },
                    profile: ServingProfile {
                        context_length: 8192,
                    },
                    speculative_method: None,
                },
                catalog_attribution: CustomModelCatalogAttribution::NotInCatalog,
            },
        }
    }

    struct StubLocalModels {
        configuration: std::sync::Mutex<LocalModelsConfiguration>,
    }

    impl StubLocalModels {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                configuration: std::sync::Mutex::new(LocalModelsConfiguration::default()),
            })
        }

        fn response(&self) -> LocalModelsResponse {
            LocalModelsResponse {
                revision: 4,
                reconciliation_complete: true,
                models: vec![custom_model(
                    local_model_id(),
                    CustomModelOrigin::Found {
                        source: FoundSource::Ollama,
                    },
                )],
                hidden: Vec::new(),
                unsupported: vec![UnsupportedLocalModel {
                    display_name: "DeepSeek V4 Flash".to_owned(),
                    origin: CustomModelOrigin::Found {
                        source: FoundSource::LmStudio,
                    },
                    path: "/lmstudio/deepseek.gguf".into(),
                    reasons: vec![UnsupportedReason::Known(
                        KnownUnsupportedReason::Architecture {
                            architecture: "deepseek4".to_owned(),
                        },
                    )],
                    detail: "unsupported model family".to_owned(),
                }],
            }
        }
    }

    impl LocalModels for StubLocalModels {
        fn configuration(&self) -> BoxFuture<'_, Result<LocalModelsConfiguration, InventoryError>> {
            Box::pin(async { Ok(self.configuration.lock().unwrap().clone()) })
        }

        fn configure(
            &self,
            configuration: LocalModelsConfiguration,
        ) -> BoxFuture<'_, Result<LocalModelsConfiguration, InventoryError>> {
            Box::pin(async move {
                *self.configuration.lock().unwrap() = configuration.clone();
                Ok(configuration)
            })
        }

        fn list_local(&self) -> BoxFuture<'_, Result<LocalModelsResponse, InventoryError>> {
            Box::pin(async { Ok(self.response()) })
        }

        fn refresh_local(&self) -> BoxFuture<'_, Result<LocalModelsResponse, InventoryError>> {
            Box::pin(async { Ok(self.response()) })
        }

        fn remove_local(
            &self,
            model_id: &ModelId,
        ) -> BoxFuture<'_, Result<LocalModelRemoval, InventoryError>> {
            let model_id = model_id.clone();
            Box::pin(async move {
                if model_id == local_model_id() {
                    return Err(InventoryError::Unsupported(format!(
                        "{model_id} is not an added model"
                    )));
                }
                Ok(LocalModelRemoval {
                    model_id,
                    removed: true,
                    freed_bytes: 7,
                })
            })
        }

        fn prune_drafters(&self) -> BoxFuture<'_, Result<DrafterPruning, InventoryError>> {
            Box::pin(async {
                Ok(DrafterPruning {
                    removed: Vec::new(),
                    freed_bytes: 0,
                })
            })
        }

        fn watch_local(&self) -> BoxStream<'static, ModelDomainInvalidation> {
            Box::pin(futures_util::stream::empty())
        }
    }

    fn hugging_face_source() -> CandidateSource {
        CandidateSource::HuggingFace {
            repository: magnitude_service_contracts::models::HuggingFaceRepositoryId::new(
                "owner/repo",
            )
            .unwrap(),
            commit: "c".repeat(40),
        }
    }

    fn fits_assessment() -> ModelAssessment {
        ModelAssessment::Fits {
            profile: ServingProfile {
                context_length: 8192,
            },
            assessment_id: magnitude_service_contracts::models::ModelAssessmentId(
                "assessment".to_owned(),
            ),
            memory: Vec::new(),
            performance: Vec::new(),
            projector: magnitude_service_contracts::models::ProjectorVerdict::None,
            drafter: DrafterVerdict::Bound {
                method: SpeculativeMethod::DFlash,
                memory_bytes: 900,
            },
        }
    }

    struct StubModelCandidates;

    impl ModelCandidates for StubModelCandidates {
        fn inspect_with_progress(
            &self,
            request: CandidateInspectionRequest,
            progress: CandidateInspectionProgressSink,
        ) -> BoxFuture<'_, Result<CandidateInspection, InventoryError>> {
            Box::pin(async move {
                let CandidateInspectionRequest::HuggingFace { revision, .. } = request else {
                    return Err(InventoryError::NotFound("local candidate".to_owned()));
                };
                let row = |assessment| magnitude_service_contracts::models::CandidateVersionProgress {
                    label: "Q4_K_M".to_owned(),
                    size_bytes: 100,
                    assessment,
                };
                progress.report(CandidateInspectionProgress {
                    phase: magnitude_service_contracts::models::CandidateInspectionPhase::ReadingHeaders {
                        completed: 1,
                        total: 1,
                    },
                    versions: vec![row(None)],
                });
                if revision == "fails-late" {
                    return Err(InventoryError::Internal("header reads failed".to_owned()));
                }
                progress.report(CandidateInspectionProgress {
                    phase: magnitude_service_contracts::models::CandidateInspectionPhase::CheckingVersions {
                        completed: 1,
                        total: 1,
                    },
                    versions: vec![row(Some(fits_assessment()))],
                });
                Ok(CandidateInspection {
                    source: hugging_face_source(),
                    display_name: "Repo".to_owned(),
                    license: Some("apache-2.0".to_owned()),
                    variant_of: None,
                    versions: vec![magnitude_service_contracts::models::CandidateVersion {
                        label: "Q4_K_M".to_owned(),
                        model_id: "hf:owner/repo/repo-Q4_K_M.gguf".parse().unwrap(),
                        files: vec![CandidateFile {
                            path: "repo-Q4_K_M.gguf".to_owned(),
                            size_bytes: 100,
                        }],
                        size_bytes: 100,
                        download_bytes: 100,
                        assessment: Some(fits_assessment()),
                        recommended: true,
                        catalog_model: None,
                        on_this_computer: None,
                    }],
                    projector: None,
                    unsupported_architecture: None,
                })
            })
        }

        fn drafters(
            &self,
            _request: DrafterCandidatesRequest,
        ) -> BoxFuture<'_, Result<DrafterCandidatesResponse, InventoryError>> {
            Box::pin(async {
                Ok(DrafterCandidatesResponse {
                    candidates: vec![DrafterCandidate {
                        category: DrafterCategory::CatalogDrafter,
                        reference: DrafterReference::Catalog {
                            model_id: "qwen3.6-35b-a3b:gguf:q4".parse().unwrap(),
                        },
                        name: "Magnitude DFlash for Qwen3.6 35B-A3B".to_owned(),
                        label: "Trained for Qwen3.6 35B-A3B".to_owned(),
                        size_bytes: 900,
                        lineage: DrafterLineage::Lineage,
                        verdict: DrafterVerdict::Bound {
                            method: SpeculativeMethod::DFlash,
                            memory_bytes: 900,
                        },
                        automatic: true,
                        on_this_computer: false,
                        method: Some(SpeculativeMethod::DFlash),
                    }],
                    entered: None,
                })
            })
        }

        fn assess_candidate(
            &self,
            _candidate: ModelCandidate,
        ) -> BoxFuture<'_, Result<ModelAssessment, InventoryError>> {
            Box::pin(async { Ok(fits_assessment()) })
        }

        fn bundle(
            &self,
            _candidate: ModelCandidate,
        ) -> BoxFuture<'_, Result<ServableModelBundle, InventoryError>> {
            Box::pin(async { Ok(test_bundle()) })
        }
    }

    fn test_bundle() -> ServableModelBundle {
        ServableModelBundle::Standalone {
            package: ModelPackage {
                id: ModelPackageId("package".to_owned()),
                source: ModelPackageSource::HuggingFace {
                    repository: "owner/repo".to_owned(),
                    revision: "c".repeat(40),
                },
                files: Vec::new(),
                relationships: Vec::new(),
                properties: ModelPackageProperties {
                    format: "gguf".to_owned(),
                    quantization: "Q4_K_M".to_owned(),
                    quantization_name: "4-bit".to_owned(),
                    architecture: "qwen35moe".to_owned(),
                    maximum_context_length: None,
                    intrinsic_model_id: None,
                    intrinsic_quality_id: None,
                },
            },
        }
    }

    struct StubModelDownloads;

    impl StubModelDownloads {
        fn download(id: &ModelDownloadId, state: ModelDownloadState) -> ModelDownload {
            ModelDownload {
                id: id.clone(),
                bundle: test_bundle(),
                for_model: None,
                state,
            }
        }
    }

    impl ModelDownloads for StubModelDownloads {
        fn start(
            &self,
            request: StartModelDownloadRequest,
        ) -> BoxFuture<'_, Result<StartModelDownloadResponse, InventoryError>> {
            Box::pin(async move {
                Ok(StartModelDownloadResponse {
                    download: Some(ModelDownload {
                        id: ModelDownloadId("download-1".to_owned()),
                        bundle: request.bundle,
                        for_model: request.for_model,
                        state: ModelDownloadState::Pending {
                            completed_bytes: 0,
                            total_bytes: 100,
                        },
                    }),
                })
            })
        }

        fn list(&self) -> BoxFuture<'_, Result<ModelDownloadsResponse, InventoryError>> {
            Box::pin(async {
                Ok(ModelDownloadsResponse {
                    downloads: vec![Self::download(
                        &ModelDownloadId("download-1".to_owned()),
                        ModelDownloadState::Completed,
                    )],
                })
            })
        }

        fn cancel(
            &self,
            id: &ModelDownloadId,
        ) -> BoxFuture<'_, Result<ModelDownload, InventoryError>> {
            let id = id.clone();
            Box::pin(async move {
                if id.0 != "download-1" {
                    return Err(InventoryError::NotFound(id.0));
                }
                Ok(Self::download(
                    &id,
                    ModelDownloadState::Cancelled {
                        completed_bytes: 10,
                        total_bytes: 100,
                    },
                ))
            })
        }

        fn acknowledge_failure(
            &self,
            id: &ModelDownloadId,
        ) -> BoxFuture<'_, Result<ModelDownload, InventoryError>> {
            let id = id.clone();
            Box::pin(async move {
                Ok(Self::download(
                    &id,
                    ModelDownloadState::Failed {
                        completed_bytes: 10,
                        total_bytes: 100,
                        failure: magnitude_service_contracts::DownloadFailure::Interrupted,
                        acknowledged: true,
                    },
                ))
            })
        }

        fn watch(&self) -> BoxStream<'static, ModelDownloadsInvalidation> {
            Box::pin(futures_util::stream::empty())
        }
    }

    struct EmptyCatalog;

    impl CatalogModels for EmptyCatalog {
        fn list_catalog(&self) -> BoxFuture<'_, Result<CatalogModelsResponse, InventoryError>> {
            Box::pin(async {
                Ok(CatalogModelsResponse {
                    revision: 0,
                    reconciliation_complete: true,
                    models: Vec::new(),
                })
            })
        }

        fn install_catalog_model(
            &self,
            model_id: &ModelId,
        ) -> BoxFuture<'_, Result<CatalogInstallationAdmission, InventoryError>> {
            let model_id = model_id.to_string();
            Box::pin(async move { Err(InventoryError::NotFound(model_id)) })
        }

        fn install_catalog_component(
            &self,
            model_id: &ModelId,
            _component: CatalogComponent,
        ) -> BoxFuture<'_, Result<CatalogInstallationAdmission, InventoryError>> {
            let model_id = model_id.to_string();
            Box::pin(async move { Err(InventoryError::NotFound(model_id)) })
        }

        fn remove_catalog_model_installation(
            &self,
            model_id: &ModelId,
        ) -> BoxFuture<'_, Result<CatalogInstallationRemoval, InventoryError>> {
            let model_id = model_id.to_string();
            Box::pin(async move { Err(InventoryError::NotFound(model_id)) })
        }

        fn watch_catalog(&self) -> BoxStream<'static, ModelDomainInvalidation> {
            Box::pin(futures_util::stream::empty())
        }
    }

    struct NoInstallations;

    impl CatalogInstallations for NoInstallations {
        fn watch_catalog_installations(
            &self,
        ) -> BoxStream<'static, magnitude_service_contracts::models::CatalogInstallationsInvalidation>
        {
            Box::pin(futures_util::stream::empty())
        }

        fn list_catalog_installations(
            &self,
        ) -> BoxFuture<'_, Result<CatalogInstallationsResponse, InventoryError>> {
            Box::pin(async {
                Ok(CatalogInstallationsResponse {
                    operations: Vec::new(),
                })
            })
        }

        fn cancel_catalog_installation(
            &self,
            id: &CatalogInstallationOperationId,
        ) -> BoxFuture<'_, Result<CatalogInstallationOperation, InventoryError>> {
            let id = id.0.clone();
            Box::pin(async move { Err(InventoryError::NotFound(id)) })
        }

        fn acknowledge_catalog_installation_failure(
            &self,
            id: &CatalogInstallationOperationId,
        ) -> BoxFuture<'_, Result<CatalogInstallationOperation, InventoryError>> {
            let id = id.0.clone();
            Box::pin(async move { Err(InventoryError::NotFound(id)) })
        }
    }

    fn local_models_state(local: Arc<StubLocalModels>) -> AppState {
        AppState::new().with_model_domains(Arc::new(EmptyCatalog), local, Arc::new(NoInstallations))
    }

    async fn send(app: Router, request: Request<Body>) -> (StatusCode, Value) {
        let response = app.oneshot(request).await.unwrap();
        let status = response.status();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        let body = serde_json::from_slice(&bytes)
            .unwrap_or_else(|_| Value::String(String::from_utf8_lossy(&bytes).into_owned()));
        (status, body)
    }

    fn json_request(method: &str, uri: &str, body: Value) -> Request<Body> {
        Request::builder()
            .method(method)
            .uri(uri)
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    #[tokio::test]
    async fn local_models_configuration_is_replaced_whole_and_read_back() {
        let state = local_models_state(StubLocalModels::new());
        let mut changes = state.resource_changes.subscribe();
        let configuration = json!({
            "roots": [{ "kind": "Ollama", "path": "/home/u/.ollama/models", "enabled": true }],
            "declarations": [],
            "hidden": [local_model_id()],
            "drafters": [{ "modelId": local_model_id(), "drafter": { "_tag": "None" } }]
        });
        let (status, body) = send(
            test_app(state.clone()),
            json_request(
                "PUT",
                "/api/v1/local-models/configuration",
                configuration.clone(),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, configuration);
        assert_eq!(
            changes.recv().await.unwrap().topic,
            InferenceResourceTopic::LocalModels
        );

        let (status, body) = send(
            test_app(state),
            Request::get("/api/v1/local-models/configuration")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body, configuration);
    }

    #[tokio::test]
    async fn local_models_configuration_rejects_an_invalid_snapshot() {
        let (status, _) = send(
            test_app(local_models_state(StubLocalModels::new())),
            json_request(
                "PUT",
                "/api/v1/local-models/configuration",
                json!({ "roots": [], "declarations": [], "hidden": ["local:short"], "drafters": [] }),
            ),
        )
        .await;
        assert!(status.is_client_error());
    }

    #[tokio::test]
    async fn local_models_list_and_refresh_return_custom_and_unsupported_models() {
        let state = local_models_state(StubLocalModels::new());
        let (status, body) = send(
            test_app(state.clone()),
            Request::get("/api/v1/local-models")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["models"][0]["modelId"], local_model_id().to_string());
        assert_eq!(body["models"][0]["origin"]["source"]["_tag"], "Ollama");
        assert_eq!(
            body["unsupported"][0]["reasons"][0],
            json!({ "_tag": "Architecture", "architecture": "deepseek4" })
        );

        let mut changes = state.resource_changes.subscribe();
        let (status, body) = send(
            test_app(state),
            Request::post("/api/v1/local-models/refresh")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["revision"], 4);
        assert_eq!(
            changes.recv().await.unwrap().topic,
            InferenceResourceTopic::LocalModels
        );
    }

    #[tokio::test]
    async fn local_model_file_removal_reaches_only_added_models() {
        let state = local_models_state(StubLocalModels::new());
        let added = "hf:owner/repo/model-Q4_K_M.gguf";
        let (status, body) = send(
            test_app(state.clone()),
            Request::delete(format!(
                "/api/v1/local-models/{}/files",
                added.replace('/', "%2F")
            ))
            .body(Body::empty())
            .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            body,
            json!({ "modelId": added, "removed": true, "freedBytes": 7 })
        );

        let (status, _) = send(
            test_app(state.clone()),
            Request::delete(format!(
                "/api/v1/local-models/{}/files",
                local_model_id().to_string().replace('/', "%2F")
            ))
            .body(Body::empty())
            .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::CONFLICT);

        let (status, _) = send(
            test_app(state),
            Request::delete("/api/v1/local-models/qwen3.6-35b-a3b:gguf:q4/files")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn local_model_routes_report_an_unconfigured_domain() {
        let (status, body) = send(
            test_app(AppState::new()),
            Request::get("/api/v1/local-models")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(body["error"]["message"], "local models are not configured");
    }

    async fn inspection_stream(
        state: AppState,
        request: serde_json::Value,
    ) -> (StatusCode, Vec<(Option<String>, serde_json::Value)>) {
        let response = test_app(state)
            .oneshot(json_request("POST", "/api/v1/candidates/inspect/stream", request))
            .await
            .unwrap();
        let status = response.status();
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let text = String::from_utf8(body.to_vec()).unwrap();
        if status != StatusCode::OK {
            return (status, vec![(None, serde_json::from_str(&text).unwrap())]);
        }
        let events = text
            .split("\n\n")
            .filter(|frame| frame.contains("data:"))
            .map(|frame| {
                let name = frame
                    .lines()
                    .find_map(|line| line.strip_prefix("event:"))
                    .map(|name| name.trim().to_owned());
                let data = frame
                    .lines()
                    .find_map(|line| line.strip_prefix("data:"))
                    .unwrap()
                    .trim();
                (name, serde_json::from_str(data).unwrap())
            })
            .collect();
        (status, events)
    }

    #[tokio::test]
    async fn inspection_stream_reports_progress_in_order_then_the_inspection() {
        let state = AppState::new().with_model_candidates(Arc::new(StubModelCandidates));
        let (status, events) = inspection_stream(
            state.clone(),
            json!({ "_tag": "HuggingFace", "repository": "owner/repo", "revision": "main" }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let tags = events
            .iter()
            .map(|(_, event)| event["_tag"].as_str().unwrap().to_owned())
            .collect::<Vec<_>>();
        assert_eq!(tags, ["Progress", "Progress", "Inspected"]);
        assert_eq!(events[0].1["progress"]["phase"]["_tag"], "ReadingHeaders");
        assert!(events[0].1["progress"]["versions"][0].get("assessment").is_none());
        assert_eq!(events[1].1["progress"]["phase"]["_tag"], "CheckingVersions");
        assert_eq!(events[1].1["progress"]["phase"]["completed"], 1);
        assert_eq!(events[1].1["progress"]["versions"][0]["assessment"]["_tag"], "Fits");
        assert_eq!(events[2].1["inspection"]["versions"][0]["label"], "Q4_K_M");

        // A failure before any progress is the response's status; one after it, an error event.
        let (status, events) =
            inspection_stream(state.clone(), json!({ "_tag": "Local", "path": "/missing.gguf" }))
                .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert!(events[0].1["error"]["message"].is_string());
        let (status, events) = inspection_stream(
            state,
            json!({ "_tag": "HuggingFace", "repository": "owner/repo", "revision": "fails-late" }),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(events.len(), 2);
        assert_eq!(events[1].0.as_deref(), Some("error"));
        assert_eq!(events[1].1["error"]["type"], "server_error");
    }

    #[tokio::test]
    async fn candidate_routes_inspect_list_drafters_and_assess() {
        let state = AppState::new().with_model_candidates(Arc::new(StubModelCandidates));
        let (status, body) = send(
            test_app(state.clone()),
            json_request(
                "POST",
                "/api/v1/candidates/inspect",
                json!({ "_tag": "HuggingFace", "repository": "owner/repo", "revision": "main" }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["versions"][0]["label"], "Q4_K_M");
        assert_eq!(body["versions"][0]["recommended"], true);
        assert_eq!(
            body["versions"][0]["assessment"]["drafter"]["_tag"],
            "Bound"
        );
        assert!(body.get("unsupportedArchitecture").is_none());

        let (status, _) = send(
            test_app(state.clone()),
            json_request(
                "POST",
                "/api/v1/candidates/inspect",
                json!({ "_tag": "Local", "path": "/missing.gguf" }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);

        let candidate = json!({
            "source": { "_tag": "HuggingFace", "repository": "owner/repo", "commit": "c".repeat(40) },
            "target": ["repo-Q4_K_M.gguf"]
        });
        let (status, body) = send(
            test_app(state.clone()),
            json_request(
                "POST",
                "/api/v1/candidates/drafters",
                json!({ "_tag": "Candidate", "candidate": candidate.clone() }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["candidates"][0]["category"], "CatalogDrafter");
        assert_eq!(body["candidates"][0]["lineage"], "Lineage");
        assert_eq!(body["candidates"][0]["onThisComputer"], false);

        let (status, body) = send(
            test_app(state.clone()),
            json_request(
                "POST",
                "/api/v1/candidates/drafters",
                json!({ "_tag": "Model", "modelId": local_model_id() }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["candidates"].as_array().unwrap().len(), 1);

        let (status, body) = send(
            test_app(state.clone()),
            json_request("POST", "/api/v1/candidates/bundle", candidate.clone()),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["_tag"], "Standalone");

        let (status, body) = send(
            test_app(state),
            json_request("POST", "/api/v1/model-assessments/candidates", candidate),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["_tag"], "Fits");
        assert_eq!(body["drafter"]["memoryBytes"], 900);
    }

    #[tokio::test]
    async fn candidate_routes_report_an_unconfigured_domain() {
        let (status, body) = send(
            test_app(AppState::new()),
            json_request(
                "POST",
                "/api/v1/candidates/inspect",
                json!({ "_tag": "Local", "path": "/model.gguf" }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR);
        assert_eq!(
            body["error"]["message"],
            "model candidates are not configured"
        );
    }

    #[tokio::test]
    async fn download_routes_list_start_cancel_and_acknowledge() {
        let state = AppState::new().with_model_downloads(Arc::new(StubModelDownloads));
        let mut changes = state.resource_changes.subscribe();

        let (status, body) = send(
            test_app(state.clone()),
            Request::get("/api/v1/downloads")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["downloads"][0]["state"]["_tag"], "Completed");

        let (status, body) = send(
            test_app(state.clone()),
            json_request(
                "POST",
                "/api/v1/downloads",
                json!({ "bundle": serde_json::to_value(test_bundle()).unwrap() }),
            ),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["download"]["id"], "download-1");
        assert_eq!(body["download"]["state"]["totalBytes"], 100);
        assert_eq!(
            changes.recv().await.unwrap().topic,
            InferenceResourceTopic::Downloads
        );

        let (status, body) = send(
            test_app(state.clone()),
            Request::post("/api/v1/downloads/download-1/cancel")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["state"]["_tag"], "Cancelled");

        let (status, _) = send(
            test_app(state.clone()),
            Request::post("/api/v1/downloads/unknown/cancel")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::NOT_FOUND);

        let (status, body) = send(
            test_app(state),
            Request::post("/api/v1/downloads/download-1/acknowledge-failure")
                .body(Body::empty())
                .unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["state"]["acknowledged"], true);
    }

    fn does_not_fit_assessment() -> ModelAssessment {
        ModelAssessment::DoesNotFit {
            profile: ServingProfile {
                context_length: 8192,
            },
            assessment_id: magnitude_service_contracts::models::ModelAssessmentId(
                "assessment".to_owned(),
            ),
            memory: Vec::new(),
            limiting_resource: "gpu".to_owned(),
            deficit_bytes: 1,
            projector: magnitude_service_contracts::models::ProjectorVerdict::None,
            drafter: DrafterVerdict::None,
        }
    }

    fn unsupported_assessment() -> ModelAssessment {
        ModelAssessment::Unsupported {
            profile: Some(ServingProfile {
                context_length: 8192,
            }),
            reasons: Vec::new(),
            detail: "unknown GGUF tensor type 31".to_owned(),
        }
    }

    fn assessed_entry(
        subject: ModelAssessmentSubject,
        assessment: ModelAssessment,
    ) -> magnitude_service_contracts::models::ModelAssessmentEntry {
        magnitude_service_contracts::models::ModelAssessmentEntry {
            subject,
            state: ModelAssessmentEntryState::Assessed {
                capabilities: ModelCapabilities {
                    vision: false,
                    tools: true,
                    structured_output: false,
                    reasoning: magnitude_service_contracts::models::ModelReasoningCapabilities {
                        supported: false,
                        efforts: Vec::new(),
                        default_effort: None,
                    },
                },
                template_fingerprint: "template".to_owned(),
                fingerprint: None,
                profiles: vec![assessment],
            },
        }
    }

    fn assessments_snapshot(
        catalog: Vec<magnitude_service_contracts::models::ModelAssessmentEntry>,
        custom: Vec<magnitude_service_contracts::models::ModelAssessmentEntry>,
    ) -> ModelAssessmentsSnapshot {
        ModelAssessmentsSnapshot {
            revision: 1,
            environment_id: magnitude_service_contracts::models::AssessmentEnvironmentId(
                "environment".to_owned(),
            ),
            catalog: ModelAssessmentDomainSnapshot::Available {
                source_revision: 3,
                entries: catalog,
            },
            custom: ModelAssessmentDomainSnapshot::Available {
                source_revision: 4,
                entries: custom,
            },
        }
    }

    async fn standard_models_with_custom_assessment(assessment: ModelAssessment) -> Value {
        let custom = ModelAssessmentSubject::Custom {
            model_id: local_model_id(),
        };
        let assessments = Arc::new(StubModelAssessments(assessments_snapshot(
            Vec::new(),
            vec![assessed_entry(custom, assessment)],
        )));
        let state = local_models_state(StubLocalModels::new()).with_model_assessments(assessments);
        let (status, body) = send(
            test_app(state),
            Request::get("/v1/models").body(Body::empty()).unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        body
    }

    #[tokio::test]
    async fn standard_models_list_fitting_custom_models_with_their_source() {
        let body = standard_models_with_custom_assessment(fits_assessment()).await;
        assert_eq!(body["data"][0]["id"], local_model_id().to_string());
        assert_eq!(body["data"][0]["owned_by"], "ollama");
        assert_eq!(body["data"][0]["name"], "Fine Tune");
        assert_eq!(body["data"][0]["description"], "Found in Ollama");
        assert_eq!(body["data"][0]["context_length"], 8192);
    }

    #[tokio::test]
    async fn standard_models_omit_a_reasoning_model_without_a_default_effort() {
        let mut entry = assessed_entry(
            ModelAssessmentSubject::Custom {
                model_id: local_model_id(),
            },
            fits_assessment(),
        );
        let ModelAssessmentEntryState::Assessed { capabilities, .. } = &mut entry.state else {
            unreachable!("assessed entry");
        };
        capabilities.reasoning = magnitude_service_contracts::models::ModelReasoningCapabilities {
            supported: true,
            efforts: vec!["none".to_owned(), "high".to_owned()],
            default_effort: None,
        };
        let assessments = Arc::new(StubModelAssessments(assessments_snapshot(
            Vec::new(),
            vec![entry],
        )));
        let state = local_models_state(StubLocalModels::new()).with_model_assessments(assessments);
        let (status, body) = send(
            test_app(state),
            Request::get("/v1/models").body(Body::empty()).unwrap(),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body["data"], json!([]));
    }

    #[tokio::test]
    async fn standard_models_omit_custom_models_that_are_not_offered() {
        for assessment in [unsupported_assessment(), does_not_fit_assessment()] {
            let body = standard_models_with_custom_assessment(assessment).await;
            assert_eq!(body["data"], json!([]));
        }
    }

    #[test]
    fn only_fitting_models_are_offered() {
        let catalog_fits = ModelAssessmentSubject::Catalog {
            model_id: "qwen3.6-35b-a3b:gguf:q4".parse().unwrap(),
            selection: magnitude_service_contracts::models::CatalogModelSelection::Effective,
        };
        let catalog_too_large = ModelAssessmentSubject::Catalog {
            model_id: "qwen3.6-27b:gguf:q4".parse().unwrap(),
            selection: magnitude_service_contracts::models::CatalogModelSelection::Effective,
        };
        let custom_fits = ModelAssessmentSubject::Custom {
            model_id: "hf:owner/repo/fits.gguf".parse().unwrap(),
        };
        let custom_unsupported = ModelAssessmentSubject::Custom {
            model_id: "hf:owner/repo/unsupported.gguf".parse().unwrap(),
        };
        let snapshot = assessments_snapshot(
            vec![
                assessed_entry(catalog_fits.clone(), fits_assessment()),
                assessed_entry(catalog_too_large, does_not_fit_assessment()),
            ],
            vec![
                assessed_entry(custom_fits.clone(), fits_assessment()),
                assessed_entry(custom_unsupported, unsupported_assessment()),
            ],
        );
        assert_eq!(
            offered_capabilities(&snapshot)
                .into_keys()
                .collect::<Vec<_>>(),
            {
                let mut expected = vec![catalog_fits, custom_fits];
                expected.sort();
                expected
            }
        );
    }

    #[test]
    fn custom_model_sources_name_where_the_files_came_from() {
        let hf_id: ModelId = "hf:owner/repo/model.gguf".parse().unwrap();
        let repository =
            magnitude_service_contracts::models::HuggingFaceRepositoryId::new("owner/repo")
                .unwrap();
        let cases = [
            (
                custom_model(
                    hf_id.clone(),
                    CustomModelOrigin::Added {
                        repository,
                        commit: "c".repeat(40),
                    },
                ),
                ("huggingface", "Added from Hugging Face (owner/repo)"),
            ),
            (
                custom_model(
                    hf_id.clone(),
                    CustomModelOrigin::Found {
                        source: FoundSource::HuggingFaceCache,
                    },
                ),
                (
                    "huggingface-cache",
                    "Found in the Hugging Face cache (owner/repo)",
                ),
            ),
            (
                custom_model(
                    hf_id,
                    CustomModelOrigin::Found {
                        source: FoundSource::LmStudio,
                    },
                ),
                ("lm-studio", "Found in LM Studio (owner/repo)"),
            ),
            (
                custom_model(
                    local_model_id(),
                    CustomModelOrigin::Imported {
                        path: "/models/a.gguf".into(),
                    },
                ),
                ("local-file", "Added from /models/a.gguf"),
            ),
            (
                custom_model(
                    local_model_id(),
                    CustomModelOrigin::Found {
                        source: FoundSource::Folder {
                            path: "/models".into(),
                        },
                    },
                ),
                ("folder", "Found in /models"),
            ),
        ];
        for (model, (owned_by, description)) in cases {
            assert_eq!(
                custom_model_source(&model),
                (owned_by, description.to_owned())
            );
        }
    }

    #[test]
    fn resource_topics_name_local_models_and_downloads() {
        assert_eq!(
            InferenceResourceTopic::parse("local-models"),
            Some(InferenceResourceTopic::LocalModels)
        );
        assert_eq!(
            InferenceResourceTopic::parse("downloads"),
            Some(InferenceResourceTopic::Downloads)
        );
        assert_eq!(InferenceResourceTopic::parse("discovery"), None);
        assert_eq!(
            serde_json::to_value(InferenceResourceTopic::LocalModels).unwrap(),
            "local-models"
        );
    }

    #[test]
    fn exported_document_declares_every_custom_model_operation() {
        let value = serde_json::to_value(openapi().unwrap()).unwrap();
        for (path, method) in [
            ("/api/v1/local-models/configuration", "get"),
            ("/api/v1/local-models/configuration", "put"),
            ("/api/v1/local-models", "get"),
            ("/api/v1/local-models/refresh", "post"),
            ("/api/v1/sources/hugging-face/credential", "put"),
            ("/api/v1/candidates/inspect", "post"),
            ("/api/v1/candidates/drafters", "post"),
            ("/api/v1/candidates/bundle", "post"),
            ("/api/v1/model-assessments/candidates", "post"),
            ("/api/v1/downloads", "get"),
            ("/api/v1/downloads", "post"),
            ("/api/v1/downloads/{download_id}/cancel", "post"),
            (
                "/api/v1/downloads/{download_id}/acknowledge-failure",
                "post",
            ),
        ] {
            assert!(
                value["paths"][path][method].is_object(),
                "{method} {path} is not exported"
            );
        }
        assert!(value["paths"]["/api/v1/discovery/models"].is_null());
        let schemas = &value["components"]["schemas"];
        for schema in [
            "UnsupportedReason",
            "DrafterVerdict",
            "ProjectorVerdict",
            "ModelFingerprint",
            "CustomModel",
            "LocalModelsConfiguration",
            "HuggingFaceCredential",
            "HuggingFaceAccess",
            "CandidateInspection",
            "DrafterCandidate",
        ] {
            assert!(
                schemas[schema].is_object(),
                "schema {schema} is not exported"
            );
        }
    }
}
