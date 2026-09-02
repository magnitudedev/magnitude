use std::collections::{BTreeMap, BTreeSet};
use std::convert::Infallible;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::{Path, Query, Request, State};
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::{future::BoxFuture, stream::BoxStream};
use icn_contracts::bootstrap_protocol::{
    BackendEligibilityReport, CudaEligibility, IcnBinaryIdentity, IcnInstallationBackend,
    IcnInstallationDeclaration, IcnStartupBackend, IcnStartupProgressRecord,
    IcnStartupProgressRecordType, IcnStartupRecord, IcnStartupRecordType, MetalEligibility,
    VulkanEligibility,
};
use icn_contracts::inference as domain;
use icn_contracts::models::{
    CatalogInstallationAdmission, CatalogInstallationOperation, CatalogInstallationOperationId,
    CatalogInstallationRemoval, CatalogInstallations, CatalogInstallationsResponse, CatalogModel,
    CatalogModelState, CatalogModels, CatalogModelsResponse, DiscoveredModel, DiscoveredModelState,
    DiscoveredModels, DiscoveredModelsResponse, EffectiveModel, ModelAssessmentDomainSnapshot,
    ModelAssessmentEntryState, ModelAssessmentPoolState, ModelAssessmentSubject, ModelAssessments,
    ModelAssessmentsSnapshot, ModelCapabilities, ModelDownloads, ModelId, ModelInstance,
    ModelInstanceId, ModelInstancesInvalidation, ModelInstancesSnapshot, ModelLoadPlan,
    ParsedModelId,
};
use icn_contracts::{
    CacheType, CompletionBackend, ExecutionConfig, ExecutionConfigReport, FlashAttention,
    GenerationMetrics, GenerationSnapshot, GpuLayers, HardwareProvider, HardwareSnapshot,
    HuggingFaceModelCatalog, HuggingFaceModelSearchRequest, HuggingFaceModelSearchResults,
    HuggingFaceRepositoryRequest, HuggingFaceRepositorySnapshot, ImageInput, InferenceError,
    InventoryError, ModelModalities, ModelProperties, PreparedChatInfo, SplitMode,
    TemplateCapabilities,
};
use serde::{Deserialize, Serialize};
use utoipa::openapi::extensions::Extensions;
use utoipa::openapi::path::Operation;
use utoipa::openapi::schema::{AdditionalProperties, Schema};
use utoipa::openapi::{Components, OpenApi as OpenApiDocument, RefOr};
use utoipa::{OpenApi, PartialSchema, ToSchema};

const CONNECTOR_MAX_OUTPUT_TOKENS: u32 = 32_768;

mod media;
mod protocols;

use protocols::chat::{
    AllowedToolRequest, AllowedToolsChoiceRequest, AllowedToolsModeRequest, AllowedToolsRequest,
    AllowedToolsType, ApplyTemplateRequest, ApplyTemplateResponse, ChatCompletionChoice,
    ChatCompletionChunk, ChatCompletionMessage, ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionStreamEvent, ChatContentPartRequest, ChatContentRequest, ChatMessageRequest,
    ChatToolCallRequest, ChatToolRequest, ChunkChoice, ChunkDelta, ChunkFunctionDelta,
    ChunkToolCall, CompletionFunctionCall, CompletionToolCall, FunctionDefinitionRequest,
    FunctionNameRequest, FunctionToolChoiceRequest, FunctionType, GrammarTriggerResponse,
    ImageUrlRequest, JsonSchemaRequest, NamedFunctionCallRequest, ReasoningEffortRequest,
    ResponseFormatRequest, StopRequest, StreamOptions, Timings, ToolChoiceModeRequest,
    ToolChoiceRequest, Usage, apply_template_response, validate_apply_template_request,
};
pub use protocols::responses::ResponseCreateRequest;
use protocols::responses::ResponseStreamEvent;

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
    next_id: Arc<AtomicU64>,
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

pub struct ModelInstanceLease {
    backend: Arc<dyn CompletionBackend>,
    instance_id: ModelInstanceId,
    model_aliases: Arc<BTreeSet<String>>,
    release: Option<Arc<dyn Fn() + Send + Sync>>,
}

pub type ModelLoadingObserver = Arc<dyn Fn(f32) + Send + Sync>;

impl ModelInstanceLease {
    pub fn new(
        backend: Arc<dyn CompletionBackend>,
        instance_id: ModelInstanceId,
        model_aliases: Arc<BTreeSet<String>>,
        release: impl Fn() + Send + Sync + 'static,
    ) -> Self {
        Self {
            backend,
            instance_id,
            model_aliases,
            release: Some(Arc::new(release)),
        }
    }

    pub fn backend(&self) -> &Arc<dyn CompletionBackend> {
        &self.backend
    }

    pub fn instance_id(&self) -> &ModelInstanceId {
        &self.instance_id
    }

    pub fn model_id(&self) -> &str {
        self.backend.model_id()
    }

    #[must_use]
    pub fn with_model_alias(mut self, alias: String) -> Self {
        self.model_aliases = Arc::new(BTreeSet::from([alias]));
        self
    }

    fn accepts_model(&self, requested: &str) -> bool {
        requested == self.backend.model_id() || self.model_aliases.contains(requested)
    }
}

impl Drop for ModelInstanceLease {
    fn drop(&mut self) {
        if let Some(release) = self.release.take() {
            release();
        }
    }
}

impl AppState {
    pub fn new(backend: impl CompletionBackend) -> Self {
        Self::from_shared_backend(Arc::new(backend))
    }

    /// Construct API state from a backend shared with another server-owned service.
    pub fn from_shared_backend(backend: Arc<dyn CompletionBackend>) -> Self {
        let (resource_changes, _) = tokio::sync::broadcast::channel(64);
        Self {
            catalog_models: None,
            discovered_models: None,
            catalog_installations: None,
            model_assessments: None,
            model_downloads: None,
            hardware: None,
            hugging_face_catalog: None,
            model_controller: Some(Arc::new(StaticModelInstanceController::new(backend))),
            identity: ServerIdentity::default(),
            authorization: None,
            next_id: Arc::new(AtomicU64::new(1)),
            resource_revision: Arc::new(AtomicU64::new(0)),
            resource_changes,
        }
    }

    pub fn model_free() -> Self {
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
            next_id: Arc::new(AtomicU64::new(1)),
            resource_revision: Arc::new(AtomicU64::new(0)),
            resource_changes,
        }
    }

    /// Accept an additional OpenAI request model name for the loaded backend.
    ///
    /// The backend's stable model ID remains authoritative; aliases are routing names only.
    pub fn with_model_alias(self, alias: impl Into<String>) -> Self {
        if let Some(controller) = &self.model_controller {
            controller.add_alias(alias.into());
        }
        self
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

pub fn app(state: AppState) -> Router {
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
        .route("/api/v1/models/{model_id}/properties", post(props))
        .route("/api/v1/chat/templates/apply", post(apply_template))
        .route("/v1/models", get(standard_models))
        .route(
            "/v1/chat/completions",
            post(protocols::chat::chat_completions),
        )
        .route(
            "/v1/responses",
            get(protocols::responses::responses_websocket).post(protocols::responses::responses),
        )
        .route(
            "/anthropic/v1/messages",
            post(protocols::anthropic::anthropic_messages),
        )
        .route(
            "/anthropic/v1/messages/count_tokens",
            post(protocols::anthropic::anthropic_count_tokens),
        )
        .route("/openapi.json", get(serve_openapi))
        .with_state(state.clone());
    if let Some(capability) = state.authorization.clone() {
        protected = protected.route_layer(middleware::from_fn_with_state(capability, authorize));
    }
    Router::new()
        .route("/health", get(health))
        .merge(protected)
        .with_state(state)
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
    fn instances(&self) -> BoxFuture<'_, ModelInstancesSnapshot>;
    fn watch_instances(&self) -> BoxStream<'static, ModelInstancesInvalidation>;
    fn lease(
        &self,
        instance_id: ModelInstanceId,
    ) -> BoxFuture<'_, Result<ModelInstanceLease, InventoryError>>;
    fn acquire_for_inference(
        &self,
        model_id: String,
        progress: Option<ModelLoadingObserver>,
    ) -> BoxFuture<'_, Result<ModelInstanceLease, InventoryError>>;
    fn add_alias(&self, _alias: String) {}
}

struct StaticModelInstanceController {
    backend: Arc<dyn CompletionBackend>,
    aliases: Arc<RwLock<BTreeSet<String>>>,
}

impl StaticModelInstanceController {
    const INSTANCE_ID: &'static str = "static-instance";

    fn new(backend: Arc<dyn CompletionBackend>) -> Self {
        Self {
            backend,
            aliases: Arc::new(RwLock::new(BTreeSet::new())),
        }
    }
}

impl ModelInstanceController for StaticModelInstanceController {
    fn preview_load(
        &self,
        _model_id: String,
    ) -> BoxFuture<'_, Result<ModelLoadPlan, InventoryError>> {
        Box::pin(async {
            Err(InventoryError::Unsupported(
                "static test model cannot preview loading".to_owned(),
            ))
        })
    }

    fn ensure_resident(
        &self,
        _model_id: String,
    ) -> BoxFuture<'_, Result<ModelInstance, InventoryError>> {
        Box::pin(async {
            Err(InventoryError::Unsupported(
                "static backends do not expose managed model instances".to_owned(),
            ))
        })
    }

    fn stop_instance(
        &self,
        _instance_id: ModelInstanceId,
    ) -> BoxFuture<'_, Result<(), InventoryError>> {
        Box::pin(async {
            Err(InventoryError::Unsupported(
                "static test model cannot be stopped".to_owned(),
            ))
        })
    }

    fn instances(&self) -> BoxFuture<'_, ModelInstancesSnapshot> {
        Box::pin(async {
            ModelInstancesSnapshot {
                revision: 0,
                instances: Vec::new(),
            }
        })
    }

    fn watch_instances(&self) -> BoxStream<'static, ModelInstancesInvalidation> {
        Box::pin(futures_util::stream::empty())
    }

    fn lease(
        &self,
        instance_id: ModelInstanceId,
    ) -> BoxFuture<'_, Result<ModelInstanceLease, InventoryError>> {
        let backend = Arc::clone(&self.backend);
        let aliases = self
            .aliases
            .read()
            .map(|aliases| Arc::new(aliases.clone()))
            .unwrap_or_else(|_| Arc::new(BTreeSet::new()));
        Box::pin(async move {
            if instance_id.0 != Self::INSTANCE_ID {
                return Err(InventoryError::NotReady(
                    "the requested model instance is not ready".to_owned(),
                ));
            }
            Ok(ModelInstanceLease::new(
                backend,
                instance_id,
                aliases,
                || {},
            ))
        })
    }

    fn acquire_for_inference(
        &self,
        model_id: String,
        _progress: Option<ModelLoadingObserver>,
    ) -> BoxFuture<'_, Result<ModelInstanceLease, InventoryError>> {
        let instance_id = ModelInstanceId(Self::INSTANCE_ID.to_owned());
        let accepted = model_id == self.backend.model_id()
            || self
                .aliases
                .read()
                .is_ok_and(|aliases| aliases.contains(&model_id));
        Box::pin(async move {
            if !accepted {
                return Err(InventoryError::NotReady(
                    "requested model instance is not ready".to_owned(),
                ));
            }
            self.lease(instance_id)
                .await
                .map(|lease| lease.with_model_alias(model_id))
        })
    }

    fn add_alias(&self, alias: String) {
        if let Ok(mut aliases) = self.aliases.write() {
            aliases.insert(alias);
        }
    }
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
    pub reasoning: ReasoningProfileResponse,
    pub training_context_tokens: u32,
    pub sliding_window_tokens: i32,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ReasoningProfileResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(nullable = false)]
    pub default_reasoning_effort: Option<String>,
    pub reasoning_efforts: Vec<String>,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ExecutionConfigResponse {
    pub requested: ExecutionSettingsResponse,
    /// Concrete parameters passed to the native backend after planning and thread selection.
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

    fn with_param(mut self, param: &'static str) -> Self {
        self.body.error.param = Some(param.to_owned());
        self
    }

    fn from_inference(error: InferenceError) -> Self {
        match error {
            InferenceError::InvalidConfig(message) => Self::invalid(message),
            error @ InferenceError::ContextLengthExceeded { .. } => Self {
                status: StatusCode::BAD_REQUEST,
                body: ErrorResponse {
                    error: inference_error_body(&error),
                },
            },
            InferenceError::ModelInstanceStopped => Self {
                status: StatusCode::CONFLICT,
                body: ErrorResponse {
                    error: inference_error_body(&InferenceError::ModelInstanceStopped),
                },
            },
            error => Self::server(error.to_string()),
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

fn with_openai_request_id(mut response: Response, request_id: &str) -> Response {
    if let Ok(value) = request_id.parse() {
        response.headers_mut().insert("x-request-id", value);
    }
    response
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
    Ok(Json(controller.instances().await))
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
    let downloads = state
        .model_downloads
        .as_ref()
        .ok_or_else(|| ApiError::server("model downloads are not configured"))?;
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
    let download_events = futures_util::StreamExt::flat_map(downloads.watch(), |event| {
        futures_util::stream::once(async move {
            InferenceResourceInvalidation {
                topic: InferenceResourceTopic::CatalogInstallations,
                revision: event.revision,
            }
        })
    });
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
                futures_util::stream::select(instance_events, download_events),
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
            selection: icn_contracts::models::CatalogModelSelection::Effective,
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
    capabilities: icn_contracts::models::ModelCapabilities,
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

#[utoipa::path(post, path = "/api/v1/models/{model_id}/properties", operation_id = "getModelProperties", tag = "models",
    params(("model_id" = String, Path, description = "Canonical model ID")), responses(
    (status = 200, description = "Loaded model and active template properties", body = PropsResponse),
    (status = 400, description = "Invalid model identity or request", body = ErrorResponse),
    (status = 404, description = "Model is not installed", body = ErrorResponse),
    (status = 409, description = "Model cannot currently be admitted", body = ErrorResponse),
    (status = 422, description = "Installed model failed integrity validation", body = ErrorResponse),
    (status = 500, description = "Properties unavailable", body = ErrorResponse)
))]
#[tracing::instrument(
    name = "icn.model.properties",
    skip_all,
    fields(model.id = tracing::field::Empty),
    err(Debug)
)]
async fn props(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> Result<Json<PropsResponse>, ApiError> {
    let controller = state
        .model_controller
        .as_ref()
        .ok_or_else(|| ApiError::server("model control is not configured"))?;
    let lease = controller
        .acquire_for_inference(model_id, None)
        .await
        .map_err(ApiError::from_inventory)?;
    tracing::Span::current().record("model.id", lease.model_id());
    let properties = lease
        .backend()
        .properties()
        .map_err(ApiError::from_inference)?;
    Ok(Json(props_response(properties)))
}

#[utoipa::path(post, path = "/api/v1/chat/templates/apply", operation_id = "applyChatTemplate", tag = "chat",
    request_body = ApplyTemplateRequest,
    responses(
        (status = 200, description = "Prepared native chat prompt and constraints", body = ApplyTemplateResponse),
        (status = 400, description = "Invalid request", body = ErrorResponse),
        (status = 404, description = "Model is not installed", body = ErrorResponse),
        (status = 409, description = "Model cannot currently be admitted", body = ErrorResponse),
        (status = 422, description = "Installed model failed integrity validation", body = ErrorResponse),
        (status = 500, description = "Template preparation failed", body = ErrorResponse)
    )
)]
#[tracing::instrument(
    name = "icn.apply_template",
    skip_all,
    fields(model.id = tracing::field::Empty),
    err(Debug)
)]
async fn apply_template(
    State(state): State<AppState>,
    Json(request): Json<ApplyTemplateRequest>,
) -> Result<Json<ApplyTemplateResponse>, ApiError> {
    let model_id = request
        .model
        .as_ref()
        .filter(|model| !model.is_empty())
        .cloned()
        .ok_or_else(|| ApiError::invalid("model is required"))?;
    let controller = state
        .model_controller
        .as_ref()
        .ok_or_else(|| ApiError::server("model control is not configured"))?;
    let lease = controller
        .acquire_for_inference(model_id, None)
        .await
        .map_err(ApiError::from_inventory)?;
    tracing::Span::current().record("model.id", lease.model_id());
    validate_model_selection(request.model.as_deref(), &lease)?;
    let properties = lease
        .backend()
        .properties()
        .map_err(ApiError::from_inference)?;
    let request = validate_apply_template_request(request)?;
    let request = resolve_request(request, &properties.reasoning)?;
    let span = tracing::Span::current();
    let prepared = tokio::task::spawn_blocking(move || {
        span.in_scope(|| lease.backend().apply_template(request))
    })
    .await
    .map_err(|error| ApiError::server(format!("template task failed: {error}")))?
    .map_err(ApiError::from_inference)?;
    Ok(Json(apply_template_response(prepared)))
}

pub(crate) struct ResidentInvocation {
    lease: ModelInstanceLease,
    model: String,
    request: domain::ResolvedInferenceRequest,
}

async fn acquire_invocation(
    controller: &Arc<dyn ModelInstanceController>,
    invocation: domain::InferenceInvocation,
    progress: Option<ModelLoadingObserver>,
) -> Result<ResidentInvocation, ApiError> {
    let (model, request) = invocation.into_parts();
    let model = model.into_inner();
    let lease = controller
        .acquire_for_inference(model.clone(), progress)
        .await
        .map_err(ApiError::from_inventory)?;
    validate_model_selection(Some(&model), &lease)?;
    let properties = lease
        .backend()
        .properties()
        .map_err(ApiError::from_inference)?;
    let request = resolve_request(request, &properties.reasoning)?;
    Ok(ResidentInvocation {
        lease,
        model,
        request,
    })
}

pub(crate) struct InferenceAdmission {
    sender: Option<tokio::sync::oneshot::Sender<Result<u64, InferenceError>>>,
}

impl InferenceAdmission {
    pub(crate) fn detached() -> Self {
        Self { sender: None }
    }

    pub(crate) fn channel() -> (
        Self,
        tokio::sync::oneshot::Receiver<Result<u64, InferenceError>>,
    ) {
        let (sender, receiver) = tokio::sync::oneshot::channel();
        (
            Self {
                sender: Some(sender),
            },
            receiver,
        )
    }

    pub(crate) fn admitted(&mut self, prompt_tokens: u64) -> Result<(), InferenceError> {
        let Some(sender) = self.sender.take() else {
            return Ok(());
        };
        sender
            .send(Ok(prompt_tokens))
            .map_err(|_| InferenceError::Callback("inference admission waiter disconnected".into()))
    }

    pub(crate) fn finish<T>(&mut self, result: &Result<T, InferenceError>) {
        let Some(sender) = self.sender.take() else {
            return;
        };
        let error = match result {
            Err(error) => error.clone(),
            Ok(_) => InferenceError::Backend(
                "inference completed without reporting admission".to_owned(),
            ),
        };
        let _ = sender.send(Err(error));
    }
}

pub(crate) async fn await_inference_admission(
    receiver: tokio::sync::oneshot::Receiver<Result<u64, InferenceError>>,
) -> Result<u64, ApiError> {
    receiver
        .await
        .map_err(|_| ApiError::server("inference task stopped before admission"))?
        .map_err(ApiError::from_inference)
}

fn execute_with_journal(
    backend: &dyn CompletionBackend,
    request: domain::ResolvedInferenceRequest,
    mut admit: impl FnMut(u64) -> Result<(), InferenceError>,
    mut observe: impl FnMut(&domain::InferenceObservation) -> Result<(), InferenceError>,
) -> Result<domain::InferenceResult, InferenceError> {
    let mut journal = domain::OutputJournal::default();
    let completion = backend.complete(request, &mut admit, &mut |observation| {
        if let domain::InferenceObservationEvent::Output { event } = observation.event() {
            journal
                .push(event)
                .map_err(|error| InferenceError::Backend(error.to_string()))?;
        }
        observe(&observation)
    })?;
    let output = journal
        .finish()
        .map_err(|error| InferenceError::Backend(error.to_string()))?;
    Ok(completion.into_result(output))
}

fn inference_error_body(error: &InferenceError) -> ApiErrorBody {
    let (error_type, code, retryable) = match error {
        InferenceError::InvalidConfig(_) => ("invalid_request_error", "invalid_request", false),
        InferenceError::ContextLengthExceeded { .. } => {
            ("invalid_request_error", "context_length_exceeded", false)
        }
        InferenceError::Backend(_) => ("server_error", "backend_error", true),
        InferenceError::Cancelled => ("cancelled", "request_cancelled", true),
        InferenceError::ModelInstanceStopped => ("model_error", "model_instance_stopped", false),
        InferenceError::Overloaded => ("server_error", "overloaded", true),
        InferenceError::ExecutorStopped => ("server_error", "executor_stopped", true),
        InferenceError::Callback(_) => ("server_error", "stream_callback_error", true),
    };
    ApiErrorBody {
        message: error.to_string(),
        r#type: error_type,
        param: None,
        code: code.to_owned(),
        retryable,
    }
}

fn validate_model_selection(
    requested: Option<&str>,
    lease: &ModelInstanceLease,
) -> Result<(), ApiError> {
    match requested {
        Some("") => Err(ApiError::invalid("model must not be empty")),
        Some(requested) if !lease.accepts_model(requested) => Err(ApiError::invalid(format!(
            "model {requested} is not loaded by this inference node"
        ))),
        _ => Ok(()),
    }
}

fn props_response(properties: ModelProperties) -> PropsResponse {
    let reasoning = ReasoningProfileResponse {
        default_reasoning_effort: properties
            .reasoning
            .default_effort
            .as_ref()
            .map(|effort| effort.0.clone()),
        reasoning_efforts: properties
            .reasoning
            .mappings
            .iter()
            .map(|mapping| mapping.effort.0.clone())
            .collect(),
    };
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
        reasoning,
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

fn resolve_request(
    request: domain::InferenceRequest<domain::ReasoningIntent>,
    profile: &icn_contracts::ReasoningProfile,
) -> Result<domain::ResolvedInferenceRequest, ApiError> {
    icn_reasoning::resolve_inference_request(request, profile).map_err(|error| match error {
        icn_reasoning::ReasoningResolutionError::InvalidRequest(message) => {
            ApiError::invalid(message)
        }
        icn_reasoning::ReasoningResolutionError::InvalidProfile(message) => {
            ApiError::server(message)
        }
    })
}

fn non_empty_text(value: String, field: &'static str) -> Result<domain::NonEmptyText, ApiError> {
    domain::NonEmptyText::try_new(value, field).map_err(domain_error)
}

fn non_empty_vec<T>(
    values: Vec<T>,
    field: &'static str,
) -> Result<domain::NonEmptyVec<T>, ApiError> {
    domain::NonEmptyVec::try_new(values, field).map_err(domain_error)
}

fn domain_error(error: domain::InferenceRequestError) -> ApiError {
    ApiError::invalid(error.to_string())
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
        props,
        apply_template,
        protocols::chat::chat_completions,
        protocols::responses::responses,
        protocols::anthropic::anthropic_messages,
        protocols::anthropic::anthropic_count_tokens
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
        ResponseCreateRequest,
        protocols::responses::ResponseObject,
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
        BackendEligibilityReport,
        CudaEligibility,
        VulkanEligibility,
        MetalEligibility,
        IcnBinaryIdentity,
        IcnStartupRecord,
        IcnStartupRecordType,
        IcnStartupProgressRecord,
        IcnStartupProgressRecordType,
        IcnStartupBackend,
        IcnInstallationDeclaration,
        IcnInstallationBackend
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
    const TYPED_REQUEST_SCHEMAS: [&str; 24] = [
        "AllowedToolRequest",
        "AllowedToolsChoiceRequest",
        "AllowedToolsRequest",
        "ChatCompletionRequest",
        "ChatToolCallRequest",
        "ChatToolRequest",
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

fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0)
}

pub struct FakeBackend {
    model_id: String,
    response: String,
    context_tokens: u32,
}
impl FakeBackend {
    pub fn new(model_id: impl Into<String>, response: impl Into<String>) -> Self {
        Self {
            model_id: model_id.into(),
            response: response.into(),
            context_tokens: 4096,
        }
    }

    #[cfg(test)]
    fn with_context_tokens(mut self, context_tokens: u32) -> Self {
        self.context_tokens = context_tokens;
        self
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
            context_tokens: self.context_tokens,
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
            reasoning: icn_contracts::ReasoningProfile {
                default_effort: Some(icn_contracts::NormalizedReasoningEffort("high".into())),
                mappings: vec![
                    icn_contracts::ReasoningEffortMapping {
                        effort: icn_contracts::NormalizedReasoningEffort("none".into()),
                        controls: icn_contracts::NativeReasoningControls {
                            enable_thinking: Some(false),
                            template_args: BTreeMap::new(),
                        },
                        automatic_budget: icn_contracts::AutomaticReasoningBudget::Disabled,
                    },
                    icn_contracts::ReasoningEffortMapping {
                        effort: icn_contracts::NormalizedReasoningEffort("high".into()),
                        controls: icn_contracts::NativeReasoningControls {
                            enable_thinking: Some(true),
                            template_args: BTreeMap::new(),
                        },
                        automatic_budget: icn_contracts::AutomaticReasoningBudget::Disabled,
                    },
                ],
                template_fingerprint: "fake-fingerprint".into(),
            },
            modalities: ModelModalities::default(),
            speculative: icn_contracts::SpeculativeDecodingRuntimeProperties::Disabled {
                reason: "fake_backend".into(),
            },
            execution: ExecutionConfigReport {
                requested: ExecutionConfig::default(),
                resolved: ExecutionConfig::default(),
            },
            template_fingerprint: "fake-fingerprint".into(),
        })
    }

    fn count_tokens(
        &self,
        request: domain::ResolvedInferenceRequest,
    ) -> Result<u64, InferenceError> {
        Ok(request.context().entries().len() as u64)
    }

    fn apply_template(
        &self,
        request: domain::ResolvedInferenceRequest,
    ) -> Result<PreparedChatInfo, InferenceError> {
        Ok(PreparedChatInfo {
            prompt: request
                .context()
                .system()
                .map(|system| system.as_str().to_owned())
                .into_iter()
                .chain(request.context().entries().iter().filter_map(|entry| {
                    match entry {
                        domain::ContextEntry::User { entry } => Some(
                            entry
                                .content()
                                .iter()
                                .filter_map(|part| match part {
                                    domain::UserContent::Text { text } => Some(text.as_str()),
                                    domain::UserContent::Image { .. } => None,
                                })
                                .collect::<String>(),
                        ),
                        domain::ContextEntry::Assistant { entry } => {
                            entry.text().map(|text| text.as_str().to_owned())
                        }
                    }
                }))
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
        request: domain::ResolvedInferenceRequest,
        on_admitted: &mut dyn FnMut(u64) -> Result<(), InferenceError>,
        on_event: &mut dyn FnMut(domain::InferenceObservation) -> Result<(), InferenceError>,
    ) -> Result<domain::InferenceCompletion, InferenceError> {
        let prompt_tokens = request.context().entries().len();
        let logical_prompt_tokens =
            u64::try_from(prompt_tokens).expect("fake prompt token count fits u64");
        icn_contracts::validate_inference_capacity(
            logical_prompt_tokens,
            u64::from(self.context_tokens),
        )?;
        on_admitted(logical_prompt_tokens)?;
        on_event(domain::InferenceObservation::new(
            domain::InferenceObservationEvent::Output {
                event: domain::InferenceOutputEvent::Started,
            },
            None,
        ))?;
        for (index, token) in self.response.split_inclusive(' ').enumerate() {
            on_event(domain::InferenceObservation::new(
                domain::InferenceObservationEvent::Output {
                    event: domain::InferenceOutputEvent::TextDelta {
                        text: domain::NonEmptyText::try_new(token, "fake response delta")
                            .expect("split_inclusive never yields empty tokens"),
                    },
                },
                Some(GenerationSnapshot {
                    cached_prompt_tokens: 0,
                    prompt_tokens,
                    generated_tokens: index + 1,
                    metrics: GenerationMetrics::default(),
                }),
            ))?;
        }
        Ok(domain::InferenceCompletion::new(
            domain::TokenUsage::new(
                prompt_tokens as u64,
                0,
                self.response.split_whitespace().count() as u64,
                0,
            ),
            domain::Termination::Natural,
            GenerationMetrics::default(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::chat::{
        adapt_request, finalize_request, timing_values, validate_request,
    };
    use axum::body::Body;
    use axum::http::Request;
    use http_body_util::BodyExt;
    use icn_contracts::InferenceProgress;
    use icn_contracts::models::{
        ModelAssessmentsInvalidation, ModelInstanceAllocation, ModelInstanceLifecycle,
    };
    use serde_json::Value as JsonValue;
    use serde_json::{Value, json};
    use std::sync::atomic::AtomicBool;
    use tower::ServiceExt;

    #[test]
    fn openai_model_discovery_projects_harness_metadata_into_data() {
        let model = open_ai_model(
            "local/model".to_owned(),
            "magnitude",
            "Local Model".to_owned(),
            "Local fixture.".to_owned(),
            65_536,
            icn_contracts::models::ModelCapabilities {
                vision: true,
                tools: true,
                structured_output: true,
                reasoning: icn_contracts::models::ModelReasoningCapabilities {
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
        let state = AppState::model_free();
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
            let response = app(AppState::model_free())
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
                    models: vec![icn_contracts::HuggingFaceModelSearchResult {
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
                    gguf_files: vec![icn_contracts::HuggingFaceRepositoryFile {
                        path: "model.gguf".into(),
                        size_bytes: 123,
                        content: icn_contracts::ContentIdentity::Sha256 {
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
                            "native_index": 0,
                            "backend": "cpu",
                            "physical_id": null,
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
        let response =
            app(AppState::new(FakeBackend::new("test-model", ""))
                .with_hardware(Arc::new(StubHardware)))
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
        let state = AppState::new(FakeBackend::new("test-model", ""))
            .with_hugging_face_catalog(Arc::new(StubHuggingFaceCatalog));
        let search = app(state.clone())
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

        let resolve = app(state)
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

    fn validate_test_request(
        request: ChatCompletionRequest,
    ) -> Result<(domain::ResolvedInferenceRequest, bool), ApiError> {
        let profile = FakeBackend::new("test-model", "")
            .properties()
            .expect("fake properties")
            .reasoning;
        let (request, include_usage) = validate_request(request).and_then(finalize_request)?;
        Ok((resolve_request(request, &profile)?, include_usage))
    }

    fn stream_json(body: &str) -> Vec<Value> {
        body.lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter(|data| *data != "[DONE]")
            .map(|data| serde_json::from_str(data).expect("SSE data must be JSON"))
            .collect()
    }

    #[test]
    fn equivalent_wire_dialects_construct_equal_canonical_requests() {
        let chat: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "test-model",
            "messages": [
                { "role": "system", "content": "be concise" },
                { "role": "user", "content": "hello" }
            ],
            "max_tokens": 16,
            "temperature": 1.0,
            "top_p": 1.0,
            "seed": 0
        }))
        .unwrap();
        let responses: protocols::responses::ResponseCreateRequest =
            serde_json::from_value(json!({
                "model": "test-model",
                "instructions": "be concise",
                "input": "hello",
                "max_output_tokens": 16,
                "temperature": 1.0,
                "top_p": 1.0
            }))
            .unwrap();
        let anthropic: protocols::anthropic::MessagesRequest = serde_json::from_value(json!({
            "model": "test-model",
            "system": "be concise",
            "messages": [{ "role": "user", "content": "hello" }],
            "max_tokens": 16,
            "temperature": 1.0,
            "top_p": 1.0
        }))
        .unwrap();

        let (_, chat) = adapt_request(chat).unwrap().invocation.into_parts();
        let (_, responses) = protocols::responses::adapt(responses)
            .unwrap()
            .invocation
            .into_parts();
        let (_, anthropic) = protocols::anthropic::adapt(anthropic)
            .unwrap()
            .invocation
            .into_parts();
        assert_eq!(chat, responses);
        assert_eq!(responses, anthropic);

        let profile = FakeBackend::new("test-model", "")
            .properties()
            .unwrap()
            .reasoning;
        let chat = resolve_request(chat, &profile).unwrap();
        let responses = resolve_request(responses, &profile).unwrap();
        let anthropic = resolve_request(anthropic, &profile).unwrap();
        assert_eq!(chat, responses);
        assert_eq!(responses, anthropic);
    }

    #[test]
    fn openai_adapters_do_not_invent_output_token_limits() {
        let chat = request_from_json(minimal_request());
        let (_, chat) = adapt_request(chat).unwrap().invocation.into_parts();
        assert_eq!(chat.generation().max_output_tokens(), None);

        let responses: protocols::responses::ResponseCreateRequest =
            serde_json::from_value(json!({
                "model": "test-model",
                "input": "hello"
            }))
            .unwrap();
        let (_, responses) = protocols::responses::adapt(responses)
            .unwrap()
            .invocation
            .into_parts();
        assert_eq!(responses.generation().max_output_tokens(), None);
    }

    #[test]
    fn compatibility_requests_ignore_unknown_object_fields_recursively() {
        let chat = serde_json::from_value::<ChatCompletionRequest>(json!({
            "model": "test-model",
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,AA==",
                        "future_image_option": true
                    },
                    "future_content_option": true
                }],
                "future_message_option": true
            }],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"type": "object"},
                    "future_function_option": true
                },
                "future_tool_option": true
            }],
            "tool_choice": {
                "type": "function",
                "function": {"name": "lookup", "future_choice_name_option": true},
                "future_choice_option": true
            },
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {"type": "object"},
                    "future_schema_option": true
                },
                "future_format_option": true
            },
            "stream_options": {"include_usage": true, "future_stream_option": true},
            "preserve_thinking": true
        }));
        assert!(chat.is_ok());

        let responses =
            serde_json::from_value::<protocols::responses::ResponseCreateRequest>(json!({
                "model": "test-model",
                "input": [{
                    "type": "message",
                    "role": "user",
                    "content": [{
                        "type": "input_text",
                        "text": "hello",
                        "future_content_option": true
                    }],
                    "future_message_option": true
                }],
                "tools": [{
                    "type": "function",
                    "name": "lookup",
                    "parameters": {"type": "object"},
                    "future_tool_option": true
                }],
                "tool_choice": {
                    "type": "function",
                    "name": "lookup",
                    "future_choice_option": true
                },
                "reasoning": {"effort": "high", "future_reasoning_option": true},
                "text": {
                    "format": {"type": "text", "future_format_option": true},
                    "future_text_option": true
                },
                "future_request_option": true
            }));
        assert!(responses.is_ok());

        let anthropic = serde_json::from_value::<protocols::anthropic::MessagesRequest>(json!({
            "model": "test-model",
            "system": [{
                "type": "text",
                "text": "system",
                "future_system_option": true
            }],
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "AA==",
                        "future_source_option": true
                    },
                    "future_content_option": true
                }],
                "future_message_option": true
            }],
            "max_tokens": 16,
            "tools": [{
                "name": "lookup",
                "input_schema": {"type": "object"},
                "future_tool_option": true
            }],
            "tool_choice": {"type": "auto", "future_choice_option": true},
            "thinking": {
                "type": "enabled",
                "budget_tokens": 8,
                "future_thinking_option": true
            },
            "future_request_option": true
        }));
        assert!(anthropic.is_ok());
    }

    #[test]
    fn compatibility_requests_keep_known_shapes_strict() {
        assert!(
            serde_json::from_value::<ChatCompletionRequest>(json!({
                "model": "test-model",
                "messages": [{"role": "future_role", "content": "hello"}]
            }))
            .is_err()
        );
        assert!(
            serde_json::from_value::<protocols::responses::ResponseContentPart>(json!({
                "type": "future_content",
                "text": "hello"
            }))
            .is_err()
        );
        assert!(
            serde_json::from_value::<protocols::anthropic::ContentBlock>(json!({
                "type": "future_content",
                "text": "hello"
            }))
            .is_err()
        );
        assert!(
            serde_json::from_value::<ChatCompletionRequest>(json!({
                "model": "test-model",
                "messages": [{"role": "user", "content": "hello"}],
                "stream_options": {"include_usage": "yes"}
            }))
            .is_err()
        );
    }

    #[test]
    fn chat_accepts_disabled_store_but_rejects_persistence() {
        let disabled = request_from_json(json!({
            "model": "test-model",
            "messages": [{ "role": "user", "content": "hello" }],
            "store": false
        }));
        assert!(adapt_request(disabled).is_ok());

        let enabled = request_from_json(json!({
            "model": "test-model",
            "messages": [{ "role": "user", "content": "hello" }],
            "store": true
        }));
        let Err(error) = adapt_request(enabled) else {
            panic!("local persistence must be rejected");
        };
        assert_eq!(
            error.body.error.message,
            "store is not supported by this local runtime"
        );
    }

    #[test]
    fn chat_accepts_empty_assistant_content_when_tool_calls_are_present() {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "test-model",
            "messages": [
                { "role": "user", "content": "look this up" },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": { "name": "lookup", "arguments": "{\"q\":\"x\"}" }
                    }]
                },
                { "role": "tool", "tool_call_id": "call_1", "name": "lookup", "content": "result" },
                { "role": "user", "content": "continue" }
            ]
        }))
        .expect("request must decode");

        let (_, canonical) = adapt_request(request)
            .expect("empty content is the wire representation of absent assistant text")
            .invocation
            .into_parts();
        let domain::ContextEntry::Assistant { entry } = &canonical.context().entries()[1] else {
            panic!("second entry must be the assistant tool-call turn");
        };
        assert!(entry.text().is_none());
        assert_eq!(entry.tool_calls().len(), 1);
    }

    #[test]
    fn chat_normalizes_permitted_empty_content_without_placeholder_text() {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "test-model",
            "messages": [
                { "role": "system", "content": "" },
                { "role": "user", "content": "" },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": { "name": "lookup", "arguments": "{}" }
                    }]
                },
                { "role": "tool", "tool_call_id": "call_1", "content": "" }
            ]
        }))
        .expect("request must decode");

        let (_, canonical) = adapt_request(request).unwrap().invocation.into_parts();
        assert!(canonical.context().system().is_none());
        let domain::ContextEntry::User { entry } = &canonical.context().entries()[0] else {
            panic!("first entry must be user");
        };
        assert!(entry.content().is_empty());
        let domain::ContextEntry::Assistant { entry } = &canonical.context().entries()[1] else {
            panic!("second entry must be assistant");
        };
        assert!(entry.text().is_none());
        assert!(entry.reasoning().is_none());
        assert!(entry.tool_calls()[0].result().content().is_empty());
    }

    #[test]
    fn chat_combines_leading_system_and_developer_instructions() {
        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "test-model",
            "messages": [
                { "role": "system", "content": "system" },
                { "role": "developer", "content": "developer" },
                { "role": "user", "content": "hello" }
            ]
        }))
        .unwrap();
        let (_, canonical) = adapt_request(request).unwrap().invocation.into_parts();
        assert_eq!(
            canonical
                .context()
                .system()
                .map(domain::NonEmptyText::as_str),
            Some("system\ndeveloper"),
        );
    }

    #[test]
    fn responses_normalizes_permitted_empty_content_without_placeholder_text() {
        let request: protocols::responses::ResponseCreateRequest = serde_json::from_value(json!({
            "model": "test-model",
            "instructions": "",
            "input": [
                { "type": "message", "role": "user", "content": "" },
                { "type": "message", "role": "assistant", "content": "" },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "lookup",
                    "arguments": "{}"
                },
                {
                    "type": "function_call_output",
                    "id": "out_1",
                    "call_id": "call_1",
                    "output": ""
                }
            ]
        }))
        .expect("request must decode");

        let (_, canonical) = protocols::responses::adapt(request)
            .unwrap()
            .invocation
            .into_parts();
        assert!(canonical.context().system().is_none());
        let domain::ContextEntry::User { entry } = &canonical.context().entries()[0] else {
            panic!("first entry must be user");
        };
        assert!(entry.content().is_empty());
        let domain::ContextEntry::Assistant { entry } = &canonical.context().entries()[1] else {
            panic!("second entry must be assistant");
        };
        assert!(entry.text().is_none());
        assert!(entry.tool_calls().is_empty());
        let domain::ContextEntry::Assistant { entry } = &canonical.context().entries()[2] else {
            panic!("third entry must be assistant tool history");
        };
        assert!(entry.tool_calls()[0].result().content().is_empty());
    }

    #[test]
    fn responses_accepts_easy_messages_and_replayed_output_messages() {
        let request: protocols::responses::ResponseCreateRequest = serde_json::from_value(json!({
            "model": "test-model",
            "prompt_cache_key": "session-1",
            "input": [
                { "role": "developer", "content": "be concise" },
                { "type": "message", "role": "user", "content": "hello" },
                {
                    "id": "rs_1",
                    "type": "reasoning",
                    "status": "completed",
                    "summary": [],
                    "content": [{ "type": "reasoning_text", "text": "brief thought" }]
                },
                {
                    "id": "msg_1",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "phase": "final_answer",
                    "content": [{
                        "type": "output_text",
                        "text": "hi",
                        "annotations": []
                    }]
                },
                { "role": "user", "content": "again" }
            ]
        }))
        .expect("standard Responses message forms must decode");

        let (_, canonical) = protocols::responses::adapt(request)
            .expect("easy and replayed messages must adapt")
            .invocation
            .into_parts();
        assert_eq!(
            canonical
                .context()
                .system()
                .map(domain::NonEmptyText::as_str),
            Some("be concise"),
        );
        assert_eq!(canonical.context().entries().len(), 3);
        let domain::ContextEntry::Assistant { entry } = &canonical.context().entries()[1] else {
            panic!("second entry must be assistant output");
        };
        assert_eq!(
            entry.reasoning().map(domain::NonEmptyText::as_str),
            Some("brief thought")
        );
    }

    #[test]
    fn responses_retains_non_function_tools_without_executing_them() {
        let request: protocols::responses::ResponseCreateRequest = serde_json::from_value(json!({
            "model": "test-model",
            "input": "hello",
            "reasoning": { "effort": "medium", "summary": "auto" },
            "include": ["reasoning.encrypted_content"],
            "client_metadata": { "turn_id": "turn-1" },
            "tools": [
                {
                    "type": "function",
                    "name": "exec_command",
                    "description": "Run a command",
                    "parameters": { "type": "object" },
                    "strict": false
                },
                {
                    "type": "namespace",
                    "name": "mcp__example",
                    "description": "Example remote tools",
                    "tools": [{
                        "type": "function",
                        "name": "remote_action",
                        "parameters": { "type": "object" }
                    }]
                },
                { "type": "web_search", "external_web_access": true },
                { "type": "file_search", "vector_store_ids": ["vs_1"] }
            ]
        }))
        .expect("hosted tool declarations of any type must decode");

        let (_, canonical) = protocols::responses::adapt(request)
            .expect("hosted tools must not block local function tools")
            .invocation
            .into_parts();
        assert_eq!(canonical.tools().definitions().len(), 1);
        assert_eq!(
            canonical.tools().definitions()[0].name().as_str(),
            "exec_command"
        );
    }

    #[test]
    fn responses_rejects_malformed_function_tools_instead_of_demoting_them() {
        let request: protocols::responses::ResponseCreateRequest = serde_json::from_value(json!({
            "model": "test-model",
            "input": "hello",
            "tools": [{ "type": "function", "name": "broken" }]
        }))
        .expect("the declaration decodes as an opaque tool");

        let Err(error) = protocols::responses::adapt(request) else {
            panic!("a function declaration missing parameters must not silently become opaque");
        };
        assert_eq!(
            error.body.error.message,
            "malformed function tool declaration"
        );
    }

    fn full_output_result() -> domain::InferenceResult {
        domain::InferenceResult::new(
            domain::InferenceOutput::new(
                Some(domain::NonEmptyText::try_new("brief thought", "reasoning").unwrap()),
                Some(domain::NonEmptyText::try_new("calling a tool", "text").unwrap()),
                vec![domain::ToolCall::new(
                    domain::ToolCallId::try_new("call_1").unwrap(),
                    domain::ToolName::try_new("lookup").unwrap(),
                    domain::JsonObject::new(serde_json::Map::new()),
                )],
            ),
            domain::TokenUsage::default(),
            domain::Termination::ToolCalls,
            GenerationMetrics::default(),
        )
    }

    // Replay closure: everything each protocol's projection emits must parse
    // back through that protocol's input path, because clients resend emitted
    // output verbatim as later history.
    #[test]
    fn responses_output_items_replay_as_input() {
        let result = full_output_result();
        let projection = protocols::responses::adapt(
            serde_json::from_value(json!({ "model": "test-model", "input": "hi" })).unwrap(),
        )
        .unwrap()
        .projection;
        let emitted = serde_json::to_value(protocols::responses::from_result(
            "resp_test",
            1,
            "test-model",
            &projection,
            &result,
        ))
        .unwrap();
        let mut input = emitted["output"]
            .as_array()
            .expect("output is an array")
            .clone();
        input.push(
            json!({ "type": "function_call_output", "call_id": "call_1", "output": "result" }),
        );
        input.push(json!({ "role": "user", "content": "continue" }));

        let request: protocols::responses::ResponseCreateRequest = serde_json::from_value(json!({
            "model": "test-model",
            "input": input,
        }))
        .expect("every emitted output item must decode as replay input");
        let (_, canonical) = protocols::responses::adapt(request)
            .expect("replayed output must adapt")
            .invocation
            .into_parts();
        assert_eq!(canonical.context().entries().len(), 3);
        let domain::ContextEntry::Assistant { entry } = &canonical.context().entries()[0] else {
            panic!("replayed reasoning and message must form an assistant entry");
        };
        assert_eq!(
            entry.reasoning().map(domain::NonEmptyText::as_str),
            Some("brief thought")
        );
    }

    #[test]
    fn anthropic_output_blocks_replay_as_input() {
        let result = full_output_result();
        let response = protocols::anthropic::message("msg_test", "test-model", &result);
        let content = serde_json::to_value(&response.content).unwrap();

        let request: protocols::anthropic::MessagesRequest = serde_json::from_value(json!({
            "model": "test-model",
            "max_tokens": 16,
            "messages": [
                { "role": "user", "content": "hello" },
                { "role": "assistant", "content": content },
                { "role": "user", "content": [
                    { "type": "tool_result", "tool_use_id": "call_1", "content": "result" }
                ] }
            ]
        }))
        .expect("every emitted content block must decode as replay input");
        let (_, canonical) = protocols::anthropic::adapt(request)
            .expect("replayed output must adapt")
            .invocation
            .into_parts();
        assert_eq!(canonical.context().entries().len(), 2);
        let domain::ContextEntry::Assistant { entry } = &canonical.context().entries()[1] else {
            panic!("replayed blocks must form an assistant entry");
        };
        assert_eq!(
            entry.reasoning().map(domain::NonEmptyText::as_str),
            Some("brief thought")
        );
        assert_eq!(entry.tool_calls().len(), 1);
    }

    #[test]
    fn chat_output_message_replays_as_input() {
        let result = full_output_result();
        let response = serde_json::to_value(protocols::chat::chat_completion_response(
            "chatcmpl_test".into(),
            1,
            "test-model".into(),
            &result,
        ))
        .unwrap();
        let message = response["choices"][0]["message"].clone();

        let request: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "test-model",
            "messages": [
                { "role": "user", "content": "hello" },
                message,
                { "role": "tool", "tool_call_id": "call_1", "content": "result" },
                { "role": "user", "content": "continue" }
            ]
        }))
        .expect("the emitted assistant message must decode as replay input");
        let (_, canonical) = adapt_request(request)
            .expect("replayed output must adapt")
            .invocation
            .into_parts();
        let domain::ContextEntry::Assistant { entry } = &canonical.context().entries()[1] else {
            panic!("the replayed message must form an assistant entry");
        };
        assert_eq!(
            entry.reasoning().map(domain::NonEmptyText::as_str),
            Some("brief thought")
        );
        assert_eq!(entry.tool_calls().len(), 1);
    }

    #[test]
    fn anthropic_maps_omitted_tool_result_content_to_an_empty_result() {
        let request: protocols::anthropic::MessagesRequest = serde_json::from_value(json!({
            "model": "test-model",
            "max_tokens": 16,
            "output_config": { "effort": "high" },
            "system": [{
                "type": "text",
                "text": "be concise",
                "cache_control": { "type": "ephemeral" }
            }],
            "messages": [
                { "role": "user", "content": [{
                    "type": "text",
                    "text": "look this up",
                    "cache_control": { "type": "ephemeral" }
                }] },
                { "role": "assistant", "content": [{
                    "type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}
                }] },
                { "role": "user", "content": [{
                    "type": "tool_result", "tool_use_id": "toolu_1"
                }] }
            ]
        }))
        .expect("omitted tool result content is valid Anthropic input");

        let (_, canonical) = protocols::anthropic::adapt(request)
            .unwrap()
            .invocation
            .into_parts();
        let domain::ContextEntry::Assistant { entry } = &canonical.context().entries()[1] else {
            panic!("second entry must be assistant");
        };
        assert!(entry.tool_calls()[0].result().content().is_empty());
    }

    #[test]
    fn admission_rounds_up_or_clamps_effort_to_the_model_domain() {
        let base_profile = FakeBackend::new("test-model", "")
            .properties()
            .expect("fake properties")
            .reasoning;
        let enabled = base_profile
            .mapping(&icn_contracts::NormalizedReasoningEffort("high".into()))
            .expect("fake profile must have an enabled mapping")
            .clone();
        let profile = |efforts: &[&str], default: &str| icn_contracts::ReasoningProfile {
            default_effort: Some(icn_contracts::NormalizedReasoningEffort(default.into())),
            mappings: efforts
                .iter()
                .map(|effort| {
                    let mut mapping = enabled.clone();
                    mapping.effort = icn_contracts::NormalizedReasoningEffort((*effort).into());
                    mapping
                })
                .collect(),
            template_fingerprint: base_profile.template_fingerprint.clone(),
        };

        for (efforts, default, wire_effort, expected) in [
            (vec!["low", "xhigh"], "low", "low", "low"),
            (vec!["low", "xhigh"], "low", "medium", "xhigh"),
            (vec!["low", "medium", "xhigh"], "medium", "high", "xhigh"),
            (vec!["low", "high"], "high", "xhigh", "high"),
            (vec!["low", "xhigh"], "xhigh", "max", "xhigh"),
            (vec!["high"], "high", "medium", "high"),
            (vec!["adaptive"], "adaptive", "medium", "adaptive"),
        ] {
            let profile = profile(&efforts, default);
            let request: protocols::anthropic::MessagesRequest = serde_json::from_value(json!({
                "model": "test-model",
                "max_tokens": 16,
                "output_config": { "effort": wire_effort },
                "messages": [{ "role": "user", "content": "hello" }]
            }))
            .expect("Anthropic effort must decode");
            let adapted =
                protocols::anthropic::adapt(request).expect("Anthropic effort must adapt");
            let (_, request) = adapted.invocation.into_parts();
            let resolved = resolve_request(request, &profile).expect("effort must resolve");
            assert_eq!(resolved.reasoning().effort().as_str(), expected);
        }
    }

    #[test]
    fn final_reasoning_resolution_rejects_unsupported_effort_and_disable() {
        let mut profile = FakeBackend::new("test-model", "")
            .properties()
            .expect("fake properties")
            .reasoning;
        profile
            .mappings
            .retain(|mapping| mapping.effort.as_str() != "none");

        let explicit = domain::ReasoningIntent::Effort {
            effort: icn_contracts::NormalizedReasoningEffort("medium".into()),
            template_args: BTreeMap::new(),
            budget: None,
        };
        assert!(icn_reasoning::resolve_reasoning_intent(explicit, &profile).is_err());
        assert!(
            icn_reasoning::resolve_reasoning_intent(
                domain::ReasoningIntent::Disabled {
                    template_args: BTreeMap::new(),
                },
                &profile,
            )
            .is_err()
        );
    }

    #[test]
    fn anthropic_rejects_effort_when_thinking_is_disabled() {
        let request: protocols::anthropic::MessagesRequest = serde_json::from_value(json!({
            "model": "test-model",
            "max_tokens": 16,
            "thinking": { "type": "disabled" },
            "output_config": { "effort": "high" },
            "messages": [{ "role": "user", "content": "hello" }]
        }))
        .expect("request must decode");
        assert!(protocols::anthropic::adapt(request).is_err());
    }

    #[test]
    fn anthropic_output_effort_cannot_select_none() {
        let request: protocols::anthropic::MessagesRequest = serde_json::from_value(json!({
            "model": "test-model",
            "max_tokens": 16,
            "output_config": { "effort": "none" },
            "messages": [{ "role": "user", "content": "hello" }]
        }))
        .expect("request must decode");
        assert!(protocols::anthropic::adapt(request).is_err());
    }

    #[test]
    fn anthropic_maps_system_role_messages_to_positioned_user_entries() {
        let request: protocols::anthropic::MessagesRequest = serde_json::from_value(json!({
            "model": "test-model",
            "max_tokens": 16,
            "system": "base instructions",
            "messages": [
                { "role": "user", "content": "hello" },
                { "role": "system", "content": [{ "type": "text", "text": "runtime instructions" }] },
                { "role": "assistant", "content": "hi" }
            ]
        }))
        .expect("Claude Code system-role compatibility input must decode");

        let (_, canonical) = protocols::anthropic::adapt(request)
            .unwrap()
            .invocation
            .into_parts();
        assert_eq!(
            canonical
                .context()
                .system()
                .map(domain::NonEmptyText::as_str),
            Some("base instructions"),
        );
        assert_eq!(canonical.context().entries().len(), 3);
        let domain::ContextEntry::User { entry } = &canonical.context().entries()[1] else {
            panic!("system-role message must become a user entry at its position");
        };
        let [domain::UserContent::Text { text }] = entry.content() else {
            panic!("system-role message must carry its text content");
        };
        assert_eq!(text.as_str(), "runtime instructions");
    }

    fn anthropic_canonical(value: Value) -> domain::InferenceRequest<domain::ReasoningIntent> {
        let request: protocols::anthropic::MessagesRequest =
            serde_json::from_value(value).expect("request must decode");
        protocols::anthropic::adapt(request)
            .expect("request must adapt")
            .invocation
            .into_parts()
            .1
    }

    #[test]
    fn anthropic_strips_claude_code_attribution_before_model_context() {
        // Inline legacy shape: attribution and the real prompt share a block.
        let inline = anthropic_canonical(json!({
            "model": "test-model",
            "max_tokens": 16,
            "system": "x-anthropic-billing-header: cc_version=2.1.101.e51; cc_entrypoint=cli; cch=a5145;You are Claude Code.",
            "messages": [{ "role": "user", "content": "hello" }]
        }));
        assert_eq!(
            inline.context().system().map(domain::NonEmptyText::as_str),
            Some("You are Claude Code."),
        );

        // Standalone shape: attribution is its own leading block.
        let standalone = anthropic_canonical(json!({
            "model": "test-model",
            "max_tokens": 16,
            "system": [
                { "type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.181; cc_entrypoint=cli; cch=a5145;" },
                { "type": "text", "text": "You are Claude Code." }
            ],
            "messages": [{ "role": "user", "content": "hello" }]
        }));
        assert_eq!(
            standalone
                .context()
                .system()
                .map(domain::NonEmptyText::as_str),
            Some("You are Claude Code."),
        );

        // Attribution-only system leaves no system prompt at all.
        let only = anthropic_canonical(json!({
            "model": "test-model",
            "max_tokens": 16,
            "system": "x-anthropic-billing-header: cch=a5145;",
            "messages": [{ "role": "user", "content": "hello" }]
        }));
        assert_eq!(only.context().system(), None);
    }

    #[test]
    fn anthropic_attribution_stripping_is_nonce_invariant_and_idempotent() {
        let request = |cch: &str| {
            json!({
                "model": "test-model",
                "max_tokens": 16,
                "system": format!("x-anthropic-billing-header: cc_version=2.1.101.e51; cc_entrypoint=cli; cch={cch};You are Claude Code."),
                "messages": [{ "role": "user", "content": "hello" }]
            })
        };
        let first = anthropic_canonical(request("a5145"));
        let second = anthropic_canonical(request("0beef"));
        // Requests differing only in the per-request stamp must produce
        // identical canonical context, or prompt-prefix reuse is defeated.
        assert_eq!(first.context(), second.context());

        // Idempotent: projecting already-projected content changes nothing.
        let replayed = anthropic_canonical(json!({
            "model": "test-model",
            "max_tokens": 16,
            "system": first.context().system().unwrap().as_str(),
            "messages": [{ "role": "user", "content": "hello" }]
        }));
        assert_eq!(first.context().system(), replayed.context().system());
    }

    #[test]
    fn anthropic_attribution_recognition_is_strictly_positional() {
        // The marker in a later block is ordinary content.
        let later_block = anthropic_canonical(json!({
            "model": "test-model",
            "max_tokens": 16,
            "system": [
                { "type": "text", "text": "Real instructions." },
                { "type": "text", "text": "x-anthropic-billing-header: cch=a5145;" }
            ],
            "messages": [{ "role": "user", "content": "hello" }]
        }));
        assert_eq!(
            later_block
                .context()
                .system()
                .map(domain::NonEmptyText::as_str),
            Some("Real instructions.\nx-anthropic-billing-header: cch=a5145;"),
        );

        // A non-leading mention within the first block is ordinary content.
        let mention = "Mention of x-anthropic-billing-header: cch=a5145; in prose.";
        let non_leading = anthropic_canonical(json!({
            "model": "test-model",
            "max_tokens": 16,
            "system": mention,
            "messages": [{ "role": "user", "content": "hello" }]
        }));
        assert_eq!(
            non_leading
                .context()
                .system()
                .map(domain::NonEmptyText::as_str),
            Some(mention),
        );

        // A sentinel without the documented stamp is preserved, not rejected
        // and not guessed at.
        let malformed = "x-anthropic-billing-header: cc_version=2.1.101; no stamp";
        let preserved = anthropic_canonical(json!({
            "model": "test-model",
            "max_tokens": 16,
            "system": malformed,
            "messages": [{ "role": "user", "content": "hello" }]
        }));
        assert_eq!(
            preserved
                .context()
                .system()
                .map(domain::NonEmptyText::as_str),
            Some(malformed),
        );
    }

    #[test]
    fn protocol_forbidden_empty_assistant_forms_remain_invalid() {
        let chat: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "test-model",
            "messages": [
                { "role": "user", "content": "hi" },
                { "role": "assistant", "content": null }
            ]
        }))
        .unwrap();
        assert!(adapt_request(chat).is_err());

        let chat: ChatCompletionRequest = serde_json::from_value(json!({
            "model": "test-model",
            "messages": [
                { "role": "user", "content": "hi" },
                { "role": "assistant", "content": "" }
            ]
        }))
        .unwrap();
        assert!(adapt_request(chat).is_err());

        let anthropic: protocols::anthropic::MessagesRequest = serde_json::from_value(json!({
            "model": "test-model",
            "max_tokens": 16,
            "messages": [
                { "role": "user", "content": "hi" },
                { "role": "assistant", "content": [] }
            ]
        }))
        .unwrap();
        assert!(protocols::anthropic::adapt(anthropic).is_err());
    }

    #[test]
    fn empty_canonical_output_uses_each_protocols_native_empty_shape() {
        let result = domain::InferenceResult::new(
            domain::InferenceOutput::new(None, None, Vec::new()),
            domain::TokenUsage::default(),
            domain::Termination::Natural,
            GenerationMetrics::default(),
        );

        let chat = serde_json::to_value(protocols::chat::chat_completion_response(
            "chatcmpl_test".into(),
            1,
            "test-model".into(),
            &result,
        ))
        .unwrap();
        assert!(chat["choices"][0]["message"]["content"].is_null());
        assert!(chat["choices"][0]["message"].get("tool_calls").is_none());

        let projection = protocols::responses::adapt(
            serde_json::from_value(json!({ "model": "test-model", "input": "hi" })).unwrap(),
        )
        .unwrap()
        .projection;
        let responses = serde_json::to_value(protocols::responses::from_result(
            "resp_test",
            1,
            "test-model",
            &projection,
            &result,
        ))
        .unwrap();
        assert_eq!(responses["output"], json!([]));

        let anthropic = serde_json::to_value(protocols::anthropic::message(
            "msg_test",
            "test-model",
            &result,
        ))
        .unwrap();
        assert_eq!(anthropic["content"], json!([]));
    }

    #[test]
    fn stopped_model_instance_has_a_non_retryable_error_contract() {
        let error = inference_error_body(&InferenceError::ModelInstanceStopped);
        assert_eq!(error.r#type, "model_error");
        assert_eq!(error.code, "model_instance_stopped");
        assert!(!error.retryable);
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

    #[tokio::test]
    async fn streaming_context_overflow_uses_protocol_native_http_errors() {
        let (chat_status, chat_body) = post_chat(
            FakeBackend::new("test-model", "hello").with_context_tokens(1),
            json!({
                "model": "test-model",
                "stream": true,
                "messages": [{ "role": "user", "content": "hi" }]
            }),
        )
        .await;
        assert_eq!(chat_status, StatusCode::BAD_REQUEST);
        assert_eq!(
            serde_json::from_str::<Value>(&chat_body).unwrap(),
            json!({
                "error": {
                    "message": "prompt is too long: 1 tokens leave no generation capacity in a 1-token context",
                    "type": "invalid_request_error",
                    "param": "messages",
                    "code": "context_length_exceeded"
                }
            })
        );

        let responses = app(AppState::new(
            FakeBackend::new("test-model", "hello").with_context_tokens(1),
        ))
        .oneshot(
            Request::post("/v1/responses")
                .header("content-type", "application/json")
                .body(Body::from(
                    json!({
                        "model": "test-model",
                        "input": "hi",
                        "stream": true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(responses.status(), StatusCode::BAD_REQUEST);
        let responses_body: Value =
            serde_json::from_slice(&responses.into_body().collect().await.unwrap().to_bytes())
                .unwrap();
        assert_eq!(responses_body["error"]["code"], "context_length_exceeded");
        assert_eq!(responses_body["error"]["param"], "input");

        let anthropic = app(AppState::new(
            FakeBackend::new("test-model", "hello").with_context_tokens(1),
        ))
        .oneshot(
            Request::post("/anthropic/v1/messages")
                .header("content-type", "application/json")
                .header("anthropic-version", "2023-06-01")
                .body(Body::from(
                    json!({
                        "model": "test-model",
                        "max_tokens": 32,
                        "stream": true,
                        "messages": [{ "role": "user", "content": "hi" }]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(anthropic.status(), StatusCode::BAD_REQUEST);
        let anthropic_body: Value =
            serde_json::from_slice(&anthropic.into_body().collect().await.unwrap().to_bytes())
                .unwrap();
        assert_eq!(anthropic_body["type"], "error");
        assert_eq!(anthropic_body["error"]["type"], "invalid_request_error");
        assert_eq!(
            anthropic_body["error"]["message"],
            "prompt is too long: 1 tokens leave no generation capacity in a 1-token context"
        );
        assert!(anthropic_body["request_id"].as_str().is_some());
    }

    #[tokio::test]
    async fn explicit_progress_streams_report_context_overflow_in_stream() {
        let chat = app(AppState::new(
            FakeBackend::new("test-model", "hello").with_context_tokens(1),
        ))
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .header("Magnitude-Include-Progress", "true")
                .body(Body::from(
                    json!({
                        "model": "test-model",
                        "stream": true,
                        "messages": [{ "role": "user", "content": "hi" }]
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(chat.status(), StatusCode::OK);
        let chat_body = String::from_utf8(
            chat.into_body()
                .collect()
                .await
                .unwrap()
                .to_bytes()
                .to_vec(),
        )
        .unwrap();
        assert!(chat_body.contains("event: error"));
        assert!(chat_body.contains("context_length_exceeded"));
        assert!(!chat_body.contains("data: [DONE]"));

        let responses = app(AppState::new(
            FakeBackend::new("test-model", "hello").with_context_tokens(1),
        ))
        .oneshot(
            Request::post("/v1/responses")
                .header("content-type", "application/json")
                .header("Magnitude-Include-Progress", "true")
                .body(Body::from(
                    json!({
                        "model": "test-model",
                        "input": "hi",
                        "stream": true
                    })
                    .to_string(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(responses.status(), StatusCode::OK);
        let responses_body = String::from_utf8(
            responses
                .into_body()
                .collect()
                .await
                .unwrap()
                .to_bytes()
                .to_vec(),
        )
        .unwrap();
        assert!(responses_body.contains("response.failed"));
        assert!(responses_body.contains("context_length_exceeded"));
    }

    #[tokio::test]
    async fn requested_output_limit_does_not_change_prompt_admission() {
        let (status, body) = post_chat(
            FakeBackend::new("test-model", "hello").with_context_tokens(2),
            json!({
                "model": "test-model",
                "stream": true,
                "max_tokens": 32_768,
                "messages": [{ "role": "user", "content": "hi" }]
            }),
        )
        .await;

        assert_eq!(status, StatusCode::OK);
        assert!(body.contains("data: [DONE]"));
    }

    #[tokio::test]
    async fn every_local_protocol_reconciles_unsupported_reasoning_effort() {
        let (chat_status, _) = post_chat(
            FakeBackend::new("test-model", "hello"),
            json!({
                "model": "test-model",
                "reasoning_effort": "medium",
                "messages": [{ "role": "user", "content": "hi" }]
            }),
        )
        .await;
        assert_eq!(chat_status, StatusCode::OK);

        let responses = app(AppState::new(FakeBackend::new("test-model", "hello")))
            .oneshot(
                Request::post("/v1/responses")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "model": "test-model",
                            "reasoning": { "effort": "medium" },
                            "input": "hi"
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(responses.status(), StatusCode::OK);

        let anthropic = app(AppState::new(FakeBackend::new("test-model", "hello")))
            .oneshot(
                Request::post("/anthropic/v1/messages")
                    .header("content-type", "application/json")
                    .header("anthropic-version", "2023-06-01")
                    .body(Body::from(
                        json!({
                            "model": "test-model",
                            "max_tokens": 32,
                            "output_config": { "effort": "medium" },
                            "messages": [{ "role": "user", "content": "hi" }]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(anthropic.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn responses_stream_uses_the_same_model_pipeline() {
        let response = app(AppState::new(FakeBackend::new("test-model", "hello")))
            .oneshot(
                Request::post("/v1/responses")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "model": "test-model",
                            "input": "hi",
                            "stream": true
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = String::from_utf8(
            response
                .into_body()
                .collect()
                .await
                .unwrap()
                .to_bytes()
                .to_vec(),
        )
        .unwrap();
        assert!(body.contains("event: response.created"));
        assert!(body.contains("event: response.output_text.delta"));
        assert!(body.contains("event: response.completed"));
        assert!(body.contains("\"sequence_number\":0"));
    }

    #[tokio::test]
    async fn responses_supports_typed_non_streaming_requests() {
        let response = app(AppState::new(FakeBackend::new("test-model", "hello")))
            .oneshot(
                Request::post("/v1/responses")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "model": "test-model",
                            "instructions": "answer precisely",
                            "max_output_tokens": 17,
                            "temperature": 0.2,
                            "top_p": 0.8,
                            "parallel_tool_calls": false,
                            "metadata": { "trace": "test" },
                            "tools": [{
                                "type": "function",
                                "name": "lookup",
                                "description": "Look up a value",
                                "parameters": { "type": "object" },
                                "strict": true
                            }],
                            "tool_choice": "required",
                            "input": [{
                                "type": "message",
                                "role": "user",
                                "content": [{
                                    "type": "input_text",
                                    "text": "hi"
                                }]
                            }]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body: Value =
            serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes())
                .unwrap();
        assert_eq!(body["object"], "response");
        assert_eq!(body["status"], "completed");
        assert_eq!(body["output"][0]["content"][0]["text"], "hello");
        assert_eq!(body["usage"]["output_tokens"], 1);
        assert_eq!(body["instructions"], "answer precisely");
        assert_eq!(body["max_output_tokens"], 17);
        assert_eq!(body["temperature"], 0.2);
        assert_eq!(body["top_p"], 0.8);
        assert_eq!(body["parallel_tool_calls"], false);
        assert_eq!(body["tool_choice"], "required");
        assert_eq!(body["tools"][0]["name"], "lookup");
        assert_eq!(body["metadata"]["trace"], "test");
    }

    #[tokio::test]
    async fn anthropic_messages_echoes_the_gateway_alias_without_leaking_it_to_inference() {
        let response = app(AppState::new(FakeBackend::new("test-model", "hello")))
            .oneshot(
                Request::post("/anthropic/v1/messages")
                    .header("content-type", "application/json")
                    .header("anthropic-version", "2023-06-01")
                    .header("Magnitude-Gateway-Model", "anthropic-local/test-model")
                    .body(Body::from(
                        json!({
                            "model": "test-model",
                            "max_tokens": 32,
                            "messages": [{ "role": "user", "content": "hi" }]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body: Value =
            serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes())
                .unwrap();
        assert_eq!(body["type"], "message");
        assert_eq!(body["model"], "anthropic-local/test-model");
        assert_eq!(body["content"][0]["type"], "text");
        assert_eq!(body["content"][0]["text"], "hello");
        assert_eq!(body["stop_reason"], "end_turn");
    }

    #[tokio::test]
    async fn anthropic_count_tokens_uses_the_adapted_native_request() {
        let response = app(AppState::new(FakeBackend::new("test-model", "hello")))
            .oneshot(
                Request::post("/anthropic/v1/messages/count_tokens")
                    .header("content-type", "application/json")
                    .header("anthropic-version", "2023-06-01")
                    .body(Body::from(
                        json!({
                            "model": "test-model",
                            "max_tokens": 32,
                            "messages": [{ "role": "user", "content": "hi" }]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body: Value =
            serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes())
                .unwrap();
        assert_eq!(body, json!({ "input_tokens": 1 }));
    }

    #[tokio::test]
    async fn anthropic_stream_follows_message_and_content_block_lifecycle() {
        let response = app(AppState::new(FakeBackend::new("test-model", "hello")))
            .oneshot(
                Request::post("/anthropic/v1/messages")
                    .header("content-type", "application/json")
                    .header("anthropic-version", "2023-06-01")
                    .body(Body::from(
                        json!({
                            "model": "test-model",
                            "max_tokens": 32,
                            "stream": true,
                            "messages": [{ "role": "user", "content": "hi" }]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = String::from_utf8(
            response
                .into_body()
                .collect()
                .await
                .unwrap()
                .to_bytes()
                .to_vec(),
        )
        .unwrap();
        let start = body.find("event: message_start").unwrap();
        let block = body.find("event: content_block_start").unwrap();
        let delta = body.find("event: content_block_delta").unwrap();
        let stop = body.find("event: content_block_stop").unwrap();
        let message_stop = body.find("event: message_stop").unwrap();
        assert!(start < block && block < delta && delta < stop && stop < message_stop);
        assert!(body.contains("\"input_tokens\":1"));
    }

    #[tokio::test]
    async fn chat_progress_is_present_only_when_explicitly_requested() {
        let backend = || ScriptedBackend {
            events: vec![
                progress_event(InferenceProgress::Queued),
                output_event(domain::InferenceOutputEvent::TextDelta {
                    text: delta_text("hello"),
                }),
            ],
            fail: false,
        };
        let (status, body) = post_chat(backend(), minimal_request()).await;
        assert_eq!(status, StatusCode::OK);
        assert!(
            stream_json(&body)
                .iter()
                .all(|chunk| chunk.get("progress").is_none())
        );

        let response = app(AppState::new(backend()))
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .header("Magnitude-Include-Progress", "true")
                    .body(Body::from(minimal_request().to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let body = String::from_utf8(
            response
                .into_body()
                .collect()
                .await
                .unwrap()
                .to_bytes()
                .to_vec(),
        )
        .unwrap();
        assert!(stream_json(&body).iter().any(|chunk| {
            chunk["progress"]["phase"] == "queued" && chunk["choices"] == json!([])
        }));
    }

    struct StubModelInstanceController {
        backend: Arc<dyn CompletionBackend>,
        leases: Arc<AtomicU64>,
        pending_acquisition_dropped: Option<Arc<AtomicBool>>,
        acquisition_fails: bool,
    }

    struct DropFlag(Arc<AtomicBool>);

    impl Drop for DropFlag {
        fn drop(&mut self) {
            self.0.store(true, Ordering::Release);
        }
    }

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
            model_id: String,
        ) -> BoxFuture<'_, Result<ModelInstance, InventoryError>> {
            Box::pin(async move {
                let lease = self.acquire_for_inference(model_id, None).await?;
                self.instances()
                    .await
                    .instances
                    .into_iter()
                    .find(|instance| instance.id == lease.instance_id)
                    .ok_or_else(|| {
                        InventoryError::NotReady("model instance is not ready".to_owned())
                    })
            })
        }

        fn stop_instance(
            &self,
            _instance_id: ModelInstanceId,
        ) -> BoxFuture<'_, Result<(), InventoryError>> {
            Box::pin(async { Ok(()) })
        }

        fn instances(&self) -> BoxFuture<'_, ModelInstancesSnapshot> {
            Box::pin(async {
                ModelInstancesSnapshot {
                    revision: 0,
                    instances: vec![ModelInstance {
                        id: ModelInstanceId("test-instance".to_owned()),
                        model_id: "test-model:gguf:f16".parse().unwrap(),
                        lifecycle: ModelInstanceLifecycle::Ready {
                            allocation: ModelInstanceAllocation {
                                context_window_tokens: 1,
                                parallel_sequences: 1,
                                physical_context_tokens: 1,
                                memory_domains: Vec::new(),
                            },
                        },
                    }],
                }
            })
        }

        fn watch_instances(&self) -> BoxStream<'static, ModelInstancesInvalidation> {
            Box::pin(futures_util::stream::empty())
        }

        fn lease(
            &self,
            instance_id: ModelInstanceId,
        ) -> BoxFuture<'_, Result<ModelInstanceLease, InventoryError>> {
            let backend = Arc::clone(&self.backend);
            Box::pin(async move {
                if instance_id.0 != "test-instance" {
                    return Err(InventoryError::NotReady(
                        "test model instance unavailable".to_owned(),
                    ));
                }
                self.leases.fetch_add(1, Ordering::Relaxed);
                Ok(ModelInstanceLease::new(
                    backend,
                    instance_id,
                    Arc::new(BTreeSet::new()),
                    || {},
                ))
            })
        }

        fn acquire_for_inference(
            &self,
            model_id: String,
            _progress: Option<ModelLoadingObserver>,
        ) -> BoxFuture<'_, Result<ModelInstanceLease, InventoryError>> {
            if self.acquisition_fails {
                return Box::pin(async {
                    Err(InventoryError::ModelOperation {
                        code: "model_instance_stopped".to_owned(),
                        message: "model instance was stopped".to_owned(),
                        retryable: false,
                    })
                });
            }
            if let Some(dropped) = self.pending_acquisition_dropped.clone() {
                return Box::pin(async move {
                    let _drop = DropFlag(dropped);
                    std::future::pending().await
                });
            }
            Box::pin(async move {
                self.lease(ModelInstanceId("test-instance".to_owned()))
                    .await
                    .map(|lease| lease.with_model_alias(model_id))
            })
        }
    }

    #[tokio::test]
    async fn exposes_the_authoritative_model_instances_snapshot() {
        let controller = Arc::new(StubModelInstanceController {
            backend: Arc::new(FakeBackend::new("test-model", "ready")),
            leases: Arc::new(AtomicU64::new(0)),
            pending_acquisition_dropped: None,
            acquisition_fails: false,
        });
        let response = app(AppState::model_free().with_model_controller(controller))
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
            state: icn_contracts::models::ModelAssessmentPoolState::Preparing,
        }));
        let response = app(AppState::model_free().with_model_assessments(assessments))
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

    #[tokio::test]
    async fn ordinary_chat_leases_the_requested_resident_model_before_streaming() {
        let leases = Arc::new(AtomicU64::new(0));
        let controller = Arc::new(StubModelInstanceController {
            backend: Arc::new(FakeBackend::new("test-model", "ready")),
            leases: Arc::clone(&leases),
            pending_acquisition_dropped: None,
            acquisition_fails: false,
        });
        let response = app(AppState::model_free().with_model_controller(controller))
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "model": "test-model",
                            "messages": [{"role": "user", "content": "hi"}],
                            "stream": true
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = response.into_body().collect().await.unwrap().to_bytes();
        let body = String::from_utf8(body.to_vec()).unwrap();
        assert!(body.contains("ready"));
        assert_eq!(leases.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn progress_chat_reports_acquisition_failure_in_stream() {
        let controller = Arc::new(StubModelInstanceController {
            backend: Arc::new(FakeBackend::new("test-model", "unused")),
            leases: Arc::new(AtomicU64::new(0)),
            pending_acquisition_dropped: None,
            acquisition_fails: true,
        });
        let response = app(AppState::model_free().with_model_controller(controller))
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .header("Magnitude-Include-Progress", "true")
                    .body(Body::from(minimal_request().to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = String::from_utf8(
            response
                .into_body()
                .collect()
                .await
                .unwrap()
                .to_bytes()
                .to_vec(),
        )
        .unwrap();
        let error = stream_json(&body).pop().unwrap();
        assert_eq!(error["error"]["type"], "model_error");
        assert_eq!(error["error"]["code"], "model_instance_stopped");
    }

    #[tokio::test]
    async fn invalid_chat_is_rejected_before_model_lease() {
        let leases = Arc::new(AtomicU64::new(0));
        let controller = Arc::new(StubModelInstanceController {
            backend: Arc::new(FakeBackend::new("test-model", "ready")),
            leases: Arc::clone(&leases),
            pending_acquisition_dropped: None,
            acquisition_fails: false,
        });
        let response = app(AppState::model_free().with_model_controller(controller))
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "model": "test-model",
                            "messages": [{"role": "assistant", "content": null}],
                            "stream": true
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        assert_eq!(leases.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn ordinary_stream_waits_for_acquisition_before_opening() {
        let dropped = Arc::new(AtomicBool::new(false));
        let controller = Arc::new(StubModelInstanceController {
            backend: Arc::new(FakeBackend::new("test-model", "ready")),
            leases: Arc::new(AtomicU64::new(0)),
            pending_acquisition_dropped: Some(Arc::clone(&dropped)),
            acquisition_fails: false,
        });
        let mut response = Box::pin(
            app(AppState::model_free().with_model_controller(controller)).oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(minimal_request().to_string()))
                    .unwrap(),
            ),
        );
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(10), &mut response)
                .await
                .is_err()
        );
        drop(response);

        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            while !dropped.load(Ordering::Acquire) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("dropping the pending HTTP request must cancel model acquisition");
    }

    #[tokio::test]
    async fn progress_stream_opens_before_pending_model_acquisition_finishes() {
        let dropped = Arc::new(AtomicBool::new(false));
        let controller = Arc::new(StubModelInstanceController {
            backend: Arc::new(FakeBackend::new("test-model", "ready")),
            leases: Arc::new(AtomicU64::new(0)),
            pending_acquisition_dropped: Some(Arc::clone(&dropped)),
            acquisition_fails: false,
        });
        let response = tokio::time::timeout(
            std::time::Duration::from_secs(1),
            app(AppState::model_free().with_model_controller(controller)).oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .header("Magnitude-Include-Progress", "true")
                    .body(Body::from(minimal_request().to_string()))
                    .unwrap(),
            ),
        )
        .await
        .expect("progress response opens before model acquisition")
        .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        tokio::task::yield_now().await;
        drop(response);
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            while !dropped.load(Ordering::Acquire) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("dropping the open stream cancels its pending acquisition");
    }

    struct ScriptedBackend {
        events: Vec<domain::InferenceObservation>,
        fail: bool,
    }

    fn output_event(event: domain::InferenceOutputEvent) -> domain::InferenceObservation {
        domain::InferenceObservation::new(domain::InferenceObservationEvent::Output { event }, None)
    }

    fn delta_text(value: &str) -> domain::NonEmptyText {
        domain::NonEmptyText::try_new(value, "scripted delta").expect("nonempty fixture")
    }

    fn timed_output_event(
        event: domain::InferenceOutputEvent,
        generated_tokens: usize,
        decode_ms: f64,
    ) -> domain::InferenceObservation {
        domain::InferenceObservation::new(
            domain::InferenceObservationEvent::Output { event },
            Some(GenerationSnapshot {
                cached_prompt_tokens: 0,
                prompt_tokens: 11,
                generated_tokens,
                metrics: GenerationMetrics {
                    prompt_ms: 2.0,
                    decode_ms,
                    ..GenerationMetrics::default()
                },
            }),
        )
    }

    fn progress_event(progress: InferenceProgress) -> domain::InferenceObservation {
        domain::InferenceObservation::new(
            domain::InferenceObservationEvent::Progress { progress },
            None,
        )
    }

    impl CompletionBackend for ScriptedBackend {
        fn model_id(&self) -> &str {
            "test-model"
        }

        fn properties(&self) -> Result<ModelProperties, InferenceError> {
            FakeBackend::new("test-model", "").properties()
        }

        fn complete(
            &self,
            request: domain::ResolvedInferenceRequest,
            on_admitted: &mut dyn FnMut(u64) -> Result<(), InferenceError>,
            on_event: &mut dyn FnMut(domain::InferenceObservation) -> Result<(), InferenceError>,
        ) -> Result<domain::InferenceCompletion, InferenceError> {
            on_admitted(request.context().entries().len() as u64)?;
            if !matches!(
                self.events.first().map(domain::InferenceObservation::event),
                Some(domain::InferenceObservationEvent::Output {
                    event: domain::InferenceOutputEvent::Started
                })
            ) {
                on_event(output_event(domain::InferenceOutputEvent::Started))?;
            }
            for event in &self.events {
                on_event(event.clone())?;
            }
            if self.fail {
                return Err(InferenceError::Backend("scripted failure".into()));
            }
            let termination = if self.events.iter().any(|observation| {
                matches!(
                    observation.event(),
                    domain::InferenceObservationEvent::Output {
                        event: domain::InferenceOutputEvent::ToolCallFinished { .. }
                    }
                )
            }) {
                domain::Termination::ToolCalls
            } else {
                domain::Termination::Natural
            };
            Ok(domain::InferenceCompletion::new(
                domain::TokenUsage::new(11, 0, 7, 1),
                termination,
                GenerationMetrics {
                    queue_ms: 1.0,
                    prompt_ms: 2.0,
                    decode_ms: 3.0,
                    time_to_first_token_ms: 4.0,
                    prompt_tokens_per_second: 5.0,
                    decode_tokens_per_second: 6.0,
                    sampler_ms: 0.5,
                    parser_ms: 0.25,
                    draft_tokens: 0,
                    accepted_draft_tokens: 0,
                    draft_ms: 0.0,
                    verification_ms: 0.0,
                },
            ))
        }
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
        for (path, method) in [
            ("/api/v1/instances", "post"),
            ("/api/v1/models/{model_id}/properties", "post"),
            ("/api/v1/chat/templates/apply", "post"),
            ("/v1/chat/completions", "post"),
            ("/v1/responses", "post"),
        ] {
            let responses = &value["paths"][path][method]["responses"];
            for status in ["400", "404", "409", "422", "500"] {
                assert!(
                    responses[status].is_object(),
                    "{method} {path} must declare inventory error status {status}",
                );
            }
        }
    }

    #[test]
    fn speculative_counts_are_exposed_only_when_drafting_ran() {
        let ordinary = timing_values(0, 10, 2, &GenerationMetrics::default());
        assert_eq!(ordinary.draft_n, None);
        assert_eq!(ordinary.draft_n_accepted, None);

        let speculative = timing_values(
            0,
            10,
            4,
            &GenerationMetrics {
                draft_tokens: 3,
                accepted_draft_tokens: 2,
                ..GenerationMetrics::default()
            },
        );
        assert_eq!(speculative.draft_n, Some(3));
        assert_eq!(speculative.draft_n_accepted, Some(2));
    }

    #[tokio::test]
    async fn private_routes_require_the_owner_capability_but_health_does_not() {
        let service = app(AppState::new(FakeBackend::new("test-model", "ok"))
            .with_authorization("private-capability"));
        let health = service
            .clone()
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(health.status(), StatusCode::OK);

        let denied = service
            .clone()
            .oneshot(
                Request::post("/api/v1/models/test-model/properties")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(denied.status(), StatusCode::UNAUTHORIZED);

        let allowed = service
            .oneshot(
                Request::post("/api/v1/models/test-model/properties")
                    .header("authorization", "Bearer private-capability")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"model":"test-model"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(allowed.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn exposes_typed_properties_and_template_preparation() {
        let service = app(AppState::new(FakeBackend::new("test-model", "ok")));
        let properties = service
            .clone()
            .oneshot(
                Request::post("/api/v1/models/test-model/properties")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"model":"test-model"}"#))
                    .unwrap(),
            )
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
                Request::post("/api/v1/chat/templates/apply")
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
    async fn rejects_a_model_that_does_not_match_the_exact_instance_configuration() {
        let (status, body) = post_chat(
            FakeBackend::new("test-model", "ok"),
            json!({
                "model": "different-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": true
            }),
        )
        .await;
        assert_eq!(status, StatusCode::CONFLICT);
        assert!(body.contains("requested model instance is not ready"));
    }

    #[tokio::test]
    async fn accepts_the_loaded_model_id_and_configured_aliases() {
        let state =
            AppState::new(FakeBackend::new("test-model", "ok")).with_model_alias("friendly-name");
        let lease = state
            .model_controller
            .as_ref()
            .expect("model controller")
            .acquire_for_inference("friendly-name".to_owned(), None)
            .await
            .expect("instance lease");

        assert!(validate_model_selection(Some("test-model"), &lease).is_ok());
        assert!(validate_model_selection(Some("friendly-name"), &lease).is_ok());
        assert!(validate_model_selection(None, &lease).is_ok());
        assert!(validate_model_selection(Some("different-model"), &lease).is_err());
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
                    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
                    "strict": true
                }},
                {"type": "function", "function": {
                    "name": "other",
                    "parameters": {"type": "object"}
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
            "chat_template_kwargs": {"custom": 7},
            "stop": ["END", "STOP"],
            "max_completion_tokens": 99,
            "temperature": 0.25,
            "top_p": 0.75,
            "seed": 9,
            "stream": true,
            "stream_options": {"include_usage": true},
            "cache_prompt": false,
            "ignore_eos": true,
            "timings_per_token": true
        }));

        let (request, include_usage) = validate_test_request(request).unwrap();
        assert!(include_usage);
        assert_eq!(
            request.context().system().map(domain::NonEmptyText::as_str),
            Some("system")
        );
        assert_eq!(request.context().entries().len(), 2);
        let domain::ContextEntry::User { entry } = &request.context().entries()[0] else {
            panic!("first entry must be user")
        };
        assert!(matches!(
            entry.content(),
            [domain::UserContent::Text { text }, domain::UserContent::Image { image }]
                if text.as_str() == "look" && image.media_type() == "image/png"
        ));
        let domain::ContextEntry::Assistant { entry } = &request.context().entries()[1] else {
            panic!("second entry must be assistant")
        };
        assert_eq!(
            entry.reasoning().map(domain::NonEmptyText::as_str),
            Some("because")
        );
        let exchange = &entry.tool_calls()[0];
        assert_eq!(exchange.call().name().as_str(), "lookup");
        assert_eq!(exchange.call().id().as_str(), "call-1");
        assert_eq!(request.tools().definitions().len(), 2);
        assert_eq!(request.tools().definitions()[0].name().as_str(), "lookup");
        assert!(matches!(
            request.tools().choice(),
            domain::ToolChoice::Allowed { names, required: true }
                if names[0].as_str() == "lookup"
        ));
        assert_eq!(
            request.tools().parallelism(),
            domain::ToolParallelism::Sequential
        );
        assert_eq!(request.reasoning().effort().as_str(), "high");
        assert_eq!(request.reasoning().controls().enable_thinking, Some(true));
        assert_eq!(
            request.reasoning().explicit_budget().map(NonZeroU32::get),
            Some(64)
        );
        assert_eq!(
            request.reasoning().controls().template_args.get("custom"),
            Some(&json!(7))
        );
        match request.output() {
            domain::OutputConstraint::JsonSchema { constraint } => {
                assert_eq!(constraint.name(), "answer");
                assert_eq!(constraint.schema().as_map()["type"], "object");
                assert!(constraint.strict());
            }
            response => panic!("unexpected response format: {response:?}"),
        }
        assert_eq!(
            request
                .generation()
                .stop_sequences()
                .iter()
                .map(domain::StopSequence::as_str)
                .collect::<Vec<_>>(),
            ["END", "STOP"]
        );
        assert_eq!(
            request
                .generation()
                .max_output_tokens()
                .map(NonZeroU32::get),
            Some(99)
        );
        assert_eq!(request.generation().sampling().temperature().get(), 0.25);
        assert_eq!(request.generation().sampling().top_p().get(), 0.75);
        assert_eq!(request.generation().sampling().seed(), 9);
        assert_eq!(request.prompt_reuse(), domain::PromptReusePolicy::Disabled);
        assert_eq!(
            request.generation().end_of_generation(),
            domain::EndOfGenerationPolicy::IgnoreModelEnd
        );
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

        let error = validate_test_request(request_from_json(request)).unwrap_err();
        assert!(
            error
                .body
                .error
                .message
                .contains("network URLs are not supported")
        );
    }

    #[test]
    fn rejects_partial_or_separated_tool_exchanges_before_model_admission() {
        let mut dangling = minimal_request();
        dangling["messages"] = json!([
            {"role": "user", "content": "find it"},
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"}
                }]
            }
        ]);
        let error = validate_test_request(request_from_json(dangling)).unwrap_err();
        assert!(
            error
                .body
                .error
                .message
                .contains("no immediately following result")
        );

        let mut separated = minimal_request();
        separated["messages"] = json!([
            {"role": "user", "content": "find it"},
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"}
                }]
            },
            {"role": "user", "content": "interrupt"},
            {"role": "tool", "tool_call_id": "call-1", "content": "result"}
        ]);
        let error = validate_test_request(request_from_json(separated)).unwrap_err();
        assert!(
            error
                .body
                .error
                .message
                .contains("no immediately following result")
        );
    }

    #[test]
    fn preserves_model_defaults_when_optional_controls_are_omitted() {
        let (request, include_usage) =
            validate_test_request(request_from_json(minimal_request())).unwrap();
        assert!(!include_usage);
        assert_eq!(request.tools().choice(), &domain::ToolChoice::Auto);
        assert_eq!(
            request.tools().parallelism(),
            domain::ToolParallelism::Parallel
        );
        assert_eq!(request.reasoning().effort().as_str(), "high");
        assert!(request.reasoning().explicit_budget().is_none());
        assert_eq!(request.output(), &domain::OutputConstraint::Text);
        assert!(request.reasoning().controls().template_args.is_empty());
        assert!(request.generation().stop_sequences().is_empty());
        assert_eq!(request.prompt_reuse(), domain::PromptReusePolicy::Allowed);
        assert_eq!(
            request.generation().end_of_generation(),
            domain::EndOfGenerationPolicy::StopAtModelEnd
        );
    }

    #[test]
    fn history_preservation_does_not_conflict_with_reasoning_effort() {
        let mut request = minimal_request();
        request["reasoning_effort"] = json!("high");
        request["chat_template_kwargs"] = json!({"preserve_thinking": false});

        let (request, _) = validate_test_request(request_from_json(request)).unwrap();

        assert_eq!(request.reasoning().effort().as_str(), "high");
        assert_eq!(
            request
                .reasoning()
                .controls()
                .template_args
                .get("preserve_thinking"),
            Some(&json!(false))
        );
    }

    #[test]
    fn timing_control_accepts_tolerant_boolean_semantics() {
        for value in [
            JsonValue::Null,
            json!(false),
            json!("true"),
            json!(1),
            json!({"enabled": true}),
        ] {
            let mut request = minimal_request();
            request["timings_per_token"] = value;
            let request = validate_request(request_from_json(request)).unwrap();
            assert!(!request.timings_per_token);
        }

        let mut request = minimal_request();
        request["timings_per_token"] = json!(true);
        let request = validate_request(request_from_json(request)).unwrap();
        assert!(request.timings_per_token);
    }

    #[test]
    fn maps_grammar_response_format() {
        let mut request = minimal_request();
        request["response_format"] = json!({
            "type": "grammar",
            "grammar": "root ::= \"yes\" | \"no\""
        });

        let (request, _) = validate_test_request(request_from_json(request)).unwrap();
        assert!(matches!(
            request.output(),
            domain::OutputConstraint::Grammar { constraint }
                if constraint.as_str() == "root ::= \"yes\" | \"no\""
        ));
    }

    #[test]
    fn rejects_conflicting_or_lossy_request_controls() {
        let mut request = minimal_request();
        request["reasoning_effort"] = json!("none");
        request["thinking_budget_tokens"] = json!(10);
        let error = validate_test_request(request_from_json(request)).unwrap_err();
        assert!(error.body.error.message.contains("reasoning is disabled"));

        let mut request = minimal_request();
        request["reasoning_effort"] = json!("high");
        request["chat_template_kwargs"] = json!({"enable_thinking": true});
        let error = validate_test_request(request_from_json(request)).unwrap_err();
        assert!(error.body.error.message.contains("conflicts"));

        let mut request = minimal_request();
        request["reasoning_effort"] = json!("medium");
        let (request, _) = validate_test_request(request_from_json(request)).unwrap();
        assert_eq!(request.reasoning().effort().as_str(), "high");

        let mut request = minimal_request();
        request["tools"] = json!([{"type": "function", "function": {
            "name": "known", "parameters": {"type": "object"}
        }}]);
        request["tool_choice"] = json!({
            "type": "function", "function": {"name": "missing"}
        });
        let error = validate_test_request(request_from_json(request)).unwrap_err();
        assert!(error.body.error.message.contains("undefined tool"));

        let mut request = minimal_request();
        request["response_format"] = json!({
            "type": "json_schema",
            "json_schema": {"name": "bad", "schema": 42}
        });
        let error = validate_test_request(request_from_json(request)).unwrap_err();
        assert!(error.body.error.message.contains("JSON Schema"));

        let mut request = minimal_request();
        request["response_format"] = json!({"type": "grammar", "grammar": ""});
        let error = validate_test_request(request_from_json(request)).unwrap_err();
        assert!(
            error
                .body
                .error
                .message
                .contains("grammar must not be empty")
        );

        let mut request = minimal_request();
        request["stop"] = json!(["END", "END"]);
        let error = validate_test_request(request_from_json(request)).unwrap_err();
        assert!(error.body.error.message.contains("duplicate stop"));
    }

    #[test]
    fn normalizes_disabled_aliases_to_the_none_mapping() {
        let mut request = minimal_request();
        request["reasoning_effort"] = json!("off");
        let (request, _) = validate_test_request(request_from_json(request)).unwrap();
        assert_eq!(request.reasoning().effort().as_str(), "none");
        assert_eq!(request.reasoning().controls().enable_thinking, Some(false));
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
    async fn chat_defaults_to_one_non_streaming_completion() {
        let response = app(AppState::new(FakeBackend::new("test-model", "hello world")))
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "model": "test-model",
                            "messages": [{"role": "user", "content": "hi"}]
                        })
                        .to_string(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get("content-type").unwrap(),
            "application/json"
        );
        let body: Value =
            serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes())
                .unwrap();
        assert_eq!(body["object"], "chat.completion");
        assert_eq!(body["choices"][0]["message"]["role"], "assistant");
        assert_eq!(body["choices"][0]["message"]["content"], "hello world");
        assert_eq!(body["choices"][0]["finish_reason"], "stop");
        assert_eq!(body["usage"]["completion_tokens"], 2);
    }

    #[tokio::test]
    async fn streams_cumulative_native_timings_on_group_terminal_deltas() {
        let backend = ScriptedBackend {
            events: vec![
                output_event(domain::InferenceOutputEvent::ReasoningDelta {
                    text: delta_text("buffered group prefix"),
                }),
                timed_output_event(
                    domain::InferenceOutputEvent::TextDelta {
                        text: delta_text("first group end"),
                    },
                    1,
                    0.001,
                ),
                timed_output_event(
                    domain::InferenceOutputEvent::TextDelta {
                        text: delta_text("second group"),
                    },
                    2,
                    4.0,
                ),
            ],
            fail: false,
        };
        let mut request = minimal_request();
        request["timings_per_token"] = json!(true);

        let (status, body) = post_chat(backend, request).await;
        assert_eq!(status, StatusCode::OK);
        let chunks = stream_json(&body);

        assert_eq!(chunks.len(), 5);
        assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
        assert!(chunks[0].get("timings").is_none());
        assert_eq!(
            chunks[1]["choices"][0]["delta"]["reasoning_content"],
            "buffered group prefix"
        );
        assert!(chunks[1].get("timings").is_none());
        assert_eq!(chunks[2]["timings"]["predicted_n"], 1);
        assert_eq!(chunks[2]["timings"]["predicted_ms"], 0.001);
        assert_eq!(chunks[3]["timings"]["predicted_n"], 2);
        assert_eq!(chunks[3]["timings"]["predicted_ms"], 4.0);

        let terminal = &chunks[4];
        assert_eq!(terminal["choices"][0]["finish_reason"], "stop");
        assert_eq!(terminal["timings"]["predicted_n"], 7);
        let timing_fields = terminal["timings"]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        assert_eq!(
            timing_fields,
            BTreeSet::from([
                "cache_n",
                "parser_ms",
                "predicted_ms",
                "predicted_n",
                "predicted_per_second",
                "predicted_per_token_ms",
                "prompt_ms",
                "prompt_n",
                "prompt_per_second",
                "prompt_per_token_ms",
                "sampler_ms",
                "time_to_first_token_ms",
            ])
        );
        assert_eq!(terminal["timings"]["cache_n"], 0);
        assert_eq!(terminal["timings"]["prompt_n"], 11);
        assert_eq!(terminal["timings"]["prompt_per_second"], 5_500.0);
        assert_eq!(terminal["timings"]["time_to_first_token_ms"], 4.0);
        assert!(
            (terminal["timings"]["prompt_per_token_ms"].as_f64().unwrap() - 2.0 / 11.0).abs()
                < f64::EPSILON
        );
        assert!(
            (terminal["timings"]["predicted_per_token_ms"]
                .as_f64()
                .unwrap()
                - 3.0 / 7.0)
                .abs()
                < f64::EPSILON
        );
    }

    #[tokio::test]
    async fn first_sample_without_semantic_delta_attaches_timing_to_role() {
        let backend = ScriptedBackend {
            events: vec![timed_output_event(
                domain::InferenceOutputEvent::Started,
                1,
                0.001,
            )],
            fail: false,
        };
        let mut request = minimal_request();
        request["timings_per_token"] = json!(true);
        request["stream_options"] = json!({"include_usage": true});

        let (status, body) = post_chat(backend, request).await;
        assert_eq!(status, StatusCode::OK);
        let chunks = stream_json(&body);

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
        assert!(chunks[0]["choices"][0]["delta"]["content"].is_null());
        assert_eq!(chunks[0]["timings"]["predicted_n"], 1);
        assert_eq!(chunks[0]["timings"]["predicted_ms"], 0.001);
        assert_eq!(chunks[1]["choices"][0]["finish_reason"], "stop");
        assert!(chunks[1].get("timings").is_none());
        assert_eq!(chunks[2]["choices"], json!([]));
        assert_eq!(chunks[2]["timings"]["predicted_n"], 7);
    }

    #[tokio::test]
    async fn backend_signaled_stop_word_partial_timing_is_kept_when_flag_is_false() {
        let backend = ScriptedBackend {
            events: vec![timed_output_event(
                domain::InferenceOutputEvent::Started,
                1,
                0.001,
            )],
            fail: false,
        };
        let mut request = minimal_request();
        request["timings_per_token"] = json!(false);

        let (status, body) = post_chat(backend, request).await;
        assert_eq!(status, StatusCode::OK);
        let chunks = stream_json(&body);

        assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(chunks[0]["timings"]["predicted_n"], 1);
        assert_eq!(chunks[1]["choices"][0]["finish_reason"], "stop");
        assert_eq!(chunks[1]["timings"]["predicted_n"], 7);
    }

    #[tokio::test]
    async fn false_timing_control_suppresses_partial_snapshots_but_not_final_timings() {
        let backend = ScriptedBackend {
            events: vec![output_event(domain::InferenceOutputEvent::TextDelta {
                text: delta_text("answer"),
            })],
            fail: false,
        };
        let mut request = minimal_request();
        request["timings_per_token"] = json!(false);

        let (status, body) = post_chat(backend, request).await;
        assert_eq!(status, StatusCode::OK);
        let chunks = stream_json(&body);
        assert!(chunks[0].get("timings").is_none());
        assert!(chunks[1].get("timings").is_none());
        assert_eq!(chunks[2]["timings"]["predicted_n"], 7);
    }

    #[tokio::test]
    async fn streams_reasoning_content_tool_calls_finish_usage_and_timings() {
        let backend = ScriptedBackend {
            events: vec![
                output_event(domain::InferenceOutputEvent::ReasoningDelta {
                    text: delta_text("thought"),
                }),
                output_event(domain::InferenceOutputEvent::TextDelta {
                    text: delta_text("answer"),
                }),
                output_event(domain::InferenceOutputEvent::ToolCallStarted {
                    index: 0,
                    id: domain::ToolCallId::try_new("call-1").expect("valid"),
                    name: domain::ToolName::try_new("lookup").expect("valid"),
                }),
                output_event(domain::InferenceOutputEvent::ToolInputDelta {
                    index: 0,
                    json_fragment: delta_text("{}"),
                }),
                output_event(domain::InferenceOutputEvent::ToolCallFinished { index: 0 }),
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
        assert!(chunks[5].get("timings").is_none());
        assert_eq!(chunks[6]["choices"], json!([]));
        assert_eq!(chunks[6]["usage"]["prompt_tokens"], 11);
        assert_eq!(chunks[6]["usage"]["completion_tokens"], 7);
        assert_eq!(chunks[6]["usage"]["total_tokens"], 18);
        assert_eq!(chunks[6]["timings"]["prompt_ms"], 2.0);
        assert_eq!(chunks[6]["timings"]["predicted_per_second"], 7_000.0 / 3.0);
    }

    #[tokio::test]
    async fn backend_failure_is_an_explicit_stream_error_without_success_sentinel() {
        let backend = ScriptedBackend {
            events: vec![output_event(domain::InferenceOutputEvent::TextDelta {
                text: delta_text("partial"),
            })],
            fail: true,
        };
        let (status, body) = post_chat(backend, minimal_request()).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.contains("event: error"));
        assert!(!body.contains("data: [DONE]"));
        let chunks = stream_json(&body);
        let error = chunks.last().unwrap();
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
