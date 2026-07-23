use std::collections::{BTreeMap, BTreeSet};
use std::convert::Infallible;
use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::RwLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::{Path, Request, State};
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::{future::BoxFuture, stream::BoxStream};
use icn_contracts::models::{
    AssessModelsRequest, AssessModelsResponse, DownloadAttempt, DownloadAttemptId,
    FitModelsRequest, FitModelsResponse, InstalledModelPackages, InstalledModelPackagesResponse,
    LoadModelRequest, ModelDownloads, ModelDownloadsResponse, ModelEvaluator, ModelLoadEvent,
    ModelPackageId, RecommendableModelCatalog, RecommendableModelCatalogProvider,
    RemoveInstalledModelPackageResponse, StartModelDownloadRequest, StartModelDownloadResponse,
};
use icn_contracts::{
    AllowedToolsMode, CacheType, ChatContent, ChatContentPart, ChatMessage, ChatRequest, ChatRole,
    ChatTemplateRequest, CompletionBackend, ExecutionConfig, ExecutionConfigReport, FinishReason,
    FlashAttention, Generation, GenerationMetrics, GenerationSnapshot, GpuLayers, GrammarTrigger,
    HardwareProvider, HardwareSnapshot, HuggingFaceModelCatalog, HuggingFaceModelSearchRequest,
    HuggingFaceModelSearchResults, HuggingFaceRepositoryRequest, HuggingFaceRepositorySnapshot,
    ImageInput, InferenceError, InferenceEvent, InferenceStreamEvent, InventoryError,
    ModelModalities, ModelProperties, PreparedChatInfo, ReasoningControl, ResponseFormat,
    SplitMode, TemplateCapabilities, ToolCall, ToolChoice, ToolDefinition,
};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value as JsonValue;
use tokio::sync::{Notify, mpsc};
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

#[derive(Clone)]
pub struct AppState {
    backends: BackendRegistry,
    installed_packages: Option<Arc<dyn InstalledModelPackages>>,
    recommendable_catalog: Option<Arc<dyn RecommendableModelCatalogProvider>>,
    model_evaluator: Option<Arc<dyn ModelEvaluator>>,
    model_downloads: Option<Arc<dyn ModelDownloads>>,
    hardware: Option<Arc<dyn HardwareProvider>>,
    hugging_face_catalog: Option<Arc<dyn HuggingFaceModelCatalog>>,
    runtime: Option<Arc<dyn RuntimeController>>,
    identity: ServerIdentity,
    authorization: Option<Arc<str>>,
    next_id: Arc<AtomicU64>,
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

#[derive(Clone)]
pub struct BackendRegistry {
    inner: Arc<RwLock<BackendRegistryState>>,
    active_leases: Arc<AtomicU64>,
    mutating: Arc<AtomicBool>,
    mutation_available: Arc<Notify>,
    lease_released: Arc<Notify>,
}

struct BackendRegistryState {
    backend: Option<Arc<dyn CompletionBackend>>,
    model_id: Option<String>,
    properties: Option<Result<ModelProperties, String>>,
    model_aliases: Arc<BTreeSet<String>>,
    generation: u64,
}

pub struct BackendLease {
    backend: Arc<dyn CompletionBackend>,
    model_aliases: Arc<BTreeSet<String>>,
    generation: u64,
    active_leases: Arc<AtomicU64>,
    lease_released: Arc<Notify>,
}

pub struct BackendMutationGuard {
    mutating: Arc<AtomicBool>,
    mutation_available: Arc<Notify>,
}

#[derive(Clone)]
pub struct BackendObservation {
    pub model_id: String,
    pub properties: Result<ModelProperties, String>,
}

impl Drop for BackendMutationGuard {
    fn drop(&mut self) {
        self.mutating.store(false, Ordering::Release);
        self.mutation_available.notify_one();
    }
}

impl BackendLease {
    pub fn backend(&self) -> &Arc<dyn CompletionBackend> {
        &self.backend
    }

    pub fn model_id(&self) -> &str {
        self.backend.model_id()
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    fn accepts_model(&self, requested: &str) -> bool {
        requested == self.backend.model_id() || self.model_aliases.contains(requested)
    }
}

impl Drop for BackendLease {
    fn drop(&mut self) {
        self.active_leases.fetch_sub(1, Ordering::AcqRel);
        self.lease_released.notify_waiters();
    }
}

impl BackendRegistry {
    pub fn empty() -> Self {
        Self {
            inner: Arc::new(RwLock::new(BackendRegistryState {
                backend: None,
                model_id: None,
                properties: None,
                model_aliases: Arc::new(BTreeSet::new()),
                generation: 0,
            })),
            active_leases: Arc::new(AtomicU64::new(0)),
            mutating: Arc::new(AtomicBool::new(false)),
            mutation_available: Arc::new(Notify::new()),
            lease_released: Arc::new(Notify::new()),
        }
    }

    pub fn with_backend(backend: Arc<dyn CompletionBackend>) -> Self {
        let registry = Self::empty();
        registry.replace(backend, BTreeSet::new());
        registry
    }

    pub fn lease(&self) -> Option<BackendLease> {
        if self.mutating.load(Ordering::Acquire) {
            return None;
        }
        let state = self.inner.read().ok()?;
        let backend = state.backend.clone()?;
        self.active_leases.fetch_add(1, Ordering::AcqRel);
        if self.mutating.load(Ordering::Acquire) {
            self.active_leases.fetch_sub(1, Ordering::AcqRel);
            self.lease_released.notify_waiters();
            return None;
        }
        Some(BackendLease {
            backend,
            model_aliases: Arc::clone(&state.model_aliases),
            generation: state.generation,
            active_leases: Arc::clone(&self.active_leases),
            lease_released: Arc::clone(&self.lease_released),
        })
    }

    pub fn active_leases(&self) -> u64 {
        self.active_leases.load(Ordering::Acquire)
    }

    /// Immutable metadata copied when the backend generation is published.
    /// Observation never borrows the executable backend or changes idleness.
    pub fn observe(&self) -> Option<BackendObservation> {
        let state = self.inner.read().ok()?;
        Some(BackendObservation {
            model_id: state.model_id.clone()?,
            properties: state.properties.clone()?,
        })
    }

    pub fn add_alias(&self, alias: String) {
        let mut state = self.inner.write().expect("backend registry lock poisoned");
        let mut aliases = (*state.model_aliases).clone();
        aliases.insert(alias);
        state.model_aliases = Arc::new(aliases);
    }

    pub fn try_begin_mutation(&self) -> Option<BackendMutationGuard> {
        self.mutating
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .ok()?;
        if self.active_leases() > 0 {
            self.mutating.store(false, Ordering::Release);
            self.mutation_available.notify_one();
            return None;
        }
        Some(BackendMutationGuard {
            mutating: Arc::clone(&self.mutating),
            mutation_available: Arc::clone(&self.mutation_available),
        })
    }

    pub async fn begin_mutation(&self) -> BackendMutationGuard {
        loop {
            let available = self.mutation_available.notified();
            if self
                .mutating
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                // Closing mutation admission first prevents an unbounded sequence of new
                // inference leases from starving a queued replacement or unload.
                while self.active_leases() > 0 {
                    let released = self.lease_released.notified();
                    if self.active_leases() > 0 {
                        released.await;
                    }
                }
                return BackendMutationGuard {
                    mutating: Arc::clone(&self.mutating),
                    mutation_available: Arc::clone(&self.mutation_available),
                };
            }
            available.await;
        }
    }

    pub fn replace(&self, backend: Arc<dyn CompletionBackend>, aliases: BTreeSet<String>) -> u64 {
        let model_id = backend.model_id().to_owned();
        let properties = backend.properties().map_err(|error| error.to_string());
        let mut state = self.inner.write().expect("backend registry lock poisoned");
        state.generation = state.generation.saturating_add(1);
        state.backend = Some(backend);
        state.model_id = Some(model_id);
        state.properties = Some(properties);
        state.model_aliases = Arc::new(aliases);
        let generation = state.generation;
        drop(state);
        generation
    }

    pub fn clear(&self) -> u64 {
        let mut state = self.inner.write().expect("backend registry lock poisoned");
        state.generation = state.generation.saturating_add(1);
        state.backend = None;
        state.model_id = None;
        state.properties = None;
        state.model_aliases = Arc::new(BTreeSet::new());
        let generation = state.generation;
        drop(state);
        generation
    }

    pub fn generation(&self) -> u64 {
        self.inner.read().map(|state| state.generation).unwrap_or(0)
    }
}

impl AppState {
    pub fn new(backend: impl CompletionBackend) -> Self {
        Self::from_shared_backend(Arc::new(backend))
    }

    /// Construct API state from a backend shared with another server-owned service.
    pub fn from_shared_backend(backend: Arc<dyn CompletionBackend>) -> Self {
        Self {
            backends: BackendRegistry::with_backend(backend),
            installed_packages: None,
            recommendable_catalog: None,
            model_evaluator: None,
            model_downloads: None,
            hardware: None,
            hugging_face_catalog: None,
            runtime: None,
            identity: ServerIdentity::default(),
            authorization: None,
            next_id: Arc::new(AtomicU64::new(1)),
        }
    }

    pub fn model_free(backends: BackendRegistry) -> Self {
        Self {
            backends,
            installed_packages: None,
            recommendable_catalog: None,
            model_evaluator: None,
            model_downloads: None,
            hardware: None,
            hugging_face_catalog: None,
            runtime: None,
            identity: ServerIdentity::default(),
            authorization: None,
            next_id: Arc::new(AtomicU64::new(1)),
        }
    }

    /// Accept an additional OpenAI request model name for the loaded backend.
    ///
    /// The backend's stable model ID remains authoritative; aliases are routing names only.
    pub fn with_model_alias(self, alias: impl Into<String>) -> Self {
        self.backends.add_alias(alias.into());
        self
    }

    pub fn with_installed_packages(
        mut self,
        installed_packages: Arc<dyn InstalledModelPackages>,
    ) -> Self {
        self.installed_packages = Some(installed_packages);
        self
    }

    pub fn with_recommendable_catalog(
        mut self,
        recommendable_catalog: Arc<dyn RecommendableModelCatalogProvider>,
    ) -> Self {
        self.recommendable_catalog = Some(recommendable_catalog);
        self
    }

    pub fn with_model_evaluator(mut self, model_evaluator: Arc<dyn ModelEvaluator>) -> Self {
        self.model_evaluator = Some(model_evaluator);
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

    pub fn with_runtime(mut self, runtime: Arc<dyn RuntimeController>) -> Self {
        self.runtime = Some(runtime);
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
}

pub fn app(state: AppState) -> Router {
    let mut protected = Router::new()
        .route("/v1/hardware", get(hardware))
        .route("/v1/models/installed", get(installed_models))
        .route(
            "/v1/models/installed/{package_id}",
            axum::routing::delete(remove_installed_model),
        )
        .route("/v1/models/catalog", get(recommendable_model_catalog))
        .route("/v1/models/assess", post(assess_models))
        .route("/v1/models/fit", post(fit_models))
        .route(
            "/v1/models/downloads",
            get(model_downloads).post(start_model_download),
        )
        .route("/v1/models/load", post(load_model_configuration))
        .route(
            "/v1/models/residencies/{residency_id}",
            axum::routing::delete(unload_model_residency),
        )
        .route(
            "/v1/models/downloads/{attempt_id}",
            get(model_download_attempt),
        )
        .route(
            "/v1/models/downloads/{attempt_id}/cancel",
            post(cancel_model_download),
        )
        .route(
            "/v1/hugging-face/models/search",
            post(search_hugging_face_models),
        )
        .route(
            "/v1/hugging-face/models/resolve",
            post(resolve_hugging_face_repository),
        )
        .route("/props", get(props))
        .route("/v1/props", get(props))
        .route("/apply-template", post(apply_template))
        .route("/v1/apply-template", post(apply_template))
        .route("/v1/chat/completions", post(chat_completions))
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

pub trait RuntimeController: Send + Sync + 'static {
    fn load_configuration(&self, request: LoadModelRequest) -> BoxStream<'static, ModelLoadEvent>;
    fn unload_residency(&self, residency_id: String) -> BoxFuture<'_, Result<(), InventoryError>>;
    fn remove_installed(
        &self,
        package_id: ModelPackageId,
    ) -> BoxFuture<'_, Result<RemoveInstalledModelPackageResponse, InventoryError>>;
    fn lease(
        &self,
        configuration_id: String,
    ) -> BoxFuture<'_, Result<BackendLease, InventoryError>>;
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
    pub default_reasoning_effort: String,
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

#[derive(Debug, Clone, Deserialize, ToSchema)]
#[serde(transparent)]
pub struct ReasoningEffortRequest(pub String);

impl ReasoningEffortRequest {
    fn normalize(&self) -> Result<icn_contracts::NormalizedReasoningEffort, ApiError> {
        icn_contracts::NormalizedReasoningEffort::parse(&self.0).ok_or_else(|| {
            ApiError::invalid(format!("unsupported reasoning_effort spelling: {}", self.0))
        })
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
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct Timings {
    pub cache_n: u64,
    pub prompt_n: u64,
    pub prompt_ms: f64,
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
    pub code: String,
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
                    code: "backend_error".to_owned(),
                    retryable: true,
                },
            },
        }
    }

    fn conflict(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::CONFLICT,
            body: ErrorResponse {
                error: ApiErrorBody {
                    message: message.into(),
                    r#type: "invalid_request_error",
                    code: code.to_owned(),
                    retryable: false,
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
            InventoryError::Runtime {
                code, retryable, ..
            } => (
                StatusCode::CONFLICT,
                "runtime_error",
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
                    code: code.to_owned(),
                    retryable,
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

#[utoipa::path(post, path = "/v1/models/load", operation_id = "loadModelConfiguration", tag = "models",
    request_body(content = LoadModelRequest, content_type = "application/json"),
    responses(
        (status = 200, description = "Exact model configuration load progress", body = String, content_type = "text/event-stream"),
        (status = 500, description = "Runtime control unavailable", body = ErrorResponse)
    )
)]
#[tracing::instrument(name = "icn.model_configuration.load", skip_all, err(Debug))]
async fn load_model_configuration(
    State(state): State<AppState>,
    Json(request): Json<LoadModelRequest>,
) -> Result<Sse<impl tokio_stream::Stream<Item = Result<Event, Infallible>>>, ApiError> {
    let runtime = state
        .runtime
        .as_ref()
        .ok_or_else(|| ApiError::server("runtime control is not configured"))?;
    let events = runtime.load_configuration(request);
    let framed = tokio_stream::StreamExt::map(events, |event| {
        let data = serde_json::to_string(&event).unwrap_or_else(|error| {
            serde_json::json!({
                "_tag": "Failed",
                "failure": {
                    "code": "serialization_failed",
                    "message": error.to_string(),
                    "retryable": false
                }
            })
            .to_string()
        });
        Ok(Event::default().data(data))
    });
    Ok(Sse::new(framed).keep_alive(KeepAlive::default()))
}

#[utoipa::path(delete, path = "/v1/models/residencies/{residency_id}", operation_id = "unloadModelResidency", tag = "models",
    params(("residency_id" = String, Path, description = "Runtime residency ID")),
    responses(
        (status = 204, description = "Residency is no longer loaded"),
        (status = 404, description = "Residency not found", body = ErrorResponse),
        (status = 500, description = "Runtime unload failed", body = ErrorResponse)
    )
)]
#[tracing::instrument(name = "icn.model_residency.unload", skip_all, err(Debug))]
async fn unload_model_residency(
    State(state): State<AppState>,
    Path(residency_id): Path<String>,
) -> Result<StatusCode, ApiError> {
    let runtime = state
        .runtime
        .as_ref()
        .ok_or_else(|| ApiError::server("runtime control is not configured"))?;
    runtime
        .unload_residency(residency_id)
        .await
        .map(|()| StatusCode::NO_CONTENT)
        .map_err(ApiError::from_inventory)
}

#[utoipa::path(get, path = "/v1/hardware", operation_id = "getHardware", tag = "system", responses(
    (status = 200, description = "Hardware visible to the pinned ICN runtime", body = HardwareSnapshot),
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

#[utoipa::path(post, path = "/v1/hugging-face/models/search", operation_id = "searchHuggingFaceModels", tag = "hugging-face",
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

#[utoipa::path(post, path = "/v1/hugging-face/models/resolve", operation_id = "resolveHuggingFaceRepository", tag = "hugging-face",
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

#[utoipa::path(get, path = "/v1/models/installed", operation_id = "listInstalledModels", tag = "models",
    responses(
        (status = 200, description = "Verified model packages installed on this machine", body = InstalledModelPackagesResponse),
        (status = 500, description = "Installed package discovery failed", body = ErrorResponse)
    )
)]
#[tracing::instrument(name = "icn.models.installed", skip_all, err(Debug))]
async fn installed_models(
    State(state): State<AppState>,
) -> Result<Json<InstalledModelPackagesResponse>, ApiError> {
    let installed = state
        .installed_packages
        .as_ref()
        .ok_or_else(|| ApiError::server("installed model packages are not configured"))?;
    installed
        .list_installed()
        .await
        .map(Json)
        .map_err(ApiError::from_inventory)
}

#[utoipa::path(delete, path = "/v1/models/installed/{package_id}", operation_id = "removeInstalledModel", tag = "models",
    params(("package_id" = String, Path, description = "Canonical model package identity")),
    responses(
        (status = 200, description = "Installed package removal result", body = RemoveInstalledModelPackageResponse),
        (status = 404, description = "Installed package not found", body = ErrorResponse),
        (status = 500, description = "Installed package removal failed", body = ErrorResponse)
    )
)]
#[tracing::instrument(name = "icn.models.installed.remove", skip_all, fields(model.package.id = %package_id), err(Debug))]
async fn remove_installed_model(
    State(state): State<AppState>,
    Path(package_id): Path<String>,
) -> Result<Json<RemoveInstalledModelPackageResponse>, ApiError> {
    let runtime = state
        .runtime
        .as_ref()
        .ok_or_else(|| ApiError::server("runtime control is not configured"))?;
    runtime
        .remove_installed(ModelPackageId(package_id))
        .await
        .map(Json)
        .map_err(ApiError::from_inventory)
}

#[utoipa::path(get, path = "/v1/models/catalog", operation_id = "getRecommendableModelCatalog", tag = "models",
    responses(
        (status = 200, description = "Resolved recommendable model catalog", body = RecommendableModelCatalog),
        (status = 500, description = "Catalog publication failed", body = ErrorResponse)
    )
)]
#[tracing::instrument(name = "icn.models.catalog", skip_all, err(Debug))]
async fn recommendable_model_catalog(
    State(state): State<AppState>,
) -> Result<Json<RecommendableModelCatalog>, ApiError> {
    let catalog = state
        .recommendable_catalog
        .as_ref()
        .ok_or_else(|| ApiError::server("recommendable model catalog is not configured"))?;
    catalog
        .catalog()
        .await
        .map(Json)
        .map_err(ApiError::from_inventory)
}

#[utoipa::path(post, path = "/v1/models/assess", operation_id = "assessModels", tag = "models",
    request_body(content = AssessModelsRequest, content_type = "application/json"),
    responses(
        (status = 200, description = "Exact target and profile assessments", body = AssessModelsResponse),
        (status = 500, description = "Assessment operation failed", body = ErrorResponse)
    )
)]
#[tracing::instrument(name = "icn.models.assess", skip_all, err(Debug))]
async fn assess_models(
    State(state): State<AppState>,
    Json(request): Json<AssessModelsRequest>,
) -> Result<Json<AssessModelsResponse>, ApiError> {
    let evaluator = state
        .model_evaluator
        .as_ref()
        .ok_or_else(|| ApiError::server("model evaluation is not configured"))?;
    evaluator
        .assess(request)
        .await
        .map(Json)
        .map_err(ApiError::from_inventory)
}

#[utoipa::path(post, path = "/v1/models/fit", operation_id = "fitModels", tag = "models",
    request_body(content = FitModelsRequest, content_type = "application/json"),
    responses(
        (status = 200, description = "Automatically selected target profiles", body = FitModelsResponse),
        (status = 500, description = "Automatic fitting operation failed", body = ErrorResponse)
    )
)]
#[tracing::instrument(name = "icn.models.fit", skip_all, err(Debug))]
async fn fit_models(
    State(state): State<AppState>,
    Json(request): Json<FitModelsRequest>,
) -> Result<Json<FitModelsResponse>, ApiError> {
    let evaluator = state
        .model_evaluator
        .as_ref()
        .ok_or_else(|| ApiError::server("model evaluation is not configured"))?;
    evaluator
        .fit(request)
        .await
        .map(Json)
        .map_err(ApiError::from_inventory)
}

#[utoipa::path(post, path = "/v1/models/downloads", operation_id = "startModelDownload", tag = "models",
    request_body(content = StartModelDownloadRequest, content_type = "application/json"),
    responses(
        (status = 200, description = "Authoritative download attempt", body = StartModelDownloadResponse),
        (status = 500, description = "Download could not start", body = ErrorResponse)
    )
)]
#[tracing::instrument(name = "icn.models.downloads.start", skip_all, err(Debug))]
async fn start_model_download(
    State(state): State<AppState>,
    Json(request): Json<StartModelDownloadRequest>,
) -> Result<Json<StartModelDownloadResponse>, ApiError> {
    let downloads = state
        .model_downloads
        .as_ref()
        .ok_or_else(|| ApiError::server("model downloads are not configured"))?;
    downloads
        .start(request)
        .await
        .map(Json)
        .map_err(ApiError::from_inventory)
}

#[utoipa::path(get, path = "/v1/models/downloads", operation_id = "listModelDownloads", tag = "models",
    responses(
        (status = 200, description = "Retained model download attempts", body = ModelDownloadsResponse),
        (status = 500, description = "Download attempts unavailable", body = ErrorResponse)
    )
)]
#[tracing::instrument(name = "icn.models.downloads.list", skip_all, err(Debug))]
async fn model_downloads(
    State(state): State<AppState>,
) -> Result<Json<ModelDownloadsResponse>, ApiError> {
    let downloads = state
        .model_downloads
        .as_ref()
        .ok_or_else(|| ApiError::server("model downloads are not configured"))?;
    downloads
        .list_attempts()
        .await
        .map(Json)
        .map_err(ApiError::from_inventory)
}

#[utoipa::path(get, path = "/v1/models/downloads/{attempt_id}", operation_id = "getModelDownload", tag = "models",
    params(("attempt_id" = String, Path, description = "Download attempt ID")),
    responses(
        (status = 200, description = "Authoritative model download attempt", body = DownloadAttempt),
        (status = 404, description = "Attempt not found", body = ErrorResponse)
    )
)]
#[tracing::instrument(name = "icn.models.downloads.get", skip_all, err(Debug))]
async fn model_download_attempt(
    State(state): State<AppState>,
    Path(attempt_id): Path<String>,
) -> Result<Json<DownloadAttempt>, ApiError> {
    let downloads = state
        .model_downloads
        .as_ref()
        .ok_or_else(|| ApiError::server("model downloads are not configured"))?;
    downloads
        .get_attempt(&DownloadAttemptId(attempt_id))
        .await
        .map(Json)
        .map_err(ApiError::from_inventory)
}

#[utoipa::path(post, path = "/v1/models/downloads/{attempt_id}/cancel", operation_id = "cancelModelDownload", tag = "models",
    params(("attempt_id" = String, Path, description = "Download attempt ID")),
    responses(
        (status = 200, description = "Cancelled or terminal model download attempt", body = DownloadAttempt),
        (status = 404, description = "Attempt not found", body = ErrorResponse)
    )
)]
#[tracing::instrument(name = "icn.models.downloads.cancel", skip_all, err(Debug))]
async fn cancel_model_download(
    State(state): State<AppState>,
    Path(attempt_id): Path<String>,
) -> Result<Json<DownloadAttempt>, ApiError> {
    let downloads = state
        .model_downloads
        .as_ref()
        .ok_or_else(|| ApiError::server("model downloads are not configured"))?;
    downloads
        .cancel(&DownloadAttemptId(attempt_id))
        .await
        .map(Json)
        .map_err(ApiError::from_inventory)
}

#[utoipa::path(get, path = "/v1/props", operation_id = "getModelProperties", tag = "models", responses(
    (status = 200, description = "Loaded model and active template properties", body = PropsResponse),
    (status = 500, description = "Properties unavailable", body = ErrorResponse)
))]
#[tracing::instrument(
    name = "icn.model.properties",
    skip_all,
    fields(model.id = tracing::field::Empty),
    err(Debug)
)]
async fn props(State(state): State<AppState>) -> Result<Json<PropsResponse>, ApiError> {
    let backend = state.backends.observe().ok_or_else(|| {
        ApiError::conflict(
            "model_not_loaded",
            "no model is loaded by this inference node",
        )
    })?;
    tracing::Span::current().record("model.id", backend.model_id.as_str());
    let properties = backend.properties.map_err(ApiError::server)?;
    Ok(Json(props_response(properties)))
}

#[utoipa::path(post, path = "/v1/apply-template", operation_id = "applyChatTemplate", tag = "chat",
    request_body = ApplyTemplateRequest,
    responses(
        (status = 200, description = "Prepared native chat prompt and constraints", body = ApplyTemplateResponse),
        (status = 400, description = "Invalid request", body = ErrorResponse),
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
    let lease = require_backend(&state)?;
    tracing::Span::current().record("model.id", lease.model_id());
    validate_model_selection(request.model.as_deref(), &lease)?;
    let properties = lease
        .backend()
        .properties()
        .map_err(ApiError::from_inference)?;
    let request = validate_apply_template_request(request, &properties.reasoning)?;
    let span = tracing::Span::current();
    let prepared = tokio::task::spawn_blocking(move || {
        span.in_scope(|| lease.backend().apply_template(request))
    })
    .await
    .map_err(|error| ApiError::server(format!("template task failed: {error}")))?
    .map_err(ApiError::from_inference)?;
    Ok(Json(apply_template_response(prepared)))
}

#[utoipa::path(post, path = "/v1/chat/completions", operation_id = "createChatCompletion", tag = "chat",
    request_body = ChatCompletionRequest,
    responses(
        (status = 200, description = "OpenAI-compatible server-sent events", body = String, content_type = "text/event-stream"),
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
async fn chat_completions(
    State(state): State<AppState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    let request = validate_request(request)?;
    let model_id = request
        .model
        .clone()
        .filter(|model| !model.is_empty())
        .ok_or_else(|| ApiError::invalid("model is required"))?;
    let lease = if let Some(runtime) = state.runtime.as_ref() {
        runtime
            .lease(model_id)
            .await
            .map_err(ApiError::from_inventory)?
    } else {
        require_backend(&state)?
    };
    chat_completion_with_lease(state, request, lease).await
}

async fn chat_completion_with_lease(
    state: AppState,
    request: ValidatedChatRequest,
    lease: BackendLease,
) -> Result<Response, ApiError> {
    validate_model_selection(request.model.as_deref(), &lease)?;
    let properties = lease
        .backend()
        .properties()
        .map_err(ApiError::from_inference)?;
    let (request, include_usage) = finalize_request(request, &properties.reasoning)?;
    let id = format!(
        "chatcmpl-icn-{}",
        state.next_id.fetch_add(1, Ordering::Relaxed)
    );
    let created = unix_timestamp();
    let model = lease.model_id().to_owned();
    let current_span = tracing::Span::current();
    current_span.record("completion.id", id.as_str());
    current_span.record("model.id", model.as_str());
    let (sender, receiver) = mpsc::channel::<Result<Event, Infallible>>(16);

    let span = current_span;
    tokio::task::spawn_blocking(move || {
        span.in_scope(|| {
            let mut callback = |event: InferenceStreamEvent| {
                let InferenceStreamEvent {
                    delta: event,
                    timings,
                } = event;
                let delta = inference_event_delta(event)?;
                let timings = timings.map(|snapshot| snapshot_timings(&snapshot));
                if emit_chunk(
                    &sender,
                    &choice_chunk(&id, created, &model, delta, None, timings),
                ) {
                    Ok(())
                } else {
                    Err(InferenceError::Callback(
                        "stream consumer disconnected".into(),
                    ))
                }
            };
            let generation = match lease.backend().complete(request, &mut callback) {
                Ok(generation) => generation,
                Err(error) => {
                    tracing::error!(error = %error, "chat completion failed");
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
            tracing::info!(
                completion.id = %id,
                model.id = %model,
                finish.reason = reason,
                input.tokens = generation.prompt_tokens,
                output.tokens = generation.generated_tokens,
                queue.ms = generation.metrics.queue_ms,
                prompt.ms = generation.metrics.prompt_ms,
                decode.ms = generation.metrics.decode_ms,
                "chat completion finished"
            );
            let terminal_timings = (!include_usage).then(|| generation_timings(&generation));
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
            if include_usage
                && !emit_chunk(&sender, &usage_chunk(&id, created, &model, &generation))
            {
                return;
            }
            emit_done(&sender);
        });
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
        timings: Some(generation_timings(generation)),
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

fn generation_timings(generation: &Generation) -> Timings {
    timing_values(
        generation.cached_prompt_tokens,
        generation.prompt_tokens,
        generation.generated_tokens,
        &generation.metrics,
    )
}

fn snapshot_timings(snapshot: &GenerationSnapshot) -> Timings {
    timing_values(
        snapshot.cached_prompt_tokens,
        snapshot.prompt_tokens,
        snapshot.generated_tokens,
        &snapshot.metrics,
    )
}

fn timing_values(
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

fn inference_event_delta(event: InferenceEvent) -> Result<ChunkDelta, InferenceError> {
    Ok(match event {
        InferenceEvent::StreamStart => ChunkDelta {
            role: Some("assistant".into()),
            content: Some(None),
            ..ChunkDelta::default()
        },
        InferenceEvent::ContentDelta { text } => ChunkDelta {
            content: Some(Some(text)),
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
    let (error_type, code, retryable) = match error {
        InferenceError::InvalidConfig(_) => ("invalid_request_error", "invalid_request", false),
        InferenceError::Backend(_) => ("server_error", "backend_error", true),
        InferenceError::Cancelled => ("cancelled", "request_cancelled", true),
        InferenceError::Overloaded => ("server_error", "overloaded", true),
        InferenceError::ExecutorStopped => ("server_error", "executor_stopped", true),
        InferenceError::Callback(_) => ("server_error", "stream_callback_error", true),
    };
    ApiErrorBody {
        message: error.to_string(),
        r#type: error_type,
        code: code.to_owned(),
        retryable,
    }
}

fn require_backend(state: &AppState) -> Result<BackendLease, ApiError> {
    state.backends.lease().ok_or_else(|| {
        ApiError::conflict(
            "model_not_loaded",
            "no model is loaded by this inference node",
        )
    })
}

fn validate_model_selection(requested: Option<&str>, lease: &BackendLease) -> Result<(), ApiError> {
    match requested {
        Some("") => Err(ApiError::invalid("model must not be empty")),
        Some(requested) if !lease.accepts_model(requested) => Err(ApiError::invalid(format!(
            "model {requested} is not loaded by this inference node"
        ))),
        _ => Ok(()),
    }
}

fn validate_apply_template_request(
    request: ApplyTemplateRequest,
    reasoning_profile: &icn_contracts::ReasoningProfile,
) -> Result<ChatTemplateRequest, ApiError> {
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
    let (request, _) = finalize_request(validated, reasoning_profile)?;
    Ok(request.template)
}

fn props_response(properties: ModelProperties) -> PropsResponse {
    let reasoning = ReasoningProfileResponse {
        default_reasoning_effort: properties.reasoning.default_effort.0.clone(),
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

struct ValidatedChatRequest {
    model: Option<String>,
    messages: Vec<ChatMessage>,
    tools: Vec<ToolDefinition>,
    tool_choice: ToolChoice,
    parallel_tool_calls: bool,
    reasoning_effort: Option<ReasoningEffortRequest>,
    thinking_budget_tokens: Option<u32>,
    response_format: ResponseFormat,
    template_args: BTreeMap<String, JsonValue>,
    stop: Vec<String>,
    max_tokens: u32,
    temperature: f32,
    top_p: f32,
    seed: u32,
    cache_prompt: bool,
    ignore_eos: bool,
    timings_per_token: bool,
    include_usage: bool,
}

fn validate_request(request: ChatCompletionRequest) -> Result<ValidatedChatRequest, ApiError> {
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
        messages,
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
    })
}

fn finalize_request(
    mut validated: ValidatedChatRequest,
    reasoning_profile: &icn_contracts::ReasoningProfile,
) -> Result<(ChatRequest, bool), ApiError> {
    let reasoning = reasoning_control(
        validated.reasoning_effort,
        validated.thinking_budget_tokens,
        &mut validated.template_args,
        reasoning_profile,
    )?;
    Ok((
        ChatRequest {
            template: ChatTemplateRequest {
                messages: validated.messages,
                tools: validated.tools,
                tool_choice: validated.tool_choice,
                parallel_tool_calls: validated.parallel_tool_calls,
                reasoning,
                response_format: validated.response_format,
                template_args: validated.template_args,
            },
            stop: validated.stop,
            max_tokens: validated.max_tokens,
            temperature: validated.temperature,
            top_p: validated.top_p,
            seed: validated.seed,
            cache_prompt: validated.cache_prompt,
            ignore_eos: validated.ignore_eos,
            timings_per_token: validated.timings_per_token,
        },
        validated.include_usage,
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
    profile: &icn_contracts::ReasoningProfile,
) -> Result<ReasoningControl, ApiError> {
    const OWNED_KEYS: &[&str] = &[
        "enable_thinking",
        "thinking",
        "thinking_mode",
        "reasoning_effort",
        "thinking_budget",
        "preserve_thinking",
        "clear_thinking",
        "drop_thinking",
    ];
    let raw_reasoning_controls = template_args
        .keys()
        .any(|key| OWNED_KEYS.contains(&key.as_str()));

    let selected = match effort {
        Some(effort) => {
            if raw_reasoning_controls {
                return Err(ApiError::invalid(
                    "reasoning_effort conflicts with reasoning controls in chat_template_kwargs",
                ));
            }
            let normalized = effort.normalize()?;
            profile.mapping(&normalized).ok_or_else(|| {
                let supported = profile
                    .mappings
                    .iter()
                    .map(|mapping| mapping.effort.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                ApiError::invalid(format!(
                    "reasoning_effort {} is unsupported for this model; supported values: {supported}",
                    normalized.as_str()
                ))
            })?
        }
        None if raw_reasoning_controls => {
            if budget_tokens.is_none() {
                return Ok(ReasoningControl::ModelDefault);
            }
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
            if explicitly_disabled {
                return Err(ApiError::invalid(
                    "thinking_budget_tokens cannot be used when raw template controls disable reasoning",
                ));
            }
            return Ok(ReasoningControl::Resolved {
                effort: profile.default_effort.clone(),
                controls: icn_contracts::NativeReasoningControls::default(),
                automatic_budget: icn_contracts::AutomaticReasoningBudget::Disabled,
                explicit_budget_tokens: budget_tokens,
                template_fingerprint: profile.template_fingerprint.clone(),
            });
        }
        None => profile
            .mapping(&profile.default_effort)
            .expect("reasoning profile contains its default mapping"),
    };

    if budget_tokens.is_some() && selected.effort.as_str() == "none" {
        return Err(ApiError::invalid(
            "thinking_budget_tokens cannot be used when reasoning is disabled (reasoning_effort none)",
        ));
    }
    Ok(ReasoningControl::Resolved {
        effort: selected.effort.clone(),
        controls: selected.controls.clone(),
        automatic_budget: selected.automatic_budget.clone(),
        explicit_budget_tokens: budget_tokens,
        template_fingerprint: profile.template_fingerprint.clone(),
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
    paths(
        health,
        hardware,
        search_hugging_face_models,
        resolve_hugging_face_repository,
        installed_models,
        remove_installed_model,
        recommendable_model_catalog,
        assess_models,
        fit_models,
        start_model_download,
        model_downloads,
        model_download_attempt,
        cancel_model_download,
        load_model_configuration,
        unload_model_residency,
        props,
        apply_template,
        chat_completions
    ),
    components(schemas(
        HealthResponse,
        HardwareSnapshot,
        HuggingFaceModelSearchRequest,
        HuggingFaceModelSearchResults,
        HuggingFaceRepositoryRequest,
        HuggingFaceRepositorySnapshot,
        InstalledModelPackagesResponse,
        RemoveInstalledModelPackageResponse,
        RecommendableModelCatalog,
        AssessModelsRequest,
        AssessModelsResponse,
        FitModelsRequest,
        FitModelsResponse,
        StartModelDownloadRequest,
        StartModelDownloadResponse,
        ModelDownloadsResponse,
        DownloadAttempt,
        LoadModelRequest,
        ModelLoadEvent,
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

struct ModelLoadStream;

impl StreamContract for ModelLoadStream {
    type Event = ModelLoadEvent;
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
    attach_stream_contract::<ModelLoadStream>(
        &mut document,
        "loadModelConfiguration",
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
            reasoning: icn_contracts::ReasoningProfile {
                default_effort: icn_contracts::NormalizedReasoningEffort("high".into()),
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
            mtp: icn_contracts::MtpRuntimeProperties::Disabled {
                reason: "fake_backend".into(),
            },
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
        on_event: &mut dyn FnMut(InferenceStreamEvent) -> Result<(), InferenceError>,
    ) -> Result<Generation, InferenceError> {
        let prompt_tokens = request.template.messages.len();
        on_event(InferenceStreamEvent {
            delta: InferenceEvent::StreamStart,
            timings: None,
        })?;
        for (index, token) in self.response.split_inclusive(' ').enumerate() {
            on_event(InferenceStreamEvent {
                delta: InferenceEvent::ContentDelta {
                    text: token.to_owned(),
                },
                timings: request.timings_per_token.then(|| GenerationSnapshot {
                    cached_prompt_tokens: 0,
                    prompt_tokens,
                    generated_tokens: index + 1,
                    metrics: GenerationMetrics::default(),
                }),
            })?;
        }
        Ok(Generation {
            text: self.response.clone(),
            reasoning: String::new(),
            tool_calls: Vec::new(),
            cached_prompt_tokens: 0,
            prompt_tokens,
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

    struct StubHardware;

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
                        "total_bytes": 1024,
                        "current_available_bytes": 512
                    },
                    "native_build": "test-build",
                    "enabled_backends": ["cpu"],
                    "topology_fingerprint": "topology",
                    "resident_memory": null,
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
            .oneshot(Request::get("/v1/hardware").body(Body::empty()).unwrap())
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
                Request::post("/v1/hugging-face/models/search")
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
                Request::post("/v1/hugging-face/models/resolve")
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
    ) -> Result<(ChatRequest, bool), ApiError> {
        let profile = FakeBackend::new("test-model", "")
            .properties()
            .expect("fake properties")
            .reasoning;
        validate_request(request).and_then(|validated| finalize_request(validated, &profile))
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

    struct StubRuntime {
        backends: BackendRegistry,
        leases: Arc<AtomicU64>,
    }

    impl RuntimeController for StubRuntime {
        fn load_configuration(
            &self,
            _request: LoadModelRequest,
        ) -> BoxStream<'static, ModelLoadEvent> {
            Box::pin(futures_util::stream::once(async {
                ModelLoadEvent::Failed {
                    failure: icn_contracts::models::ModelFailure {
                        code: "unsupported".to_owned(),
                        message: "exact configuration loading is unavailable in the stub runtime"
                            .to_owned(),
                        retryable: false,
                    },
                }
            }))
        }

        fn unload_residency(
            &self,
            _residency_id: String,
        ) -> BoxFuture<'_, Result<(), InventoryError>> {
            Box::pin(async { Ok(()) })
        }

        fn remove_installed(
            &self,
            package_id: ModelPackageId,
        ) -> BoxFuture<'_, Result<RemoveInstalledModelPackageResponse, InventoryError>> {
            Box::pin(async move {
                Err(InventoryError::Unsupported(format!(
                    "cannot remove package {} in the stub runtime",
                    package_id.0
                )))
            })
        }

        fn lease(&self, _model_id: String) -> BoxFuture<'_, Result<BackendLease, InventoryError>> {
            Box::pin(async {
                self.leases.fetch_add(1, Ordering::Relaxed);
                self.backends
                    .lease()
                    .ok_or_else(|| InventoryError::Internal("test backend unavailable".into()))
            })
        }
    }

    #[tokio::test]
    async fn ordinary_chat_leases_the_requested_resident_model_before_streaming() {
        let backends =
            BackendRegistry::with_backend(Arc::new(FakeBackend::new("test-model", "ready")));
        let leases = Arc::new(AtomicU64::new(0));
        let runtime = Arc::new(StubRuntime {
            backends: backends.clone(),
            leases: Arc::clone(&leases),
        });
        let response = app(AppState::model_free(backends).with_runtime(runtime))
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
    async fn invalid_chat_is_rejected_before_model_lease() {
        let backends =
            BackendRegistry::with_backend(Arc::new(FakeBackend::new("test-model", "ready")));
        let leases = Arc::new(AtomicU64::new(0));
        let runtime = Arc::new(StubRuntime {
            backends: backends.clone(),
            leases: Arc::clone(&leases),
        });
        let response = app(AppState::model_free(backends).with_runtime(runtime))
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

    struct ScriptedBackend {
        events: Vec<InferenceStreamEvent>,
        fail: bool,
    }

    fn stream_event(delta: InferenceEvent) -> InferenceStreamEvent {
        InferenceStreamEvent {
            delta,
            timings: None,
        }
    }

    fn timed_stream_event(
        delta: InferenceEvent,
        generated_tokens: usize,
        decode_ms: f64,
    ) -> InferenceStreamEvent {
        InferenceStreamEvent {
            delta,
            timings: Some(GenerationSnapshot {
                cached_prompt_tokens: 0,
                prompt_tokens: 11,
                generated_tokens,
                metrics: GenerationMetrics {
                    prompt_ms: 2.0,
                    decode_ms,
                    ..GenerationMetrics::default()
                },
            }),
        }
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
            _request: ChatRequest,
            on_event: &mut dyn FnMut(InferenceStreamEvent) -> Result<(), InferenceError>,
        ) -> Result<Generation, InferenceError> {
            if !matches!(
                self.events.first().map(|event| &event.delta),
                Some(InferenceEvent::StreamStart)
            ) {
                on_event(InferenceStreamEvent {
                    delta: InferenceEvent::StreamStart,
                    timings: None,
                })?;
            }
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
                cached_prompt_tokens: 0,
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
                    draft_tokens: 0,
                    accepted_draft_tokens: 0,
                    draft_ms: 0.0,
                    verification_ms: 0.0,
                    ..GenerationMetrics::default()
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
            .oneshot(Request::get("/v1/props").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(denied.status(), StatusCode::UNAUTHORIZED);

        let allowed = service
            .oneshot(
                Request::get("/v1/props")
                    .header("authorization", "Bearer private-capability")
                    .body(Body::empty())
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
    fn accepts_the_loaded_model_id_and_configured_aliases() {
        let state =
            AppState::new(FakeBackend::new("test-model", "ok")).with_model_alias("friendly-name");
        let lease = state.backends.lease().expect("backend lease");

        assert!(validate_model_selection(Some("test-model"), &lease).is_ok());
        assert!(validate_model_selection(Some("friendly-name"), &lease).is_ok());
        assert!(validate_model_selection(None, &lease).is_ok());
        assert!(validate_model_selection(Some("different-model"), &lease).is_err());
    }

    #[test]
    fn backend_mutations_exclude_inference_leases_in_both_directions() {
        let registry =
            BackendRegistry::with_backend(Arc::new(FakeBackend::new("test-model", "ok")));
        let lease = registry.lease().expect("initial lease");
        assert!(registry.try_begin_mutation().is_none());
        drop(lease);

        let mutation = registry.try_begin_mutation().expect("mutation guard");
        assert!(registry.lease().is_none());
        drop(mutation);
        assert!(registry.lease().is_some());
    }

    #[tokio::test]
    async fn coordinated_mutation_waits_for_the_response_lease() {
        let registry =
            BackendRegistry::with_backend(Arc::new(FakeBackend::new("test-model", "ok")));
        let lease = registry.lease().expect("initial lease");
        let waiting = registry.clone();
        let mutation = tokio::spawn(async move { waiting.begin_mutation().await });
        tokio::task::yield_now().await;
        assert!(!mutation.is_finished());
        assert!(
            registry.lease().is_none(),
            "queued mutation must close new lease admission"
        );

        drop(lease);
        let guard = tokio::time::timeout(std::time::Duration::from_secs(1), mutation)
            .await
            .expect("mutation should wake after lease release")
            .expect("mutation task should complete");
        assert!(registry.lease().is_none());
        drop(guard);
        assert!(registry.lease().is_some());
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
        assert!(matches!(
            &request.template.reasoning,
            ReasoningControl::Resolved {
                effort,
                controls,
                automatic_budget: icn_contracts::AutomaticReasoningBudget::Disabled,
                explicit_budget_tokens: Some(64),
                ..
            } if effort.as_str() == "high" && controls.enable_thinking == Some(true)
        ));
        assert!(
            !request
                .template
                .template_args
                .contains_key("reasoning_effort")
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
        assert!(!request.cache_prompt);
        assert!(request.ignore_eos);
        assert!(request.timings_per_token);
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
    fn preserves_model_defaults_when_optional_controls_are_omitted() {
        let (request, include_usage) =
            validate_test_request(request_from_json(minimal_request())).unwrap();
        assert!(!include_usage);
        assert_eq!(request.template.tool_choice, ToolChoice::Auto);
        assert!(request.template.parallel_tool_calls);
        assert!(matches!(
            request.template.reasoning,
            ReasoningControl::Resolved {
                ref effort,
                automatic_budget: icn_contracts::AutomaticReasoningBudget::Disabled,
                explicit_budget_tokens: None,
                ..
            } if effort.as_str() == "high"
        ));
        assert_eq!(request.template.response_format, ResponseFormat::Text);
        assert!(request.template.template_args.is_empty());
        assert!(request.stop.is_empty());
        assert!(request.cache_prompt);
        assert!(!request.ignore_eos);
        assert!(!request.timings_per_token);
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
            let (request, _) = validate_test_request(request_from_json(request)).unwrap();
            assert!(!request.timings_per_token);
        }

        let mut request = minimal_request();
        request["timings_per_token"] = json!(true);
        let (request, _) = validate_test_request(request_from_json(request)).unwrap();
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
        let error = validate_test_request(request_from_json(request)).unwrap_err();
        assert!(error.body.error.message.contains("reasoning is disabled"));

        let mut request = minimal_request();
        request["reasoning_effort"] = json!("high");
        request["chat_template_kwargs"] = json!({"enable_thinking": true});
        let error = validate_test_request(request_from_json(request)).unwrap_err();
        assert!(error.body.error.message.contains("conflicts"));

        let mut request = minimal_request();
        request["reasoning_effort"] = json!("medium");
        let error = validate_test_request(request_from_json(request)).unwrap_err();
        assert!(
            error
                .body
                .error
                .message
                .contains("supported values: none, high")
        );

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
        assert!(matches!(
            request.template.reasoning,
            ReasoningControl::Resolved {
                ref effort,
                ref controls,
                automatic_budget: icn_contracts::AutomaticReasoningBudget::Disabled,
                ..
            } if effort.as_str() == "none" && controls.enable_thinking == Some(false)
        ));
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
    async fn streams_cumulative_native_timings_on_group_terminal_deltas() {
        let backend = ScriptedBackend {
            events: vec![
                stream_event(InferenceEvent::ReasoningDelta {
                    text: "buffered group prefix".into(),
                }),
                timed_stream_event(
                    InferenceEvent::ContentDelta {
                        text: "first group end".into(),
                    },
                    1,
                    0.001,
                ),
                timed_stream_event(
                    InferenceEvent::ContentDelta {
                        text: "second group".into(),
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
        assert_eq!(terminal["choices"][0]["finish_reason"], "tool_calls");
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
            ])
        );
        assert_eq!(terminal["timings"]["cache_n"], 0);
        assert_eq!(terminal["timings"]["prompt_n"], 11);
        assert_eq!(terminal["timings"]["prompt_per_second"], 5_500.0);
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
            events: vec![timed_stream_event(InferenceEvent::StreamStart, 1, 0.001)],
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
        assert_eq!(chunks[1]["choices"][0]["finish_reason"], "tool_calls");
        assert!(chunks[1].get("timings").is_none());
        assert_eq!(chunks[2]["choices"], json!([]));
        assert_eq!(chunks[2]["timings"]["predicted_n"], 7);
    }

    #[tokio::test]
    async fn backend_signaled_stop_word_partial_timing_is_kept_when_flag_is_false() {
        let backend = ScriptedBackend {
            events: vec![timed_stream_event(InferenceEvent::StreamStart, 1, 0.001)],
            fail: false,
        };
        let mut request = minimal_request();
        request["timings_per_token"] = json!(false);

        let (status, body) = post_chat(backend, request).await;
        assert_eq!(status, StatusCode::OK);
        let chunks = stream_json(&body);

        assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(chunks[0]["timings"]["predicted_n"], 1);
        assert_eq!(chunks[1]["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(chunks[1]["timings"]["predicted_n"], 7);
    }

    #[tokio::test]
    async fn false_timing_control_suppresses_partial_snapshots_but_not_final_timings() {
        let backend = ScriptedBackend {
            events: vec![stream_event(InferenceEvent::ContentDelta {
                text: "answer".into(),
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
                stream_event(InferenceEvent::ReasoningDelta {
                    text: "thought".into(),
                }),
                stream_event(InferenceEvent::ContentDelta {
                    text: "answer".into(),
                }),
                stream_event(InferenceEvent::ToolCallDelta {
                    index: 0,
                    id: Some("call-1".into()),
                    name: Some("lookup".into()),
                    arguments: "{".into(),
                }),
                stream_event(InferenceEvent::ToolCallDelta {
                    index: 0,
                    id: None,
                    name: None,
                    arguments: "}".into(),
                }),
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
    async fn backend_failure_is_an_explicit_stream_error_followed_by_done() {
        let backend = ScriptedBackend {
            events: vec![stream_event(InferenceEvent::ContentDelta {
                text: "partial".into(),
            })],
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
