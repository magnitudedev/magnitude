//! Backend-neutral model inventory contracts.

use std::collections::BTreeMap;
use std::path::PathBuf;

use futures_util::future::BoxFuture;
use futures_util::stream::BoxStream;
use serde::{Deserialize, Serialize};

/// Stable identity of one physical memory pool.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct MemoryDomainId(String);

impl MemoryDomainId {
    const SYSTEM: &'static str = "system";

    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    #[must_use]
    pub fn system() -> Self {
        Self(Self::SYSTEM.to_owned())
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    #[must_use]
    pub fn is_system(&self) -> bool {
        self.0 == Self::SYSTEM
    }
}

impl std::fmt::Display for MemoryDomainId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

/// Stable cross-process identity of one execution device: the serialized Seismic device
/// selector (for example `metal:<registry-id>`, `cuda:<uuid>`, `vulkan:<uuid>`, `host-cpu`).
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct HardwareDeviceId(String);

impl HardwareDeviceId {
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for HardwareDeviceId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

/// Source-scoped identity of one runnable model at one local location.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct InventoryEntryId(pub String);

/// Content-derived identity shared by equivalent models at different locations.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ContentId(pub String);

impl InventoryEntryId {
    pub fn parse(value: impl Into<String>) -> Result<Self, InventoryError> {
        let value = value.into();
        validate_prefixed_digest(&value, "mdl_")?;
        Ok(Self(value))
    }
}

impl ContentId {
    pub fn parse(value: impl Into<String>) -> Result<Self, InventoryError> {
        let value = value.into();
        validate_prefixed_digest(&value, "content_")?;
        Ok(Self(value))
    }
}

fn validate_prefixed_digest(value: &str, prefix: &str) -> Result<(), InventoryError> {
    let Some(digest) = value.strip_prefix(prefix) else {
        return Err(InventoryError::InvalidId(value.to_owned()));
    };
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(InventoryError::InvalidId(value.to_owned()));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComponentRole {
    Weights,
    Shard,
    Projector,
    Auxiliary,
    Draft,
    Mtp,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ContentIdentity {
    Sha256 { value: String },
    GitOid { value: String },
    Xet { value: String },
    FileIdentity { value: String },
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelComponent {
    pub path: PathBuf,
    pub role: ComponentRole,
    pub size_bytes: u64,
    pub content: ContentIdentity,
    pub shard_index: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relationship: Option<ComponentRelationship>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ComponentRelationship {
    ProjectorFor {
        projector: PathBuf,
        model: PathBuf,
    },
    DraftFor {
        draft: PathBuf,
        model: PathBuf,
        method: crate::models::SpeculativeMethod,
    },
    MtpFor {
        mtp: PathBuf,
        model: PathBuf,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Integrity {
    Verified { method: String },
    Unverified { reason: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ModelSource {
    HuggingFace {
        repository: String,
        requested_revision: String,
        commit: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metadata: Option<Box<HubMetadata>>,
    },
    Local {
        declared_by: LocalDeclaration,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalDeclaration {
    Configuration,
    Discovery,
    ActiveProcess,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct HubMetadata {
    pub access: Option<String>,
    pub author: Option<String>,
    pub license: Option<String>,
    pub pipeline_tag: Option<String>,
    pub library_name: Option<String>,
    pub tags: Vec<String>,
    pub downloads: Option<u64>,
    pub likes: Option<u64>,
    pub last_modified: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ModelLocation {
    MagnitudeCache {
        components: Vec<ModelComponent>,
        total_bytes: u64,
        integrity: Integrity,
    },
    HuggingFaceCache {
        cache_root: PathBuf,
        repository: String,
        commit: String,
        components: Vec<ModelComponent>,
        total_bytes: u64,
        integrity: Integrity,
    },
    Directory {
        source_id: String,
        root: PathBuf,
        components: Vec<ModelComponent>,
        total_bytes: u64,
        integrity: Integrity,
    },
    File {
        path: PathBuf,
        component: ModelComponent,
        integrity: Integrity,
    },
}

impl ModelLocation {
    pub fn components(&self) -> &[ModelComponent] {
        match self {
            Self::MagnitudeCache { components, .. }
            | Self::HuggingFaceCache { components, .. }
            | Self::Directory { components, .. } => components,
            Self::File { component, .. } => std::slice::from_ref(component),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ModelAvailability {
    Downloading {
        operation_id: String,
        stage: DownloadStage,
        completed_bytes: u64,
        total_bytes: u64,
        current_component: Option<PathBuf>,
        started_at: u64,
        updated_at: u64,
    },
    Interrupted {
        completed_bytes: u64,
        total_bytes: u64,
        resumable: bool,
        failure: DownloadFailure,
        updated_at: u64,
    },
    Available {
        ready_at: u64,
    },
    InvalidArtifact {
        detected_at: u64,
        code: String,
        message: String,
    },
    IncompatibleArtifact {
        detected_at: u64,
        code: String,
        message: String,
    },
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DownloadStage {
    Queued,
    Resolving,
    CheckingSpace,
    Downloading,
    Verifying,
    Publishing,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum CapabilitySupport {
    Supported { parallel: Option<bool> },
    Unsupported,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ReasoningCapability {
    Unsupported {
        evidence: CapabilityEvidence,
    },
    Supported {
        control: ReasoningControlDomain,
        visibility: ReasoningVisibility,
        delimiters: ReasoningDelimiters,
        evidence: CapabilityEvidence,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ReasoningControlDomain {
    Toggle {
        default: bool,
    },
    Effort {
        levels: Vec<String>,
        default: Option<String>,
    },
    Budget {
        min_tokens: u32,
        max_tokens: u32,
        default_tokens: u32,
    },
    EffortAndBudget {
        levels: Vec<String>,
        default_effort: Option<String>,
        min_tokens: u32,
        max_tokens: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningVisibility {
    Hidden,
    Preserved,
    Configurable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ReasoningDelimiters {
    Unavailable,
    Known { start: String, end: String },
}

/// Product-normalized reasoning selection. Native template spellings are kept in the compiled
/// mapping rather than exposed through this identifier.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct NormalizedReasoningEffort(pub String);

impl NormalizedReasoningEffort {
    pub fn parse(value: &str) -> Option<Self> {
        let normalized = match value {
            "off" | "no_think" | "disabled" => "none",
            "extra_high" | "extra-high" | "very_high" => "xhigh",
            "none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max" | "adaptive" => value,
            _ => return None,
        };
        Some(Self(normalized.to_owned()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AutomaticReasoningBudget {
    Disabled,
    FixedTokens { tokens: u32 },
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct NativeReasoningControls {
    /// `None` omits the native backend's dedicated control and preserves the authored template default.
    pub enable_thinking: Option<bool>,
    pub template_args: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReasoningEffortMapping {
    pub effort: NormalizedReasoningEffort,
    pub controls: NativeReasoningControls,
    pub automatic_budget: AutomaticReasoningBudget,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReasoningProfile {
    pub default_effort: Option<NormalizedReasoningEffort>,
    pub mappings: Vec<ReasoningEffortMapping>,
    pub template_fingerprint: String,
}

impl ReasoningProfile {
    pub fn mapping(&self, effort: &NormalizedReasoningEffort) -> Option<&ReasoningEffortMapping> {
        self.mappings
            .iter()
            .find(|mapping| &mapping.effort == effort)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum CapabilityEvidence {
    NativeTemplate { fingerprint: String },
    BoundedTemplateProbe { fingerprint: String },
    DeclaredMetadata { source: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
// This is a wire contract: introducing a nested payload solely to equalize in-memory variant
// sizes would make the serialized shape less direct for every API consumer.
#[allow(clippy::large_enum_variant)]
pub enum InventoryProperties {
    Pending,
    Unavailable {
        reason: String,
    },
    Inspected {
        architecture: Option<String>,
        quantization: Option<String>,
        quantization_name: Option<String>,
        parameter_count: Option<u64>,
        active_parameter_count: Option<u64>,
        training_context_length: Option<u32>,
        nextn_predict_layers: Option<u32>,
        tokenizer: Option<String>,
        modalities: Vec<String>,
        base_models: Vec<String>,
        evidence_fingerprint: String,
    },
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HardwareMemoryDomainKind {
    System,
    PhysicalDevice,
    UnifiedMemory,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HardwareDeviceKind {
    Cpu,
    Gpu,
    IntegratedGpu,
    Accelerator,
    Unknown,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HardwareDeviceMemoryLimitKind {
    RecommendedWorkingSet,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareDeviceMemoryLimit {
    pub kind: HardwareDeviceMemoryLimitKind,
    pub total_bytes: u64,
    pub stable_bytes: u64,
    pub current_free_bytes: Option<u64>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareDevice {
    pub id: HardwareDeviceId,
    pub backend: ExecutionBackend,
    pub name: String,
    pub description: String,
    pub kind: HardwareDeviceKind,
    #[serde(default)]
    pub memory_limit: Option<HardwareDeviceMemoryLimit>,
    /// Why discovery found this device but it is not an execution candidate, for example a driver
    /// below its backend's floor. Absent means the device is usable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(nullable = false))]
    pub unavailable_reason: Option<String>,
}

/// An execution backend compiled into the inference engine.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionBackend {
    Cpu,
    Metal,
    Cuda,
    Vulkan,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareSystemMemory {
    pub physical_capacity_bytes: u64,
    pub physical_available_bytes: u64,
    pub allocation_capacity_bytes: u64,
    pub allocation_headroom_bytes: u64,
    pub assess_reserve_bytes: u64,
    pub abort_reserve_bytes: u64,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareMemoryDomain {
    pub id: MemoryDomainId,
    pub kind: HardwareMemoryDomainKind,
    pub total_capacity_bytes: u64,
    pub stable_capacity_bytes: u64,
    pub current_free_bytes: Option<u64>,
    pub shares_system_memory: bool,
    pub devices: Vec<HardwareDevice>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareSnapshot {
    pub captured_at: u64,
    pub platform: String,
    pub architecture: String,
    #[serde(default)]
    pub system_product_name: Option<String>,
    pub cpu_model: Option<String>,
    /// OS-reported physical CPU cores, independent of process scheduling limits.
    pub physical_cores: Option<usize>,
    pub logical_cores: usize,
    pub system_memory: HardwareSystemMemory,
    /// Engine build identity: engine version plus kernel bundle identity.
    pub native_build: String,
    /// Backends compiled into the engine build.
    pub enabled_backends: Vec<ExecutionBackend>,
    pub topology_fingerprint: String,
    pub memory_domains: Vec<HardwareMemoryDomain>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelPreviewSource {
    pub repository: String,
    pub revision: String,
    pub primary_gguf: PathBuf,
    pub additional_components: Vec<ModelPreviewComponentSource>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelPreviewComponentSource {
    pub path: PathBuf,
    pub role: ModelPreviewComponentRole,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ModelPreviewComponentRole {
    Projector,
    Draft {
        method: crate::models::SpeculativeMethod,
    },
    Mtp,
}

impl ModelPreviewComponentRole {
    #[must_use]
    pub fn component_role(&self) -> ComponentRole {
        match self {
            Self::Projector => ComponentRole::Projector,
            Self::Draft { .. } => ComponentRole::Draft,
            Self::Mtp => ComponentRole::Mtp,
        }
    }
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HuggingFaceModelSearchRequest {
    pub query: String,
    pub limit: u32,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HuggingFaceModelSearchResult {
    pub repository: String,
    pub commit: String,
    pub last_modified: Option<String>,
    pub downloads: Option<u64>,
    pub likes: Option<u64>,
    pub gated: bool,
    pub private: bool,
    pub tags: Vec<String>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HuggingFaceModelSearchResults {
    pub models: Vec<HuggingFaceModelSearchResult>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HuggingFaceRepositoryRequest {
    pub repository: String,
    pub revision: String,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HuggingFaceRepositoryFile {
    #[cfg_attr(feature = "openapi", schema(value_type = String))]
    pub path: PathBuf,
    pub size_bytes: u64,
    pub content: ContentIdentity,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HuggingFaceRepositorySnapshot {
    pub repository: String,
    pub commit: String,
    pub last_modified: Option<String>,
    pub downloads: Option<u64>,
    pub likes: Option<u64>,
    pub gated: bool,
    pub private: bool,
    pub license: Option<String>,
    pub license_url: Option<String>,
    pub base_models: Vec<String>,
    pub tags: Vec<String>,
    pub gguf_files: Vec<HuggingFaceRepositoryFile>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InventoryModel {
    pub id: InventoryEntryId,
    pub content_id: ContentId,
    pub created: u64,
    pub name: String,
    pub supported_parameters: Vec<String>,
    pub availability: ModelAvailability,
    pub source: ModelSource,
    pub location: ModelLocation,
    pub properties: InventoryProperties,
    pub operations: Vec<ModelOperation>,
    pub updated_at: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedModel {
    pub model: InventoryModel,
    pub components: Vec<ResolvedComponent>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedComponent {
    pub path: PathBuf,
    pub role: ComponentRole,
    pub shard_index: Option<u32>,
    pub relationship: Option<ComponentRelationship>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelOperation {
    Load,
    Unload,
    Delete,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ModelDownloadEvent {
    Resolving {
        operation_id: String,
        repository: String,
        revision: String,
    },
    CheckingSpace {
        operation_id: String,
        model_id: InventoryEntryId,
        required_bytes: u64,
        available_bytes: u64,
        completed_bytes: u64,
        total_bytes: u64,
    },
    Progress {
        operation_id: String,
        model_id: InventoryEntryId,
        stage: DownloadStage,
        completed_bytes: u64,
        total_bytes: u64,
        file: DownloadFileProgress,
        bytes_per_second: Option<f64>,
        resumed_from_bytes: u64,
    },
    Ready {
        operation_id: String,
        model: Box<InventoryModel>,
    },
    Cancelled {
        operation_id: String,
        model_id: Option<InventoryEntryId>,
        completed_bytes: u64,
        total_bytes: u64,
    },
    Failed {
        operation_id: String,
        model_id: Option<InventoryEntryId>,
        error: DownloadFailure,
        completed_bytes: u64,
        total_bytes: u64,
        resumable: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DownloadFileProgress {
    pub path: PathBuf,
    pub completed_bytes: u64,
    pub total_bytes: u64,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase")]
pub enum DownloadFailure {
    Interrupted,
    #[serde(rename_all = "camelCase")]
    InsufficientDiskSpace {
        required_bytes: u64,
        available_bytes: u64,
    },
    #[serde(rename_all = "camelCase")]
    SourceUnavailable,
    NetworkUnavailable,
    LocalStorageFailure,
    CorruptDownload,
    #[serde(rename_all = "camelCase")]
    Internal {
        message: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeletePlan {
    pub model_id: InventoryEntryId,
    pub supported: bool,
    pub reason: Option<String>,
    pub reclaimable_bytes: u64,
    pub retained_shared_bytes: u64,
    pub paths: Vec<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeletedModel {
    pub id: InventoryEntryId,
    pub deleted: bool,
    pub freed_bytes: u64,
    pub retained_shared_bytes: u64,
    pub plan: DeletePlan,
}

pub type DownloadEventStream = BoxStream<'static, ModelDownloadEvent>;

/// Model inventory boundary consumed by the HTTP API and server composition root.
pub trait ModelInventory: Send + Sync + 'static {
    fn list(&self) -> BoxFuture<'_, Result<Vec<InventoryModel>, InventoryError>>;
    fn get(&self, id: &InventoryEntryId) -> BoxFuture<'_, Result<InventoryModel, InventoryError>>;
    fn plan_delete(
        &self,
        id: &InventoryEntryId,
    ) -> BoxFuture<'_, Result<DeletePlan, InventoryError>>;
    fn delete(&self, id: &InventoryEntryId) -> BoxFuture<'_, Result<DeletedModel, InventoryError>>;
    fn resolve_ready(
        &self,
        id: &InventoryEntryId,
    ) -> BoxFuture<'_, Result<ResolvedModel, InventoryError>>;
}

pub trait HardwareProvider: Send + Sync + 'static {
    fn snapshot(&self) -> BoxFuture<'_, Result<HardwareSnapshot, InventoryError>>;
}

/// Live Hugging Face discovery. Resolved commits are immutable snapshots for a
/// subsequent preview or download, not catalog pins.
pub trait HuggingFaceModelCatalog: Send + Sync + 'static {
    fn search(
        &self,
        request: HuggingFaceModelSearchRequest,
    ) -> BoxFuture<'_, Result<HuggingFaceModelSearchResults, InventoryError>>;

    fn resolve(
        &self,
        request: HuggingFaceRepositoryRequest,
    ) -> BoxFuture<'_, Result<HuggingFaceRepositorySnapshot, InventoryError>>;
}

#[derive(Debug, thiserror::Error)]
pub enum InventoryError {
    #[error("invalid model id: {0}")]
    InvalidId(String),
    #[error("invalid model request: {0}")]
    InvalidRequest(String),
    #[error("model not found: {0}")]
    NotFound(String),
    #[error("model is not ready: {0}")]
    NotReady(String),
    #[error("model is busy: {0}")]
    Busy(String),
    #[error("model is loaded: {0}")]
    Loaded(String),
    #[error("deletion is unsafe: {0}")]
    DeletionUnsafe(String),
    #[error("model source does not support this operation: {0}")]
    Unsupported(String),
    #[error("inventory I/O failed: {0}")]
    Io(String),
    #[error("upstream model service failed: {0}")]
    Upstream(String),
    #[error("model integrity check failed: {0}")]
    Integrity(String),
    #[error("model artifacts changed during inspection: {0}")]
    ConcurrentMutation(String),
    #[error("{message}")]
    ModelOperation {
        code: String,
        message: String,
        retryable: bool,
    },
    #[error("internal inventory failure: {0}")]
    Internal(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_require_the_exact_versioned_prefix_and_lowercase_digest() {
        let digest = "a".repeat(64);
        assert!(InventoryEntryId::parse(format!("mdl_{digest}")).is_ok());
        assert!(ContentId::parse(format!("content_{digest}")).is_ok());
        assert!(InventoryEntryId::parse(format!("content_{digest}")).is_err());
        assert!(InventoryEntryId::parse(format!("mdl_{}", "A".repeat(64))).is_err());
        assert!(InventoryEntryId::parse("mdl_short").is_err());
    }
}
