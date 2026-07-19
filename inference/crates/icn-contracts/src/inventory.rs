//! Backend-neutral model inventory contracts.

use std::collections::BTreeMap;
use std::path::PathBuf;

use futures_util::future::BoxFuture;
use futures_util::stream::BoxStream;
use serde::{Deserialize, Serialize};

/// Source-scoped identity of one runnable model at one local location.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ModelId(pub String);

/// Content-derived identity shared by equivalent models at different locations.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ContentId(pub String);

impl ModelId {
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
    pub relationship: Option<ComponentRelationship>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ComponentRelationship {
    ProjectorFor { projector: PathBuf, model: PathBuf },
    DraftFor { draft: PathBuf, model: PathBuf },
    MtpFor { mtp: PathBuf, model: PathBuf },
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
pub enum ModelStatus {
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
        reason: Option<String>,
        last_error: String,
        updated_at: u64,
    },
    Available {
        ready_at: u64,
    },
    Loading {
        load_id: String,
        stage: LoadStage,
        started_at: u64,
    },
    Loaded {
        loaded_at: u64,
        backend: String,
        context_length: u32,
        execution: BTreeMap<String, serde_json::Value>,
    },
    Unloading {
        load_id: String,
        started_at: u64,
    },
    LoadFailed {
        attempted_at: u64,
        stage: LoadStage,
        code: String,
        retryable: bool,
    },
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoadStage {
    Opening,
    Mapping,
    Allocating,
    InitializingContext,
    Warming,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum CapabilitySupport {
    Supported { parallel: Option<bool> },
    Unsupported,
    Unknown { reason: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ReasoningCapability {
    Unknown {
        reason: String,
    },
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum CapabilityEvidence {
    NativeTemplate { fingerprint: String },
    BoundedTemplateProbe { fingerprint: String },
    DeclaredMetadata { source: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemplateAssessment {
    pub capabilities: crate::TemplateCapabilities,
    pub reasoning: ReasoningCapability,
    pub fingerprint: String,
}

/// Model-free native chat-template assessment injected into model discovery.
pub trait TemplateAssessor: Send + Sync + 'static {
    fn assess(
        &self,
        template: &str,
        bos_token: Option<&str>,
        eos_token: Option<&str>,
    ) -> Result<TemplateAssessment, String>;
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
        parameter_count: Option<u64>,
        active_parameter_count: Option<u64>,
        training_context_length: Option<u32>,
        tokenizer: Option<String>,
        modalities: Vec<String>,
        base_models: Vec<String>,
        tools: CapabilitySupport,
        structured_output: CapabilitySupport,
        reasoning: ReasoningCapability,
        evidence_fingerprint: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum HardwareAssessment {
    NotAssessed {
        reason: String,
    },
    Assessing {
        started_at: u64,
    },
    Fits {
        profile: HardwareProfile,
        memory: HardwareMemory,
        recommendation: HardwareRecommendation,
    },
    DoesNotFit {
        profile: HardwareProfile,
        memory: HardwareDeficit,
        limiting_resource: String,
        alternative: Option<HardwareProfile>,
    },
    Unknown {
        reason: HardwareUnknownReason,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareProfile {
    pub context_length: u32,
    pub acceleration: String,
    pub device: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareMemory {
    pub required_bytes: u64,
    pub available_bytes: u64,
    pub headroom_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareDeficit {
    pub required_bytes: u64,
    pub available_bytes: u64,
    pub deficit_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HardwareRecommendation {
    Recommended,
    Constrained,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HardwareUnknownReason {
    UnsupportedBackend,
    EstimatorFailed,
    InsufficientMetadata,
    NoHardwareSnapshot,
    UnsupportedExecutionPlan,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InventoryModel {
    pub id: ModelId,
    pub content_id: ContentId,
    pub created: u64,
    pub name: String,
    pub supported_parameters: Vec<String>,
    pub status: ModelStatus,
    pub source: ModelSource,
    pub location: ModelLocation,
    pub properties: InventoryProperties,
    pub hardware: HardwareAssessment,
    pub operations: Vec<ModelOperation>,
    pub updated_at: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
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
    Assess,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DownloadModelRequest {
    pub source: HuggingFaceDownloadSource,
    pub components: Vec<DownloadComponent>,
    #[serde(default)]
    pub relationships: Vec<ComponentRelationship>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum HuggingFaceDownloadSource {
    HuggingFace {
        repository: String,
        revision: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DownloadComponent {
    pub path: PathBuf,
    pub role: ComponentRole,
    pub shard_index: Option<u32>,
    pub expected_sha256: Option<String>,
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
        model_id: ModelId,
        required_bytes: u64,
        available_bytes: u64,
        completed_bytes: u64,
        total_bytes: u64,
    },
    Progress {
        operation_id: String,
        model_id: ModelId,
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
    Failed {
        operation_id: String,
        model_id: Option<ModelId>,
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DownloadFailure {
    pub code: String,
    pub message: String,
    pub retryable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeletePlan {
    pub model_id: ModelId,
    pub supported: bool,
    pub reason: Option<String>,
    pub reclaimable_bytes: u64,
    pub retained_shared_bytes: u64,
    pub paths: Vec<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeletedModel {
    pub id: ModelId,
    pub deleted: bool,
    pub freed_bytes: u64,
    pub retained_shared_bytes: u64,
    pub plan: DeletePlan,
}

pub type DownloadEventStream = BoxStream<'static, ModelDownloadEvent>;

/// Model inventory boundary consumed by the HTTP API and server composition root.
pub trait ModelInventory: Send + Sync + 'static {
    fn list(&self) -> BoxFuture<'_, Result<Vec<InventoryModel>, InventoryError>>;
    fn get(&self, id: &ModelId) -> BoxFuture<'_, Result<InventoryModel, InventoryError>>;
    fn download(
        &self,
        request: DownloadModelRequest,
    ) -> BoxFuture<'_, Result<DownloadEventStream, InventoryError>>;
    fn plan_delete(&self, id: &ModelId) -> BoxFuture<'_, Result<DeletePlan, InventoryError>>;
    fn delete(&self, id: &ModelId) -> BoxFuture<'_, Result<DeletedModel, InventoryError>>;
    fn resolve_ready(&self, id: &ModelId) -> BoxFuture<'_, Result<ResolvedModel, InventoryError>>;
    fn update_status(
        &self,
        id: &ModelId,
        status: ModelStatus,
    ) -> BoxFuture<'_, Result<(), InventoryError>>;
}

/// Hardware-assessment boundary consumed by the HTTP API.
///
/// The server composition root implements this with `icn-hardware` so the API and model inventory
/// remain independent of llama.cpp.
pub trait ModelHardwareAssessor: Send + Sync + 'static {
    fn assess(&self, id: &ModelId) -> BoxFuture<'_, Result<HardwareAssessment, InventoryError>>;
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
    #[error("internal inventory failure: {0}")]
    Internal(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_require_the_exact_versioned_prefix_and_lowercase_digest() {
        let digest = "a".repeat(64);
        assert!(ModelId::parse(format!("mdl_{digest}")).is_ok());
        assert!(ContentId::parse(format!("content_{digest}")).is_ok());
        assert!(ModelId::parse(format!("content_{digest}")).is_err());
        assert!(ModelId::parse(format!("mdl_{}", "A".repeat(64))).is_err());
        assert!(ModelId::parse("mdl_short").is_err());
    }
}
