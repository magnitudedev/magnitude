//! Transport-neutral local-model package, evaluation, download, and residency contracts.

use std::path::PathBuf;

use futures_util::future::BoxFuture;
use serde::{Deserialize, Serialize};

use crate::{InventoryError, ResolvedModel};

macro_rules! string_id {
    ($name:ident) => {
        #[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub String);
    };
}

string_id!(ModelFileId);
string_id!(ModelPackageId);
string_id!(DownloadAttemptId);
string_id!(SpeculativeDecodingPairId);
string_id!(ModelAssessmentRequestId);
string_id!(ModelOfferingTargetId);
string_id!(ModelServingConfigurationId);
string_id!(OfferingAssessmentId);
string_id!(AssessmentEnvironmentId);
string_id!(RecommendableModelId);
string_id!(RuntimeResidencyId);

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelFileRole {
    Weights,
    Projector,
    Mtp,
    Auxiliary,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelFile {
    pub id: ModelFileId,
    #[cfg_attr(feature = "openapi", schema(value_type = String))]
    pub path: PathBuf,
    pub role: ModelFileRole,
    pub size_bytes: u64,
    pub sha256: String,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub enum ModelPackageSource {
    HuggingFace {
        repository: String,
        revision: String,
    },
    Local {
        #[cfg_attr(feature = "openapi", schema(value_type = String))]
        path: PathBuf,
    },
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase")]
pub enum ModelFileRelationship {
    #[serde(rename_all = "camelCase")]
    Shard {
        file_id: ModelFileId,
        index: u32,
        count: u32,
    },
    #[serde(rename_all = "camelCase")]
    ProjectorFor {
        projector_file_id: ModelFileId,
        weights_file_id: ModelFileId,
    },
    #[serde(rename_all = "camelCase")]
    MtpFor {
        mtp_file_id: ModelFileId,
        weights_file_id: ModelFileId,
    },
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelPackageProperties {
    pub format: String,
    pub quantization: String,
    pub quantization_name: String,
    pub architecture: String,
    pub maximum_context_length: u32,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelPackage {
    pub id: ModelPackageId,
    pub source: ModelPackageSource,
    pub files: Vec<ModelFile>,
    pub relationships: Vec<ModelFileRelationship>,
    pub properties: ModelPackageProperties,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelFailure {
    pub code: String,
    pub message: String,
    pub retryable: bool,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase")]
pub enum ModelPackageInspection {
    Pending,
    Inspected { capabilities: ModelCapabilities },
    Invalid { failure: ModelFailure },
    Incompatible { failure: ModelFailure },
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct InstalledModelPackage {
    pub target_id: ModelOfferingTargetId,
    pub package: ModelPackage,
    #[cfg_attr(feature = "openapi", schema(value_type = String))]
    pub path: PathBuf,
    pub inspection: ModelPackageInspection,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstalledModelPackagesResponse {
    pub packages: Vec<InstalledModelPackage>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RemoveInstalledModelPackageResponse {
    pub package_id: ModelPackageId,
    pub removed: bool,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase")]
pub enum ModelOfferingTarget {
    Package {
        package: ModelPackage,
    },
    SpeculativeDecodingPair {
        id: SpeculativeDecodingPairId,
        target: ModelPackage,
        draft: ModelPackage,
    },
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ServingProfile {
    pub context_length: u32,
    pub parallel_sequences: u32,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelServingConfiguration {
    pub id: ModelServingConfigurationId,
    pub target: ModelOfferingTarget,
    pub profile: ServingProfile,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelReasoningCapabilities {
    pub supported: bool,
    pub efforts: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_effort: Option<String>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelCapabilities {
    pub vision: bool,
    pub tools: bool,
    pub structured_output: bool,
    pub reasoning: ModelReasoningCapabilities,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RecommendableModel {
    pub id: RecommendableModelId,
    pub checkpoint_id: String,
    pub target_id: ModelOfferingTargetId,
    pub target: ModelOfferingTarget,
    pub eligible_serving_profiles: Vec<ServingProfile>,
    pub display_name: String,
    pub description: String,
    pub license: String,
    pub capabilities: ModelCapabilities,
    pub quality_score: f64,
    pub quality_score_provenance: String,
    pub fidelity_rank: u32,
    pub quantization_aware: bool,
    pub quality_evidence: Vec<String>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CatalogDiagnostic {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entry_id: Option<RecommendableModelId>,
    pub failure: ModelFailure,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RecommendableModelCatalog {
    pub models: Vec<RecommendableModel>,
    pub diagnostics: Vec<CatalogDiagnostic>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase")]
pub enum ModelPackageOperand {
    #[serde(rename_all = "camelCase")]
    Installed { package_id: ModelPackageId },
    #[serde(rename_all = "camelCase")]
    SourceBacked { package: ModelPackage },
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase")]
pub enum ModelTargetInput {
    #[serde(rename_all = "camelCase")]
    Package { package: ModelPackageOperand },
    #[serde(rename_all = "camelCase")]
    SpeculativeDecodingPair {
        target: ModelPackageOperand,
        draft: ModelPackageOperand,
    },
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CapacityPolicy {
    pub required_reserve_bytes_per_memory_domain: u64,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AssessModelRequest {
    pub request_id: ModelAssessmentRequestId,
    pub target: ModelTargetInput,
    pub profiles: Vec<ServingProfile>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AssessModelsRequest {
    pub requests: Vec<AssessModelRequest>,
    pub capacity_policy: CapacityPolicy,
    pub include_performance: bool,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MemoryAssessment {
    pub memory_domain_id: String,
    pub capacity_bytes: u64,
    pub required_bytes: u64,
    pub required_reserve_bytes: u64,
    pub remaining_bytes: i64,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PerformanceConfidence {
    High,
    Moderate,
    Low,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PerformanceEvidence {
    pub context_tokens: u32,
    pub lower_tokens_per_second: f64,
    pub estimated_tokens_per_second: f64,
    pub upper_tokens_per_second: f64,
    pub confidence: PerformanceConfidence,
    pub method: String,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PerformanceUnavailable {
    pub method: String,
    pub code: String,
    pub message: String,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase")]
pub enum OfferingAssessment {
    #[serde(rename_all = "camelCase")]
    Fits {
        profile: ServingProfile,
        configuration_id: ModelServingConfigurationId,
        assessment_id: OfferingAssessmentId,
        memory: Vec<MemoryAssessment>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        performance: Option<PerformanceEvidence>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        performance_unavailable: Option<PerformanceUnavailable>,
    },
    #[serde(rename_all = "camelCase")]
    DoesNotFit {
        profile: ServingProfile,
        configuration_id: ModelServingConfigurationId,
        assessment_id: OfferingAssessmentId,
        memory: Vec<MemoryAssessment>,
        limiting_resource: String,
        deficit_bytes: u64,
    },
    #[serde(rename_all = "camelCase")]
    Incompatible {
        profile: ServingProfile,
        configuration_id: ModelServingConfigurationId,
        failure: ModelFailure,
    },
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase")]
pub enum AssessModelResult {
    #[serde(rename_all = "camelCase")]
    Assessed {
        request_id: ModelAssessmentRequestId,
        target_id: ModelOfferingTargetId,
        profiles: Vec<OfferingAssessment>,
    },
    #[serde(rename_all = "camelCase")]
    InvalidTarget {
        request_id: ModelAssessmentRequestId,
        failure: ModelFailure,
    },
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AssessModelsResponse {
    pub environment_id: AssessmentEnvironmentId,
    pub results: Vec<AssessModelResult>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FitModelTarget {
    pub request_id: ModelAssessmentRequestId,
    pub target: ModelTargetInput,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FitModelsRequest {
    pub targets: Vec<FitModelTarget>,
    pub capacity_policy: CapacityPolicy,
    pub minimum_context_length: u32,
    pub maximum_context_length: u32,
    pub maximum_parallel_sequences: u32,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase")]
pub enum FitModelResult {
    #[serde(rename_all = "camelCase")]
    Fitted {
        request_id: ModelAssessmentRequestId,
        target_id: ModelOfferingTargetId,
        configuration: ModelServingConfiguration,
        assessment: OfferingAssessment,
    },
    #[serde(rename_all = "camelCase")]
    DoesNotFit {
        request_id: ModelAssessmentRequestId,
        target_id: ModelOfferingTargetId,
        limiting_resource: String,
        deficit_bytes: u64,
    },
    #[serde(rename_all = "camelCase")]
    InvalidTarget {
        request_id: ModelAssessmentRequestId,
        failure: ModelFailure,
    },
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct FitModelsResponse {
    pub environment_id: AssessmentEnvironmentId,
    pub results: Vec<FitModelResult>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StartModelDownloadRequest {
    pub package: ModelPackage,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase")]
pub enum DownloadAttempt {
    #[serde(rename_all = "camelCase")]
    Pending {
        id: DownloadAttemptId,
        package_id: ModelPackageId,
    },
    #[serde(rename_all = "camelCase")]
    Downloading {
        id: DownloadAttemptId,
        package_id: ModelPackageId,
        completed_bytes: u64,
        total_bytes: u64,
    },
    #[serde(rename_all = "camelCase")]
    Completed {
        id: DownloadAttemptId,
        package_id: ModelPackageId,
    },
    #[serde(rename_all = "camelCase")]
    Failed {
        id: DownloadAttemptId,
        package_id: ModelPackageId,
        completed_bytes: u64,
        total_bytes: u64,
        failure: ModelFailure,
    },
    #[serde(rename_all = "camelCase")]
    Cancelled {
        id: DownloadAttemptId,
        package_id: ModelPackageId,
    },
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StartModelDownloadResponse {
    pub attempt: DownloadAttempt,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelDownloadsResponse {
    pub attempts: Vec<DownloadAttempt>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct LoadModelRequest {
    pub configuration: ModelServingConfiguration,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct LoadModelReady {
    pub residency_id: RuntimeResidencyId,
    pub configuration_id: ModelServingConfigurationId,
    pub execution_evidence_id: String,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelLoadStage {
    Queued,
    Resolving,
    Unloading,
    Loading,
    Verifying,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase")]
pub enum ModelLoadEvent {
    #[serde(rename_all = "camelCase")]
    Progress {
        stage: ModelLoadStage,
        #[serde(skip_serializing_if = "Option::is_none")]
        fraction: Option<f32>,
    },
    Ready {
        ready: LoadModelReady,
    },
    Failed {
        failure: ModelFailure,
    },
}

#[derive(Clone)]
pub struct ResolvedModelTarget {
    pub target_id: ModelOfferingTargetId,
    pub target: ModelOfferingTarget,
    pub target_model: ResolvedModel,
    pub draft_model: Option<ResolvedModel>,
    resolution_guards: Vec<std::sync::Arc<dyn Send + Sync>>,
}

impl std::fmt::Debug for ResolvedModelTarget {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ResolvedModelTarget")
            .field("target_id", &self.target_id)
            .field("target", &self.target)
            .field("target_model", &self.target_model)
            .field("draft_model", &self.draft_model)
            .finish_non_exhaustive()
    }
}

impl ResolvedModelTarget {
    #[must_use]
    pub fn new(
        target_id: ModelOfferingTargetId,
        target: ModelOfferingTarget,
        target_model: ResolvedModel,
        draft_model: Option<ResolvedModel>,
    ) -> Self {
        Self {
            target_id,
            target,
            target_model,
            draft_model,
            resolution_guards: Vec::new(),
        }
    }

    #[must_use]
    pub fn retain_resolution_guard(mut self, guard: impl Send + Sync + 'static) -> Self {
        self.resolution_guards.push(std::sync::Arc::new(guard));
        self
    }
}

/// Installed package and exact-target resolution boundary.
pub trait InstalledModelPackages: Send + Sync + 'static {
    fn list_installed(
        &self,
    ) -> BoxFuture<'_, Result<InstalledModelPackagesResponse, InventoryError>>;
    fn resolve_target(
        &self,
        target: ModelTargetInput,
    ) -> BoxFuture<'_, Result<ResolvedModelTarget, InventoryError>>;
    fn remove_installed(
        &self,
        package_id: &ModelPackageId,
    ) -> BoxFuture<'_, Result<RemoveInstalledModelPackageResponse, InventoryError>>;
}

pub trait RecommendableModelCatalogProvider: Send + Sync + 'static {
    fn catalog(&self) -> BoxFuture<'_, Result<RecommendableModelCatalog, InventoryError>>;
}

pub trait ModelEvaluator: Send + Sync + 'static {
    fn assess(
        &self,
        request: AssessModelsRequest,
    ) -> BoxFuture<'_, Result<AssessModelsResponse, InventoryError>>;
    fn fit(
        &self,
        request: FitModelsRequest,
    ) -> BoxFuture<'_, Result<FitModelsResponse, InventoryError>>;
}

pub trait ModelDownloads: Send + Sync + 'static {
    fn start(
        &self,
        request: StartModelDownloadRequest,
    ) -> BoxFuture<'_, Result<StartModelDownloadResponse, InventoryError>>;
    fn list_attempts(&self) -> BoxFuture<'_, Result<ModelDownloadsResponse, InventoryError>>;
    fn get_attempt(
        &self,
        id: &DownloadAttemptId,
    ) -> BoxFuture<'_, Result<DownloadAttempt, InventoryError>>;
    fn cancel(
        &self,
        id: &DownloadAttemptId,
    ) -> BoxFuture<'_, Result<DownloadAttempt, InventoryError>>;
}
