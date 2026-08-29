//! Transport-neutral local-model package, assessment, download, and residency contracts.

use std::path::PathBuf;

use futures_util::future::BoxFuture;
use futures_util::stream::BoxStream;
use serde::{Deserialize, Deserializer, Serialize, Serializer, de};

use crate::{DownloadFailure, DownloadStage, InventoryError, ResolvedModel};

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
string_id!(ModelDownloadId);
string_id!(ModelAssessmentRequestId);
string_id!(ServableModelBundleKey);
string_id!(ModelAssessmentId);
string_id!(AssessmentEnvironmentId);
string_id!(CatalogModelId);
string_id!(CatalogVariantId);
string_id!(ModelInstanceId);

#[cfg_attr(
    feature = "openapi",
    derive(utoipa::ToSchema),
    schema(value_type = String, format = Date)
)]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModelReleaseDate(String);

impl ModelReleaseDate {
    pub fn new(value: impl Into<String>) -> Result<Self, String> {
        let value = value.into();
        if is_valid_iso_date(&value) {
            Ok(Self(value))
        } else {
            Err(format!(
                "invalid model release date {value:?}; expected YYYY-MM-DD"
            ))
        }
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Serialize for ModelReleaseDate {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for ModelReleaseDate {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(de::Error::custom)
    }
}

fn is_valid_iso_date(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.len() != 10
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || bytes
            .iter()
            .enumerate()
            .any(|(index, byte)| index != 4 && index != 7 && !byte.is_ascii_digit())
    {
        return false;
    }

    let year = value[0..4].parse::<u32>().expect("validated date digits");
    let month = value[5..7].parse::<u32>().expect("validated date digits");
    let day = value[8..10].parse::<u32>().expect("validated date digits");
    if year == 0 {
        return false;
    }
    let leap_year =
        year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400));
    let days_in_month = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap_year => 29,
        2 => 28,
        _ => return false,
    };
    day > 0 && day <= days_in_month
}

#[cfg(test)]
mod model_release_date_tests {
    use super::ModelReleaseDate;

    #[test]
    fn accepts_real_iso_calendar_dates() {
        let date: ModelReleaseDate =
            serde_json::from_str(r#""2024-02-29""#).expect("deserialize leap-day release date");
        assert_eq!(date.as_str(), "2024-02-29");
        assert_eq!(
            serde_json::to_string(&date).expect("serialize date"),
            r#""2024-02-29""#
        );
    }

    #[test]
    fn rejects_malformed_and_impossible_dates() {
        for value in ["2026-8-13", "2026-02-29", "2026-13-01", "0000-01-01"] {
            assert!(ModelReleaseDate::new(value).is_err(), "accepted {value}");
        }
    }
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelInstanceMemoryDomain {
    pub memory_domain_id: crate::MemoryDomainId,
    pub model_bytes: u64,
    pub context_bytes: u64,
    pub compute_bytes: u64,
    pub auxiliary_bytes: u64,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelInstanceAllocation {
    pub context_window_tokens: u32,
    pub parallel_sequences: u32,
    pub physical_context_tokens: u32,
    pub memory_domains: Vec<ModelInstanceMemoryDomain>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelReleaseReason {
    UserStop,
    IdleTimeout,
    Replacement,
    MemoryPressure,
}

#[cfg(test)]
mod model_release_reason_tests {
    use super::ModelReleaseReason;

    #[test]
    fn serializes_the_complete_product_release_vocabulary() {
        let cases = [
            (ModelReleaseReason::UserStop, "\"user_stop\""),
            (ModelReleaseReason::IdleTimeout, "\"idle_timeout\""),
            (ModelReleaseReason::Replacement, "\"replacement\""),
            (ModelReleaseReason::MemoryPressure, "\"memory_pressure\""),
        ];

        for (reason, expected) in cases {
            assert_eq!(
                serde_json::to_string(&reason).expect("serialize reason"),
                expected
            );
        }
    }
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub enum ModelStoppingAllocation {
    Planned {
        #[serde(skip_serializing_if = "Option::is_none")]
        #[cfg_attr(feature = "openapi", schema(nullable = false))]
        allocation: Option<ModelLoadPlan>,
    },
    Resident {
        allocation: ModelInstanceAllocation,
    },
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub enum ModelInstanceFailure {
    Operation {
        code: String,
        message: String,
        retryable: bool,
    },
    #[serde(rename_all = "camelCase")]
    LowMemory {
        code: String,
        message: String,
        retryable: bool,
        required_system_memory_bytes: u64,
        allocation_headroom_bytes: u64,
        system_reserve_bytes: u64,
        load_boundary_bytes: u64,
        minimum_additional_available_bytes: u64,
        parallel_sequences: u32,
    },
}

impl From<ModelFailure> for ModelInstanceFailure {
    fn from(failure: ModelFailure) -> Self {
        Self::Operation {
            code: failure.code,
            message: failure.message,
            retryable: failure.retryable,
        }
    }
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub enum ModelInstanceLifecycle {
    #[serde(rename_all = "camelCase")]
    Loading {
        stage: ModelLoadStage,
        #[serde(skip_serializing_if = "Option::is_none")]
        #[cfg_attr(feature = "openapi", schema(nullable = false))]
        progress: Option<f32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        #[cfg_attr(feature = "openapi", schema(nullable = false))]
        planned_allocation: Option<ModelLoadPlan>,
    },
    Ready {
        allocation: ModelInstanceAllocation,
    },
    #[serde(rename_all = "camelCase")]
    Stopping {
        reason: ModelReleaseReason,
        allocation: ModelStoppingAllocation,
    },
    Stopped {
        reason: ModelReleaseReason,
    },
    Failed {
        failure: ModelInstanceFailure,
    },
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelInstance {
    pub id: ModelInstanceId,
    /// Canonical callable model identity (for example `model:format:variant`).
    pub model_id: String,
    /// Exact serving configuration admitted for this physical occurrence.
    pub configuration: ModelServingConfiguration,
    pub lifecycle: ModelInstanceLifecycle,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelInstancesSnapshot {
    pub revision: u64,
    /// Instances in lifecycle-update order, oldest first. The final matching
    /// instance is therefore the current attempt for a model.
    pub instances: Vec<ModelInstance>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelInstancesInvalidation {
    pub revision: u64,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelFileRole {
    Weights,
    Projector,
    Draft,
    Mtp,
    Auxiliary,
}

/// Native algorithm used by a speculative decoding stage.
///
/// Method and artifact packaging are intentionally independent. The enclosing speculative bundle
/// declares whether the selected method is embedded in the target or supplied by a separate draft
/// package.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub enum SpeculativeMethod {
    Mtp,
    DFlash,
    DSpark,
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
    /// Exact encoded tensor storage, when bounded GGUF inspection succeeded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tensor_storage_bytes: Option<u64>,
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
    #[serde(rename_all = "camelCase")]
    DraftFor {
        draft_file_id: ModelFileId,
        weights_file_id: ModelFileId,
        method: SpeculativeMethod,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub maximum_context_length: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intrinsic_model_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intrinsic_quality_id: Option<String>,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum ModelPackageInstallationOrigin {
    Magnitude,
    HuggingFaceCache,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum CatalogPackageRole {
    Target,
    Dependency,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CatalogPackageAffiliation {
    pub model_id: CatalogModelId,
    pub variant_id: CatalogVariantId,
    pub package_id: ModelPackageId,
    pub repository: String,
    pub role: CatalogPackageRole,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct InstalledModelPackage {
    pub package: ModelPackage,
    #[cfg_attr(feature = "openapi", schema(value_type = String))]
    pub path: PathBuf,
    pub origin: ModelPackageInstallationOrigin,
    pub inspection: ModelPackageInspection,
    pub catalog_attribution: InstalledCatalogAttribution,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub enum InstalledCatalogAttribution {
    NotCatalogTarget,
    #[serde(rename_all = "camelCase")]
    Attributed {
        model_id: CatalogModelId,
        variant_id: CatalogVariantId,
    },
    Failed {
        failure: ModelFailure,
    },
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct InstalledModelPackagesResponse {
    pub revision: u64,
    pub reconciliation_complete: bool,
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
pub enum ServableModelBundle {
    Standalone {
        package: ModelPackage,
    },
    #[serde(rename_all = "camelCase")]
    SpeculativeDecoding {
        target: ModelPackage,
        draft_source: SpeculativeDraftSource,
        method: SpeculativeMethod,
    },
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase")]
pub enum SpeculativeDraftSource {
    Embedded,
    Separate { draft: ModelPackage },
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ServingProfile {
    pub context_length: u32,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelServingConfiguration {
    pub bundle: ServableModelBundle,
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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "architecture", rename_all = "camelCase", deny_unknown_fields)]
pub enum ModelParameterization {
    Dense {
        #[serde(rename = "totalParameters")]
        total_parameters: u64,
    },
    MixtureOfExperts {
        #[serde(rename = "totalParameters")]
        total_parameters: u64,
        #[serde(rename = "activeParameters")]
        active_parameters: u64,
    },
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum IntelligenceEstimateConfidence {
    High,
    Moderate,
    Low,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum IntelligenceTarget {
    ArtificialAnalysisIntelligenceIndex,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", deny_unknown_fields)]
pub enum IntelligenceProvenance {
    #[serde(
        rename = "artificialAnalysisIntelligenceIndex",
        rename_all = "camelCase"
    )]
    ArtificialAnalysisIntelligenceIndex {
        methodology_version: String,
        as_of_date: String,
        url: String,
    },
    #[serde(rename = "estimate", rename_all = "camelCase")]
    Estimate {
        #[cfg_attr(feature = "openapi", schema(value_type = String))]
        target: IntelligenceTarget,
        methodology_version: String,
        as_of_date: String,
        confidence: IntelligenceEstimateConfidence,
        methodology: String,
        #[cfg_attr(feature = "openapi", schema(min_items = 1))]
        evidence_urls: Vec<String>,
    },
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CatalogIntelligence {
    pub score: f64,
    pub provenance: IntelligenceProvenance,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RecommendableModel {
    pub model_id: CatalogModelId,
    pub variant_id: CatalogVariantId,
    pub configuration: ModelServingConfiguration,
    pub display_name: String,
    pub variant_label: String,
    pub description: String,
    pub release_date: ModelReleaseDate,
    pub license: String,
    pub capabilities: ModelCapabilities,
    pub parameterization: ModelParameterization,
    pub intelligence: CatalogIntelligence,
    pub fidelity_rank: u32,
    pub quantization_aware: bool,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CatalogDiagnostic {
    pub model_id: CatalogModelId,
    pub variant_id: CatalogVariantId,
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
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub enum CatalogModelEffectiveConfiguration {
    Runnable {
        configuration: ModelServingConfiguration,
    },
    Unavailable {
        failure: ModelFailure,
    },
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CatalogModelInstallation {
    pub effective_configuration: CatalogModelEffectiveConfiguration,
    pub packages: Vec<InstalledModelPackage>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub enum CatalogModelUpdateState {
    Current,
    #[serde(rename_all = "camelCase")]
    Available {
        missing_package_ids: Vec<ModelPackageId>,
        superseded_package_ids: Vec<ModelPackageId>,
    },
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub enum CatalogModelLocalState {
    NotInstalled,
    #[serde(rename_all = "camelCase")]
    Installed {
        installation: CatalogModelInstallation,
        update_state: CatalogModelUpdateState,
    },
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct InferenceModel {
    /// Canonical callable identity used unchanged by native and standard inference APIs.
    pub id: String,
    pub model_id: CatalogModelId,
    pub variant_id: CatalogVariantId,
    pub desired_configuration: ModelServingConfiguration,
    pub display_name: String,
    pub variant_label: String,
    pub description: String,
    pub release_date: ModelReleaseDate,
    pub license: String,
    pub capabilities: ModelCapabilities,
    pub parameterization: ModelParameterization,
    pub intelligence: CatalogIntelligence,
    pub fidelity_rank: u32,
    pub quantization_aware: bool,
    pub local_state: CatalogModelLocalState,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelsResponse {
    pub revision: u64,
    pub reconciliation_complete: bool,
    pub models: Vec<InferenceModel>,
    pub diagnostics: Vec<CatalogDiagnostic>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct InstallCatalogModelRequest {
    pub model_id: CatalogModelId,
    pub variant_id: CatalogVariantId,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub enum InstallModelResponse {
    Current,
    #[serde(rename_all = "camelCase")]
    DownloadAdmitted {
        download_id: ModelDownloadId,
    },
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
pub enum ModelBundleInput {
    #[serde(rename_all = "camelCase")]
    Standalone { package: ModelPackageOperand },
    #[serde(rename_all = "camelCase")]
    SpeculativeDecoding {
        target: ModelPackageOperand,
        draft_source: SpeculativeDraftSourceInput,
        method: SpeculativeMethod,
    },
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase")]
pub enum SpeculativeDraftSourceInput {
    Embedded,
    Separate { draft: ModelPackageOperand },
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelAssessmentProfile {
    pub profile: ServingProfile,
    pub performance_context_tokens: Vec<u32>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AssessModelRequest {
    pub request_id: ModelAssessmentRequestId,
    pub bundle: ModelBundleInput,
    pub profiles: Vec<ModelAssessmentProfile>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AssessModelsRequest {
    pub requests: Vec<AssessModelRequest>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MemoryAssessment {
    pub memory_domain_id: crate::MemoryDomainId,
    pub capacity_bytes: u64,
    pub required_bytes: u64,
    pub compatibility_reserve_bytes: u64,
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
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase")]
pub enum ModelAssessment {
    #[serde(rename_all = "camelCase")]
    Fits {
        configuration: ModelServingConfiguration,
        assessment_id: ModelAssessmentId,
        memory: Vec<MemoryAssessment>,
        performance: Vec<PerformanceEvidence>,
    },
    #[serde(rename_all = "camelCase")]
    DoesNotFit {
        configuration: ModelServingConfiguration,
        assessment_id: ModelAssessmentId,
        memory: Vec<MemoryAssessment>,
        limiting_resource: String,
        deficit_bytes: u64,
    },
    #[serde(rename_all = "camelCase")]
    Incompatible {
        configuration: ModelServingConfiguration,
        failure: ModelFailure,
    },
}

impl ModelAssessment {
    #[must_use]
    pub fn is_valid_for(&self, topology: &crate::MemoryTopology) -> bool {
        match self {
            Self::Fits { memory, .. } | Self::DoesNotFit { memory, .. } => {
                let mut seen = std::collections::BTreeSet::new();
                memory.iter().all(|assessment| {
                    let Some(capacity) = topology.capacity(&assessment.memory_domain_id) else {
                        return false;
                    };
                    seen.insert(assessment.memory_domain_id.clone())
                        && assessment.capacity_bytes == capacity
                        && assessment.remaining_bytes
                            == (i128::from(assessment.capacity_bytes)
                                - i128::from(assessment.compatibility_reserve_bytes)
                                - i128::from(assessment.required_bytes))
                            .clamp(i128::from(i64::MIN), i128::from(i64::MAX))
                                as i64
                }) && seen.contains(&crate::MemoryDomainId::system())
            }
            Self::Incompatible { .. } => true,
        }
    }
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase")]
pub enum AssessModelResult {
    #[serde(rename_all = "camelCase")]
    Assessed {
        request_id: ModelAssessmentRequestId,
        profiles: Vec<ModelAssessment>,
    },
    #[serde(rename_all = "camelCase")]
    InvalidBundle {
        request_id: ModelAssessmentRequestId,
        failure: ModelFailure,
    },
    #[serde(rename_all = "camelCase")]
    Failed {
        request_id: ModelAssessmentRequestId,
        failure: ModelFailure,
    },
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase")]
pub enum AssessModelsEvent {
    #[serde(rename_all = "camelCase")]
    Started {
        environment_id: AssessmentEnvironmentId,
        total_targets: u32,
    },
    Result {
        result: AssessModelResult,
    },
    #[serde(rename_all = "camelCase")]
    Completed {
        environment_id: AssessmentEnvironmentId,
        total_targets: u32,
    },
}

pub type ModelAssessmentStream = BoxStream<'static, AssessModelsEvent>;

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StartModelDownloadRequest {
    pub bundle: ServableModelBundle,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase")]
pub enum ModelDownloadState {
    #[serde(rename_all = "camelCase")]
    Pending {
        completed_bytes: u64,
        total_bytes: u64,
    },
    #[serde(rename_all = "camelCase")]
    Downloading {
        stage: DownloadStage,
        completed_bytes: u64,
        total_bytes: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        bytes_per_second: Option<u64>,
    },
    Completed,
    #[serde(rename_all = "camelCase")]
    Failed {
        completed_bytes: u64,
        total_bytes: u64,
        failure: DownloadFailure,
        acknowledged: bool,
    },
    #[serde(rename_all = "camelCase")]
    Cancelled {
        completed_bytes: u64,
        total_bytes: u64,
    },
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelDownload {
    pub id: ModelDownloadId,
    pub bundle: ServableModelBundle,
    pub state: ModelDownloadState,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StartModelDownloadResponse {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(nullable = false))]
    pub download: Option<ModelDownload>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelDownloadsResponse {
    pub downloads: Vec<ModelDownload>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelLoadPlan {
    pub context_window_tokens: u32,
    pub parallel_sequences: u32,
    pub physical_context_tokens: u32,
    pub required_system_memory_bytes: u64,
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

#[derive(Clone)]
pub struct ResolvedServableModelBundle {
    pub bundle_key: ServableModelBundleKey,
    pub bundle: ServableModelBundle,
    pub target_model: ResolvedModel,
    pub draft_model: Option<ResolvedModel>,
    resolution_guards: Vec<std::sync::Arc<dyn Send + Sync>>,
}

impl std::fmt::Debug for ResolvedServableModelBundle {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ResolvedServableModelBundle")
            .field("bundle_key", &self.bundle_key)
            .field("bundle", &self.bundle)
            .field("target_model", &self.target_model)
            .field("draft_model", &self.draft_model)
            .finish_non_exhaustive()
    }
}

impl ResolvedServableModelBundle {
    #[must_use]
    pub fn new(
        bundle_key: ServableModelBundleKey,
        bundle: ServableModelBundle,
        target_model: ResolvedModel,
        draft_model: Option<ResolvedModel>,
    ) -> Self {
        Self {
            bundle_key,
            bundle,
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

/// Installed package and exact-bundle resolution boundary.
pub trait InstalledModelPackages: Send + Sync + 'static {
    fn list_installed(
        &self,
    ) -> BoxFuture<'_, Result<InstalledModelPackagesResponse, InventoryError>>;
    fn resolve_bundle(
        &self,
        bundle: ModelBundleInput,
    ) -> BoxFuture<'_, Result<ResolvedServableModelBundle, InventoryError>>;
    fn remove_installed(
        &self,
        package_id: &ModelPackageId,
    ) -> BoxFuture<'_, Result<RemoveInstalledModelPackageResponse, InventoryError>>;
}

pub trait RecommendableModelCatalogProvider: Send + Sync + 'static {
    fn catalog(&self) -> BoxFuture<'_, Result<RecommendableModelCatalog, InventoryError>>;
}

pub trait CatalogModels: Send + Sync + 'static {
    fn list(&self) -> BoxFuture<'_, Result<ModelsResponse, InventoryError>>;
    fn install(
        &self,
        request: InstallCatalogModelRequest,
    ) -> BoxFuture<'_, Result<InstallModelResponse, InventoryError>>;
}

pub trait CatalogPackageRemover: Send + Sync + 'static {
    fn remove_catalog_package(
        &self,
        package_id: ModelPackageId,
    ) -> BoxFuture<'_, Result<(), InventoryError>>;
}

pub trait ModelAssessor: Send + Sync + 'static {
    fn assess(
        &self,
        request: AssessModelsRequest,
    ) -> BoxFuture<'_, Result<ModelAssessmentStream, InventoryError>>;
}

pub trait ModelDownloads: Send + Sync + 'static {
    fn start(
        &self,
        request: StartModelDownloadRequest,
    ) -> BoxFuture<'_, Result<StartModelDownloadResponse, InventoryError>>;
    fn list(&self) -> BoxFuture<'_, Result<ModelDownloadsResponse, InventoryError>>;
    fn cancel(&self, id: &ModelDownloadId) -> BoxFuture<'_, Result<ModelDownload, InventoryError>>;
    fn acknowledge_failure(
        &self,
        id: &ModelDownloadId,
    ) -> BoxFuture<'_, Result<ModelDownload, InventoryError>>;
    fn watch(&self) -> BoxStream<'static, ModelDownloadsInvalidation>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelDownloadsInvalidation {
    pub revision: u64,
}
