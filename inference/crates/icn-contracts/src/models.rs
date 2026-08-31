//! Transport-neutral local-model package, assessment, download, and residency contracts.

use std::fmt;
use std::path::{Component, Path, PathBuf};
use std::str::FromStr;

use futures_util::future::BoxFuture;
use futures_util::stream::BoxStream;
use serde::{Deserialize, Deserializer, Serialize, Serializer, de};
use unicode_normalization::UnicodeNormalization;

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
string_id!(ModelAssessmentId);
string_id!(AssessmentEnvironmentId);
string_id!(ModelInstanceId);
string_id!(CatalogInstallationOperationId);

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema), schema(value_type = String))]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModelId(String);

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema), schema(value_type = String))]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CatalogBaseId(String);

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema), schema(value_type = String))]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CatalogVariantId(String);

/// A canonical Hugging Face model repository (`owner/repository`).
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema), schema(value_type = String))]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HuggingFaceRepositoryId(String);

/// The repository-relative GGUF weights selector used in a discovered model identity.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema), schema(value_type = String))]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HuggingFaceArtifactSelector(String);

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("invalid model ID: {message}")]
pub struct ModelIdError {
    message: String,
}

fn model_id_error(message: impl Into<String>) -> ModelIdError {
    ModelIdError {
        message: message.into(),
    }
}

fn validate_normalized_component(value: &str, label: &str) -> Result<(), ModelIdError> {
    if value.is_empty() {
        return Err(model_id_error(format!("{label} must not be empty")));
    }
    if matches!(value, "." | "..") {
        return Err(model_id_error(format!(
            "{label} must not be a traversal component"
        )));
    }
    if value.contains('\\') {
        return Err(model_id_error(format!(
            "{label} must not contain a backslash"
        )));
    }
    if value.chars().any(char::is_control) {
        return Err(model_id_error(format!(
            "{label} must not contain control characters"
        )));
    }
    if value.nfc().ne(value.chars()) {
        return Err(model_id_error(format!(
            "{label} must be NFC-normalized UTF-8"
        )));
    }
    Ok(())
}

impl CatalogBaseId {
    pub fn new(value: impl Into<String>) -> Result<Self, ModelIdError> {
        let value = value.into();
        validate_normalized_component(&value, "catalog base")?;
        if value == "hf" || value.contains(':') || value.contains('/') {
            return Err(model_id_error(
                "catalog base must be one non-hf identity component",
            ));
        }
        Ok(Self(value))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for CatalogBaseId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl CatalogVariantId {
    pub fn new(value: impl Into<String>) -> Result<Self, ModelIdError> {
        let value = value.into();
        let mut components = value.split(':');
        let format = components.next().unwrap_or_default();
        let quality = components.next().unwrap_or_default();
        if components.next().is_some() {
            return Err(model_id_error(
                "catalog variant must have format and quality components",
            ));
        }
        validate_normalized_component(format, "catalog variant format")?;
        validate_normalized_component(quality, "catalog variant quality")?;
        if format.contains('/') || quality.contains('/') {
            return Err(model_id_error(
                "catalog variant components must not contain slashes",
            ));
        }
        Ok(Self(value))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for CatalogVariantId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl HuggingFaceRepositoryId {
    pub fn new(value: impl Into<String>) -> Result<Self, ModelIdError> {
        let value = value.into();
        let mut components = value.split('/');
        let owner = components.next().unwrap_or_default();
        let repository = components.next().unwrap_or_default();
        if components.next().is_some() {
            return Err(model_id_error(
                "repository must have exactly owner and repository components",
            ));
        }
        validate_normalized_component(owner, "owner")?;
        validate_normalized_component(repository, "repository")?;
        Ok(Self(value))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Serialize for HuggingFaceRepositoryId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for HuggingFaceRepositoryId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::new(String::deserialize(deserializer)?).map_err(de::Error::custom)
    }
}

impl HuggingFaceArtifactSelector {
    pub fn new(value: impl Into<String>) -> Result<Self, ModelIdError> {
        let value = value.into();
        let path = Path::new(&value);
        if path.is_absolute() || value.starts_with('/') {
            return Err(model_id_error(
                "artifact selector must be repository-relative",
            ));
        }
        if value.contains('\\') {
            return Err(model_id_error(
                "artifact selector must not contain a backslash",
            ));
        }
        if !value.to_ascii_lowercase().ends_with(".gguf") {
            return Err(model_id_error(
                "artifact selector must identify a GGUF file",
            ));
        }
        if value.split('/').any(|component| component.is_empty()) {
            return Err(model_id_error(
                "artifact selector must not contain empty components",
            ));
        }
        for component in path.components() {
            let Component::Normal(component) = component else {
                return Err(model_id_error(
                    "artifact selector must be a normalized relative path",
                ));
            };
            let component = component
                .to_str()
                .ok_or_else(|| model_id_error("artifact selector must be valid UTF-8"))?;
            validate_normalized_component(component, "artifact selector component")?;
        }
        Ok(Self(value))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Serialize for HuggingFaceArtifactSelector {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for HuggingFaceArtifactSelector {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Self::new(String::deserialize(deserializer)?).map_err(de::Error::custom)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ParsedModelId {
    Catalog {
        base_id: CatalogBaseId,
        variant_id: CatalogVariantId,
    },
    HuggingFace {
        repository_id: HuggingFaceRepositoryId,
        artifact_selector: HuggingFaceArtifactSelector,
    },
}

impl ModelId {
    #[must_use]
    pub fn catalog(base_id: &CatalogBaseId, variant_id: &CatalogVariantId) -> Self {
        Self(format!("{}:{}", base_id.0, variant_id.0))
    }

    #[must_use]
    pub fn hugging_face(
        repository_id: &HuggingFaceRepositoryId,
        artifact_selector: &HuggingFaceArtifactSelector,
    ) -> Self {
        Self(format!("hf:{}/{}", repository_id.0, artifact_selector.0))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    #[must_use]
    pub fn parsed(&self) -> ParsedModelId {
        parse_model_id(&self.0).expect("ModelId construction validates its private representation")
    }
}

impl FromStr for ModelId {
    type Err = ModelIdError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        parse_model_id(value)?;
        Ok(Self(value.to_owned()))
    }
}

fn parse_model_id(value: &str) -> Result<ParsedModelId, ModelIdError> {
    if let Some(remainder) = value.strip_prefix("hf:") {
        let mut components = remainder.splitn(3, '/');
        let owner = components.next().unwrap_or_default();
        let repository = components.next().unwrap_or_default();
        let selector = components.next().unwrap_or_default();
        return Ok(ParsedModelId::HuggingFace {
            repository_id: HuggingFaceRepositoryId::new(format!("{owner}/{repository}"))?,
            artifact_selector: HuggingFaceArtifactSelector::new(selector)?,
        });
    }

    let mut components = value.splitn(2, ':');
    let base_id = CatalogBaseId::new(components.next().unwrap_or_default())?;
    let variant_id = CatalogVariantId::new(components.next().unwrap_or_default())?;
    Ok(ParsedModelId::Catalog {
        base_id,
        variant_id,
    })
}

impl fmt::Display for ModelId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl Serialize for ModelId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for ModelId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        String::deserialize(deserializer)?
            .parse()
            .map_err(de::Error::custom)
    }
}

macro_rules! impl_validated_string {
    ($type:ty, $constructor:expr) => {
        impl Serialize for $type {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                serializer.serialize_str(self.as_str())
            }
        }

        impl<'de> Deserialize<'de> for $type {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                ($constructor)(String::deserialize(deserializer)?).map_err(de::Error::custom)
            }
        }
    };
}

impl_validated_string!(CatalogBaseId, CatalogBaseId::new);
impl_validated_string!(CatalogVariantId, CatalogVariantId::new);

#[cfg(test)]
mod model_id_tests {
    use std::str::FromStr;

    use super::{CatalogBaseId, CatalogVariantId, ModelId, ParsedModelId};

    #[test]
    fn parses_and_round_trips_standalone_and_nested_artifacts() {
        for value in [
            "hf:unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q8_0.gguf",
            "hf:owner/repo/quants/Q4/model-00001-of-00004.gguf",
        ] {
            let parsed = ModelId::from_str(value).expect("parse model ID");
            assert_eq!(parsed.to_string(), value);
            assert_eq!(
                serde_json::to_string(&parsed).expect("serialize"),
                format!("\"{value}\"")
            );
            assert_eq!(
                serde_json::from_str::<ModelId>(&format!("\"{value}\"")).expect("deserialize"),
                parsed,
            );
        }
    }

    #[test]
    fn rejects_non_canonical_or_non_gguf_identities() {
        for value in [
            "HF:owner/repo/model.gguf",
            "hf:owner/repo",
            "hf:owner//model.gguf",
            "hf:/repo/model.gguf",
            "hf:owner/repo//model.gguf",
            "hf:owner/repo/../model.gguf",
            "hf:owner/repo/model.safetensors",
            "hf:owner/repo/Cafe\u{301}.gguf",
            "hf:owner/repo/folder\\model.gguf",
            "hf:owner/repo/model\n.gguf",
        ] {
            assert!(ModelId::from_str(value).is_err(), "accepted {value}");
        }
    }

    #[test]
    fn composes_and_parses_catalog_identity_components() {
        let base = CatalogBaseId::new("qwen3.5-4b").expect("base");
        let variant = CatalogVariantId::new("gguf:q4").expect("variant");
        let id = ModelId::catalog(&base, &variant);
        assert_eq!(id.as_str(), "qwen3.5-4b:gguf:q4");
        assert_eq!(
            id.parsed(),
            ParsedModelId::Catalog {
                base_id: base,
                variant_id: variant,
            }
        );
    }
}

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
    pub model_id: ModelId,
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
    pub model_id: CatalogBaseId,
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
        model_id: CatalogBaseId,
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
    pub freed_bytes: u64,
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
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelMetadata {
    pub format: String,
    pub architecture: String,
    pub quantization: String,
    pub quantization_name: String,
    pub storage_bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(nullable = false))]
    pub maximum_context_length: Option<u32>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ReadyModel {
    pub metadata: ModelMetadata,
    pub profile: ServingProfile,
    pub capabilities: ModelCapabilities,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(nullable = false))]
    pub speculative_method: Option<SpeculativeMethod>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub enum EffectiveModel {
    Ready { model: ReadyModel },
    Unavailable { failure: ModelFailure },
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum ModelInstallationOwnership {
    Magnitude,
    ExternalHuggingFace,
    Mixed,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub enum ModelInstallation {
    #[serde(rename_all = "camelCase")]
    Resolved {
        installed_bytes: u64,
        #[cfg_attr(feature = "openapi", schema(value_type = String))]
        primary_path: PathBuf,
        ownership: ModelInstallationOwnership,
    },
    #[serde(rename_all = "camelCase")]
    Unresolved {
        installed_bytes: u64,
        ownership: ModelInstallationOwnership,
    },
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub enum ResolvedModelInstallation {
    #[serde(rename_all = "camelCase")]
    Resolved {
        installed_bytes: u64,
        #[cfg_attr(feature = "openapi", schema(value_type = String))]
        primary_path: PathBuf,
        ownership: ModelInstallationOwnership,
    },
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
    pub model_id: CatalogBaseId,
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
    pub model_id: CatalogBaseId,
    pub variant_id: CatalogVariantId,
    pub failure: ModelFailure,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub enum CatalogModelUpdate {
    Current,
    #[serde(rename_all = "camelCase")]
    Available {
        required_download_bytes: u64,
    },
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub enum CatalogModelState {
    NotInstalled,
    #[serde(rename_all = "camelCase")]
    Installed {
        effective: EffectiveModel,
        installation: ModelInstallation,
        update_state: CatalogModelUpdate,
    },
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CatalogModel {
    pub id: ModelId,
    pub desired: ReadyModel,
    pub display_name: String,
    pub variant_label: String,
    pub description: String,
    pub release_date: ModelReleaseDate,
    pub license: String,
    pub source_urls: Vec<String>,
    pub parameterization: ModelParameterization,
    pub intelligence: CatalogIntelligence,
    pub fidelity_rank: u32,
    pub quantization_aware: bool,
    pub local_state: CatalogModelState,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CatalogModelsResponse {
    pub revision: u64,
    pub reconciliation_complete: bool,
    pub models: Vec<CatalogModel>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub enum DiscoveredModelCatalogAttribution {
    NotInCatalog,
    Failed { failure: ModelFailure },
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub enum DiscoveredModelState {
    #[serde(rename_all = "camelCase")]
    Ready {
        installation: ResolvedModelInstallation,
        model: ReadyModel,
        catalog_attribution: DiscoveredModelCatalogAttribution,
    },
    #[serde(rename_all = "camelCase")]
    Unavailable {
        installation: ResolvedModelInstallation,
        failure: ModelFailure,
    },
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DiscoveredModel {
    pub id: ModelId,
    pub state: DiscoveredModelState,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DiscoveredModelsResponse {
    pub revision: u64,
    pub reconciliation_complete: bool,
    pub models: Vec<DiscoveredModel>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub enum CatalogInstallationAdmission {
    Current,
    #[serde(rename_all = "camelCase")]
    Admitted {
        operation_id: CatalogInstallationOperationId,
    },
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CatalogInstallationProgress {
    pub stage: DownloadStage,
    pub completed_bytes: u64,
    pub total_bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(nullable = false))]
    pub bytes_per_second: Option<u64>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub enum CatalogInstallationOperationState {
    Pending {
        progress: CatalogInstallationProgress,
    },
    Running {
        progress: CatalogInstallationProgress,
    },
    Completed,
    Failed {
        progress: CatalogInstallationProgress,
        failure: DownloadFailure,
        acknowledged: bool,
    },
    Cancelled {
        progress: CatalogInstallationProgress,
    },
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CatalogInstallationOperation {
    pub operation_id: CatalogInstallationOperationId,
    pub model_id: ModelId,
    pub state: CatalogInstallationOperationState,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CatalogInstallationsResponse {
    pub operations: Vec<CatalogInstallationOperation>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum CatalogInstallationRetentionReason {
    SharedMaterial,
    ExternalOwnership,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub enum CatalogInstallationRemoval {
    #[serde(rename_all = "camelCase")]
    Removed { reclaimed_bytes: u64 },
    Retained {
        reason: CatalogInstallationRetentionReason,
    },
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum CatalogModelSelection {
    Desired,
    Effective,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CatalogAssessmentTarget {
    pub request_id: ModelAssessmentRequestId,
    pub model_id: ModelId,
    pub selection: CatalogModelSelection,
    pub profiles: Vec<ModelAssessmentProfile>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CatalogAssessmentsRequest {
    pub revision: u64,
    pub targets: Vec<CatalogAssessmentTarget>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DiscoveryAssessmentTarget {
    pub request_id: ModelAssessmentRequestId,
    pub model_id: ModelId,
    pub profiles: Vec<ModelAssessmentProfile>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DiscoveryAssessmentsRequest {
    pub revision: u64,
    pub targets: Vec<DiscoveryAssessmentTarget>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub enum ModelAssessmentSubject {
    #[serde(rename_all = "camelCase")]
    Catalog {
        model_id: ModelId,
        selection: CatalogModelSelection,
    },
    #[serde(rename_all = "camelCase")]
    Discovery { model_id: ModelId },
}

impl ModelAssessmentSubject {
    #[must_use]
    pub fn model_id(&self) -> &ModelId {
        match self {
            Self::Catalog { model_id, .. } | Self::Discovery { model_id } => model_id,
        }
    }
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AssessModelRequest {
    pub request_id: ModelAssessmentRequestId,
    pub subject: ModelAssessmentSubject,
    pub profiles: Vec<ModelAssessmentProfile>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AssessModelsRequest {
    pub revision: u64,
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
        profile: ServingProfile,
        assessment_id: ModelAssessmentId,
        memory: Vec<MemoryAssessment>,
        performance: Vec<PerformanceEvidence>,
    },
    #[serde(rename_all = "camelCase")]
    DoesNotFit {
        profile: ServingProfile,
        assessment_id: ModelAssessmentId,
        memory: Vec<MemoryAssessment>,
        limiting_resource: String,
        deficit_bytes: u64,
    },
    #[serde(rename_all = "camelCase")]
    Incompatible {
        profile: ServingProfile,
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
        subject: ModelAssessmentSubject,
        profiles: Vec<ModelAssessment>,
    },
    #[serde(rename_all = "camelCase")]
    Unavailable {
        request_id: ModelAssessmentRequestId,
        subject: ModelAssessmentSubject,
        failure: ModelFailure,
    },
    #[serde(rename_all = "camelCase")]
    Failed {
        request_id: ModelAssessmentRequestId,
        subject: ModelAssessmentSubject,
        failure: ModelFailure,
    },
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase")]
pub enum AssessModelsEvent {
    #[serde(rename_all = "camelCase")]
    Started {
        revision: u64,
        environment_id: AssessmentEnvironmentId,
        total_targets: u32,
    },
    Result {
        result: AssessModelResult,
    },
    #[serde(rename_all = "camelCase")]
    Completed {
        revision: u64,
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
    pub bundle: ServableModelBundle,
    pub target_model: ResolvedModel,
    pub draft_model: Option<ResolvedModel>,
    resolution_guards: Vec<std::sync::Arc<dyn Send + Sync>>,
}

impl std::fmt::Debug for ResolvedServableModelBundle {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ResolvedServableModelBundle")
            .field("bundle", &self.bundle)
            .field("target_model", &self.target_model)
            .field("draft_model", &self.draft_model)
            .finish_non_exhaustive()
    }
}

impl ResolvedServableModelBundle {
    #[must_use]
    pub fn new(
        bundle: ServableModelBundle,
        target_model: ResolvedModel,
        draft_model: Option<ResolvedModel>,
    ) -> Self {
        Self {
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
    fn list_catalog(&self) -> BoxFuture<'_, Result<CatalogModelsResponse, InventoryError>>;
    fn install_catalog_model(
        &self,
        model_id: &ModelId,
    ) -> BoxFuture<'_, Result<CatalogInstallationAdmission, InventoryError>>;
    fn remove_catalog_model_installation(
        &self,
        model_id: &ModelId,
    ) -> BoxFuture<'_, Result<CatalogInstallationRemoval, InventoryError>>;
    fn watch_catalog(&self) -> BoxStream<'static, ModelDomainInvalidation>;
}

pub trait DiscoveredModels: Send + Sync + 'static {
    fn list_discovered(&self) -> BoxFuture<'_, Result<DiscoveredModelsResponse, InventoryError>>;
    fn refresh_discovery(&self) -> BoxFuture<'_, Result<DiscoveredModelsResponse, InventoryError>>;
    fn watch_discovery(&self) -> BoxStream<'static, ModelDomainInvalidation>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelDomainInvalidation {
    pub revision: u64,
}

pub trait CatalogInstallations: Send + Sync + 'static {
    fn list_catalog_installations(
        &self,
    ) -> BoxFuture<'_, Result<CatalogInstallationsResponse, InventoryError>>;
    fn cancel_catalog_installation(
        &self,
        id: &CatalogInstallationOperationId,
    ) -> BoxFuture<'_, Result<CatalogInstallationOperation, InventoryError>>;
    fn acknowledge_catalog_installation_failure(
        &self,
        id: &CatalogInstallationOperationId,
    ) -> BoxFuture<'_, Result<CatalogInstallationOperation, InventoryError>>;
}

pub trait CatalogPackageRemover: Send + Sync + 'static {
    fn remove_catalog_packages(
        &self,
        package_ids: Vec<ModelPackageId>,
    ) -> BoxFuture<'_, Result<u64, InventoryError>>;
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
