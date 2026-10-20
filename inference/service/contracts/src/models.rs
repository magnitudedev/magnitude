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

/// The repository-relative GGUF weights selector used in a Hugging Face model identity.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema), schema(value_type = String))]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HuggingFaceArtifactSelector(String);

/// The lowercase hex sha256 of a primary GGUF file's header bytes, through the aligned tensor-data
/// offset. It identifies a custom model with no Hugging Face identity.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema), schema(value_type = String))]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LocalModelDigest(String);

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
        if matches!(value.as_str(), "hf" | "local") || value.contains(':') || value.contains('/') {
            return Err(model_id_error(
                "catalog base must be one identity component other than hf or local",
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

impl LocalModelDigest {
    pub fn new(value: impl Into<String>) -> Result<Self, ModelIdError> {
        let value = value.into();
        if value.len() != 64
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(model_id_error(
                "local model digest must be 64 lowercase hex characters",
            ));
        }
        Ok(Self(value))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for LocalModelDigest {
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
    Local {
        digest: LocalModelDigest,
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
    pub fn local(digest: &LocalModelDigest) -> Self {
        Self(format!("local:{}", digest.0))
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

    if let Some(digest) = value.strip_prefix("local:") {
        return Ok(ParsedModelId::Local {
            digest: LocalModelDigest::new(digest)?,
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
impl_validated_string!(LocalModelDigest, LocalModelDigest::new);

#[cfg(test)]
mod model_id_tests {
    use std::str::FromStr;

    use super::{CatalogBaseId, CatalogVariantId, LocalModelDigest, ModelId, ParsedModelId};

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

    #[test]
    fn composes_parses_and_round_trips_local_identity() {
        let digest = LocalModelDigest::new("0123456789abcdef".repeat(4)).expect("digest");
        let id = ModelId::local(&digest);
        assert_eq!(
            id.as_str(),
            format!("local:{}", "0123456789abcdef".repeat(4))
        );
        assert_eq!(id.parsed(), ParsedModelId::Local { digest });
        let encoded = serde_json::to_string(&id).expect("serialize");
        assert_eq!(
            serde_json::from_str::<ModelId>(&encoded).expect("deserialize"),
            id
        );
    }

    #[test]
    fn rejects_non_canonical_local_identities() {
        for value in [
            "local:".to_owned(),
            format!("local:{}", "A".repeat(64)),
            format!("local:{}", "a".repeat(63)),
            format!("local:{}", "a".repeat(65)),
            format!("local:{}", "g".repeat(64)),
            format!("LOCAL:{}", "a".repeat(64)),
        ] {
            assert!(ModelId::from_str(&value).is_err(), "accepted {value}");
        }
    }

    #[test]
    fn reserves_local_and_hf_as_catalog_bases() {
        assert!(CatalogBaseId::new("local").is_err());
        assert!(CatalogBaseId::new("hf").is_err());
        assert!(ModelId::from_str("local:gguf:q4").is_err());
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
    /// The engine heap's observed standing per memory domain at the last observation. Memory is
    /// elastic, so this is not a fixed reservation.
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
    Failure,
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
            (ModelReleaseReason::Failure, "\"failure\""),
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
    /// The model fits this machine but could not be given its memory now.
    MemoryShortage {
        code: String,
        message: String,
        retryable: bool,
        shortage: MemoryShortage,
    },
}

/// Why a model that fits the machine could not be given its memory.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub enum MemoryShortage {
    /// Memory that cannot be moved is in the way of the load's claim in its limiting memory
    /// domain. `available_bytes` is what a load may claim there: the reserve is already taken out.
    #[serde(rename_all = "camelCase")]
    Blocked {
        required_bytes: u64,
        available_bytes: u64,
    },
    /// A memory domain the load uses was under pressure: at its reserve, or with other programs'
    /// memory being displaced faster than the system absorbs. No byte count describes it.
    UnderPressure {},
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
        /// The load's completed fraction in `[0, 1]`, by measured work: tuning, when the load
        /// tunes, is the first half and weight import the rest. Monotonic within a load; only
        /// `Ready` means the load is complete.
        fraction: f32,
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
    DFlash2,
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
pub enum PackageValidation {
    Pending,
    Valid,
    Invalid { failure: ModelFailure },
    Unsupported { failure: ModelFailure },
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum ModelPackageInstallationOrigin {
    Magnitude,
    HuggingFaceCache,
    LmStudio,
    LlamaCpp,
    Ollama,
    Folder,
    Imported,
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
    pub validation: PackageValidation,
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
    External,
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

/// What the release promises for a catalog model. It is independent of assessment, which says
/// what one device can do with it.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "level", rename_all = "camelCase", deny_unknown_fields)]
pub enum CatalogSupport {
    /// Intended to be available as a supported catalog model.
    Supported,
    /// Temporarily unavailable or not yet supported. Never installed or loaded.
    Disabled { reason: String },
    /// Never offered for new installs and never loaded. An installed copy stays identifiable and
    /// removable, and points at its replacement.
    #[serde(rename_all = "camelCase")]
    Deprecated {
        since: String,
        replacement: ModelId,
        reason: String,
    },
}

impl CatalogSupport {
    #[must_use]
    pub fn is_deprecated(&self) -> bool {
        matches!(self, Self::Deprecated { .. })
    }

    #[must_use]
    pub fn is_available(&self) -> bool {
        matches!(self, Self::Supported)
    }
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
    pub parameterization: ModelParameterization,
    pub intelligence: u32,
    pub support: CatalogSupport,
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
        parts: Vec<CatalogUpdatePart>,
    },
}

/// What a catalog update brings: new model weights, a new projector (image input), or a new drafter.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum CatalogUpdatePart {
    Model,
    Projector,
    Drafter,
}

/// A part of a catalog model's bundle that can be fetched on its own: the projector that gives
/// its target image input, or its separate drafter.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum CatalogComponent {
    Projector,
    Drafter,
}

/// A catalog component absent from this computer while the model's exact target weights are
/// found elsewhere. It is not an update: the model serves without it until it is fetched.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CatalogMissingComponent {
    pub component: CatalogComponent,
    pub download_bytes: u64,
}

/// A catalog model's state here. `drafter_withheld`: it serves, and is assessed, without its
/// catalog drafter, which it doesn't fit with: it loads when it fits without it, and otherwise is
/// refused with its own shortfall. `stored_components`: the parts
/// Magnitude stores for a model whose weights are only in another app's location (its projector
/// package and drafter), which Remove deletes while the found weights stay and keep serving.
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
        missing_components: Vec<CatalogMissingComponent>,
        stored_components: Vec<CatalogComponent>,
        drafter_withheld: bool,
    },
}

/// A catalog variant and its state on this computer. `aliases` are the `hf:` IDs its target's
/// files found in an external location (a Hugging Face cache) would have as custom models, which
/// clients resolve to this model, so references made before the file was attributed keep working.
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
    pub intelligence: u32,
    pub support: CatalogSupport,
    pub fidelity_rank: u32,
    pub quantization_aware: bool,
    pub local_state: CatalogModelState,
    #[serde(default)]
    pub aliases: Vec<ModelId>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CatalogModelsResponse {
    pub revision: u64,
    pub reconciliation_complete: bool,
    pub models: Vec<CatalogModel>,
}

/// Where a scanned custom model was found.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub enum FoundSource {
    HuggingFaceCache,
    LmStudio,
    LlamaCpp,
    Ollama,
    Folder {
        #[cfg_attr(feature = "openapi", schema(value_type = String))]
        path: PathBuf,
    },
}

/// How a custom model came to be on this computer, which decides who owns its files.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub enum CustomModelOrigin {
    Added {
        repository: HuggingFaceRepositoryId,
        commit: String,
    },
    Imported {
        #[cfg_attr(feature = "openapi", schema(value_type = String))]
        path: PathBuf,
    },
    Found {
        source: FoundSource,
    },
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CustomModelFile {
    #[cfg_attr(feature = "openapi", schema(value_type = String))]
    pub path: PathBuf,
    pub size_bytes: u64,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub enum CustomModelCatalogAttribution {
    NotInCatalog,
    Failed { failure: ModelFailure },
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub enum CustomModelState {
    NotInstalled,
    #[serde(rename_all = "camelCase")]
    Ready {
        installation: ResolvedModelInstallation,
        model: ReadyModel,
        catalog_attribution: CustomModelCatalogAttribution,
    },
    #[serde(rename_all = "camelCase")]
    Unavailable {
        installation: ResolvedModelInstallation,
        failure: ModelFailure,
    },
}

/// Why the engine cannot execute a model. The engine updates independently of its clients, so the
/// contract stays open: a reason this contract does not name yet arrives as `Unrecognized` with its
/// tag, and clients render it with its `detail`. The service itself only emits `Known` reasons. A
/// reason decodes as the first case it matches, so the schema is an `anyOf`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum UnsupportedReason {
    Known(KnownUnsupportedReason),
    Unrecognized(UnrecognizedUnsupportedReason),
}

#[cfg(feature = "openapi")]
impl utoipa::PartialSchema for UnsupportedReason {
    fn schema() -> utoipa::openapi::RefOr<utoipa::openapi::schema::Schema> {
        use utoipa::ToSchema;
        utoipa::openapi::schema::AnyOfBuilder::new()
            .item(utoipa::openapi::Ref::from_schema_name(
                KnownUnsupportedReason::name(),
            ))
            .item(utoipa::openapi::Ref::from_schema_name(
                UnrecognizedUnsupportedReason::name(),
            ))
            .description(Some(
                "Why the engine cannot execute a model; a tag this contract does not name yet \
                 matches the unrecognized case.",
            ))
            .into()
    }
}

#[cfg(feature = "openapi")]
impl utoipa::ToSchema for UnsupportedReason {
    fn schemas(
        schemas: &mut Vec<(
            String,
            utoipa::openapi::RefOr<utoipa::openapi::schema::Schema>,
        )>,
    ) {
        use utoipa::PartialSchema;
        schemas.push((
            KnownUnsupportedReason::name().into(),
            KnownUnsupportedReason::schema(),
        ));
        <KnownUnsupportedReason as utoipa::ToSchema>::schemas(schemas);
        schemas.push((
            UnrecognizedUnsupportedReason::name().into(),
            UnrecognizedUnsupportedReason::schema(),
        ));
    }
}

impl From<KnownUnsupportedReason> for UnsupportedReason {
    fn from(reason: KnownUnsupportedReason) -> Self {
        Self::Known(reason)
    }
}

/// A reason the engine raises, typed where it raises it. `DraftArchitecture` is a separate drafter
/// whose architecture no drafter family binds, such as a Nemotron-H MTP file. `UnsupportedFormat`
/// is the service's own: a found model whose files are not GGUF, such as an Ollama model stored for
/// its MLX runner.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub enum KnownUnsupportedReason {
    Architecture {
        architecture: String,
    },
    WeightTypes {
        types: Vec<WeightTypeCount>,
    },
    RopeScaling {
        kind: String,
    },
    UnrecognizedTensor {
        name: String,
    },
    UnrecognizedMetadata {
        key: String,
    },
    MissingMetadata {
        key: String,
    },
    Tokenizer {
        model: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[cfg_attr(feature = "openapi", schema(nullable = false))]
        pre: Option<String>,
    },
    ChatTemplate,
    KernelDomain {
        backend: String,
        entry: String,
        statics: Vec<KernelStatic>,
    },
    Backend {
        backend: String,
    },
    TargetMismatch {
        detail: String,
    },
    TiedOutputHead,
    DraftArchitecture {
        architecture: String,
    },
    Unreadable,
    UnsupportedFormat,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnrecognizedUnsupportedReason {
    #[serde(rename = "_tag")]
    pub tag: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(nullable = false))]
    pub detail: Option<String>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct WeightTypeCount {
    pub ggml_type: String,
    pub tensor_count: u32,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct KernelStatic {
    pub name: String,
    pub value: u64,
}

/// The shape a drafter binds to. Requantizations, finetunes and abliterations keep it, so it
/// matches a model to its parent catalog model when nothing else does.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelFingerprint {
    pub architecture: String,
    pub block_count: u32,
    pub embedding_length: u32,
    pub head_count: u32,
    pub head_count_kv: u32,
    pub key_length: u32,
    pub vocab_size: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(nullable = false))]
    pub expert_count: Option<u32>,
    pub tokenizer_digest: String,
}

/// Whether the engine binds a model's drafter. `None` means no drafter was configured.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub enum DrafterVerdict {
    None,
    #[serde(rename_all = "camelCase")]
    Bound {
        method: SpeculativeMethod,
        memory_bytes: u64,
    },
    Rejected {
        reasons: Vec<UnsupportedReason>,
        detail: String,
    },
}

/// Whether the engine binds a model's projector. `None` means the model has no projector; a
/// `Rejected` projector leaves the model assessed and served without it, taking text input only.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub enum ProjectorVerdict {
    None,
    Bound,
    Rejected {
        reasons: Vec<UnsupportedReason>,
        detail: String,
    },
}

/// A drafter's files, independent of which model uses it.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub enum DrafterReference {
    #[serde(rename_all = "camelCase")]
    Catalog {
        model_id: ModelId,
    },
    Embedded,
    HuggingFace {
        repository: HuggingFaceRepositoryId,
        commit: String,
        path: String,
    },
    Local {
        #[cfg_attr(feature = "openapi", schema(value_type = String))]
        path: PathBuf,
    },
}

/// Where a drafter candidate comes from, in the order automatic choice prefers: a separate drafter
/// in the model's own repository or folder, the matched catalog model's drafter, MTP heads built
/// into the model, then a drafter the user entered, which is never chosen automatically.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum DrafterCategory {
    NextToModel,
    CatalogDrafter,
    BuiltIn,
    Other,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum DrafterLineage {
    Exact,
    Lineage,
    Fingerprint,
    None,
}

/// One drafter a model may use. `automatic` says whether automatic choice may take it: the engine
/// binds it, the user didn't enter it, and with it the model still fits when it fits without one.
/// The first automatic candidate, in order, is the preselected drafter.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DrafterCandidate {
    pub category: DrafterCategory,
    pub reference: DrafterReference,
    pub name: String,
    pub label: String,
    pub size_bytes: u64,
    pub lineage: DrafterLineage,
    pub verdict: DrafterVerdict,
    pub automatic: bool,
    pub on_this_computer: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(nullable = false))]
    pub method: Option<SpeculativeMethod>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub enum AttachedDrafterState {
    Ready,
    NotInstalled,
    Rejected {
        reasons: Vec<UnsupportedReason>,
        detail: String,
    },
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AttachedDrafter {
    pub reference: DrafterReference,
    pub method: SpeculativeMethod,
    pub state: AttachedDrafterState,
    pub update_available: bool,
}

/// The drafter a custom model should use: the recommended one, none, or an exact reference.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub enum DrafterChoice {
    Auto,
    None,
    Reference {
        reference: DrafterReference,
        method: SpeculativeMethod,
    },
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DrafterAttachment {
    pub model_id: ModelId,
    pub drafter: DrafterChoice,
}

/// A model on this computer that is not a catalog model. `drafter_pending`: its drafter choice is
/// automatic and its automatic drafter isn't worked out yet, so it has no drafter for now rather
/// than none. `update_commit` is set only for an added model whose repository's `main` has moved to
/// a commit where its target file has other content.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CustomModel {
    pub model_id: ModelId,
    pub origin: CustomModelOrigin,
    pub display_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(nullable = false))]
    pub variant_label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(nullable = false))]
    pub variant_of: Option<ModelId>,
    pub files: Vec<CustomModelFile>,
    pub size_bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(nullable = false))]
    pub projector: Option<CustomModelFile>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(nullable = false))]
    pub drafter: Option<AttachedDrafter>,
    pub drafter_pending: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(nullable = false))]
    pub recommended_drafter: Option<DrafterCandidate>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(nullable = false))]
    pub license: Option<String>,
    pub profile: ServingProfile,
    pub state: CustomModelState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(nullable = false))]
    pub update_commit: Option<String>,
}

/// A GGUF found on this computer that the engine cannot execute, or a directory inside a model
/// location that Magnitude can't read, which may hold more models.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct UnsupportedLocalModel {
    pub display_name: String,
    pub origin: CustomModelOrigin,
    #[cfg_attr(feature = "openapi", schema(value_type = String))]
    pub path: PathBuf,
    pub reasons: Vec<UnsupportedReason>,
    pub detail: String,
}

/// `models` are the custom models to offer and assess; `hidden` are the ones the configuration
/// hides, listed so a client can unhide them, and never assessed. `unsupported` are GGUF groups
/// that cannot be read at all, and directories inside a location that cannot be read (the rest of
/// the location is still listed); a readable model the engine rejects stays in `models` with an
/// `Unsupported` assessment.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct LocalModelsResponse {
    pub revision: u64,
    pub reconciliation_complete: bool,
    pub models: Vec<CustomModel>,
    pub hidden: Vec<CustomModel>,
    pub unsupported: Vec<UnsupportedLocalModel>,
}

/// The result of removing an added model's Magnitude-owned files. Files in external locations
/// are never touched, so the model may remain as found.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct LocalModelRemoval {
    pub model_id: ModelId,
    pub removed: bool,
    pub freed_bytes: u64,
}

/// Magnitude-owned drafter packages removed because no model uses them any more.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DrafterPruning {
    pub removed: Vec<ModelPackageId>,
    pub freed_bytes: u64,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum ModelRootKind {
    HuggingFaceCache,
    LmStudio,
    LlamaCpp,
    Ollama,
    Folder,
}

/// One resolved location ICN scans. ACN resolves the paths; ICN never guesses them.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelRoot {
    pub kind: ModelRootKind,
    #[cfg_attr(feature = "openapi", schema(value_type = String))]
    pub path: PathBuf,
    pub enabled: bool,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub enum CustomModelDeclarationOrigin {
    Added {
        repository: HuggingFaceRepositoryId,
        commit: String,
    },
    Imported {
        #[cfg_attr(feature = "openapi", schema(value_type = String))]
        path: PathBuf,
    },
}

/// The persisted record of an added or imported model. For `Added`, `target` and `projector` are
/// repository-relative paths at the pinned commit; for `Imported` they are absolute file paths,
/// with the imported file first.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CustomModelDeclaration {
    pub model_id: ModelId,
    pub origin: CustomModelDeclarationOrigin,
    pub target: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(nullable = false))]
    pub projector: Option<String>,
}

/// The whole local-models configuration ACN owns and hands to ICN, at spawn and after every
/// change. Applying it is idempotent.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct LocalModelsConfiguration {
    pub roots: Vec<ModelRoot>,
    pub declarations: Vec<CustomModelDeclaration>,
    pub hidden: Vec<ModelId>,
    pub drafters: Vec<DrafterAttachment>,
}

/// Files not yet added: a Hugging Face repository at a commit, or a local directory.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub enum CandidateSource {
    HuggingFace {
        repository: HuggingFaceRepositoryId,
        commit: String,
    },
    Local {
        #[cfg_attr(feature = "openapi", schema(value_type = String))]
        path: PathBuf,
    },
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CandidateFile {
    pub path: String,
    pub size_bytes: u64,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CandidateDrafter {
    pub reference: DrafterReference,
    pub method: SpeculativeMethod,
}

/// A target (shards in order), projector and drafter, by paths relative to `source`.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelCandidate {
    pub source: CandidateSource,
    pub target: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(nullable = false))]
    pub projector: Option<CandidateFile>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(nullable = false))]
    pub drafter: Option<CandidateDrafter>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub enum CandidateInspectionRequest {
    HuggingFace {
        repository: HuggingFaceRepositoryId,
        revision: String,
    },
    Local {
        #[cfg_attr(feature = "openapi", schema(value_type = String))]
        path: PathBuf,
    },
}

/// One version of a candidate. `download_bytes` is what adding it downloads before any drafter:
/// its files and the repository's projector, or nothing when it is already on this computer or
/// is a local file used in place. `catalog_model` is the catalog model this version exactly is (the
/// same repository, commit and file), installed or not: adding the version installs that model.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CandidateVersion {
    pub label: String,
    pub model_id: ModelId,
    pub files: Vec<CandidateFile>,
    pub size_bytes: u64,
    pub download_bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(nullable = false))]
    pub assessment: Option<ModelAssessment>,
    pub recommended: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(nullable = false))]
    pub catalog_model: Option<ModelId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(nullable = false))]
    pub on_this_computer: Option<ModelId>,
}

/// The repository step: every version with the engine's verdict, or why none can run.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CandidateInspection {
    pub source: CandidateSource,
    pub display_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(nullable = false))]
    pub license: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(nullable = false))]
    pub variant_of: Option<ModelId>,
    pub versions: Vec<CandidateVersion>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(nullable = false))]
    pub projector: Option<CandidateFile>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(nullable = false))]
    pub unsupported_architecture: Option<Vec<UnsupportedReason>>,
}

/// What an inspection is doing, with how much of it is done: reading each file's GGUF header,
/// checking each version against this computer, then checking the drafters of the version it
/// recommends. A drafter total counts the drafter checks known so far, so it can grow.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub enum CandidateInspectionPhase {
    ReadingHeaders { completed: u32, total: u32 },
    CheckingVersions { completed: u32, total: u32 },
    CheckingDrafters { version: String, completed: u32, total: u32 },
}

/// A version as an inspection in progress knows it: its label and size, and its plain verdict
/// once checked. While headers are read, the rows are the repository's GGUFs named by their files;
/// from the version checks on, they are the versions the inspection returns.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CandidateVersionProgress {
    pub label: String,
    pub size_bytes: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(nullable = false))]
    pub assessment: Option<ModelAssessment>,
}

/// How far an inspection has got, each version's verdict included as soon as it is known.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CandidateInspectionProgress {
    pub phase: CandidateInspectionPhase,
    pub versions: Vec<CandidateVersionProgress>,
}

/// One event of an inspection's stream: progress, in order, then the inspection itself.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub enum CandidateInspectionEvent {
    Progress { progress: CandidateInspectionProgress },
    Inspected { inspection: CandidateInspection },
}

/// Where an inspection reports its progress; `none` discards it.
#[derive(Clone, Default)]
pub struct CandidateInspectionProgressSink(
    Option<std::sync::Arc<dyn Fn(CandidateInspectionProgress) + Send + Sync>>,
);

impl CandidateInspectionProgressSink {
    pub fn new(report: impl Fn(CandidateInspectionProgress) + Send + Sync + 'static) -> Self {
        Self(Some(std::sync::Arc::new(report)))
    }

    #[must_use]
    pub fn none() -> Self {
        Self(None)
    }

    #[must_use]
    pub fn is_reporting(&self) -> bool {
        self.0.is_some()
    }

    pub fn report(&self, progress: CandidateInspectionProgress) {
        if let Some(report) = &self.0 {
            report(progress);
        }
    }
}

impl fmt::Debug for CandidateInspectionProgressSink {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("CandidateInspectionProgressSink")
            .field(&self.is_reporting())
            .finish()
    }
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub enum DrafterCandidatesRequest {
    Candidate {
        candidate: ModelCandidate,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[cfg_attr(feature = "openapi", schema(nullable = false))]
        other: Option<String>,
    },
    #[serde(rename_all = "camelCase")]
    Model {
        model_id: ModelId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[cfg_attr(feature = "openapi", schema(nullable = false))]
        other: Option<String>,
    },
}

/// Drafter candidates for one target, in the product's category order. `entered` is the listed
/// reference the request's different drafter resolves to: a listed candidate that is the same file
/// (a catalog drafter by its package, otherwise the same repository, commit and path), else the
/// different drafter's own Other row; absent without a different drafter.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct DrafterCandidatesResponse {
    pub candidates: Vec<DrafterCandidate>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(nullable = false))]
    pub entered: Option<DrafterReference>,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CatalogOptimizationStage {
    Preparing,
    Tuning,
}

/// Kernel tuning after a verified download, in tuning work units. `Preparing` opens the device
/// and counts the units (0 of 0); `device` is the device a load would select, once previewed.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CatalogOptimizationProgress {
    pub stage: CatalogOptimizationStage,
    pub completed: u64,
    pub total: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(nullable = false))]
    pub device: Option<ModelLoadDevice>,
}

impl CatalogOptimizationProgress {
    #[must_use]
    pub const fn preparing(device: Option<ModelLoadDevice>) -> Self {
        Self {
            stage: CatalogOptimizationStage::Preparing,
            completed: 0,
            total: 0,
            device,
        }
    }
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
    Optimizing {
        progress: CatalogOptimizationProgress,
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

/// `kept_components`: the parts Magnitude stores for a model whose weights are found elsewhere
/// that the removal kept, because another model on this computer needs them.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub enum CatalogInstallationRemoval {
    #[serde(rename_all = "camelCase")]
    Removed {
        reclaimed_bytes: u64,
        kept_components: Vec<CatalogComponent>,
    },
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum CatalogModelSelection {
    Desired,
    Effective,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub enum ModelAssessmentSubject {
    #[serde(rename_all = "camelCase")]
    Catalog {
        model_id: ModelId,
        selection: CatalogModelSelection,
    },
    #[serde(rename_all = "camelCase")]
    Custom { model_id: ModelId },
}

impl ModelAssessmentSubject {
    #[must_use]
    pub fn model_id(&self) -> &ModelId {
        match self {
            Self::Catalog { model_id, .. } | Self::Custom { model_id } => model_id,
        }
    }
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

/// An estimated decode speed at one context depth: the model's bytes per decode step over the
/// device's memory bandwidth.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PerformanceEvidence {
    pub context_tokens: u32,
    pub estimated_tokens_per_second: f64,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase")]
pub enum ModelAssessment {
    /// A fitting model, with one decode-speed estimate per requested depth.
    #[serde(rename_all = "camelCase")]
    Fits {
        profile: ServingProfile,
        assessment_id: ModelAssessmentId,
        memory: Vec<MemoryAssessment>,
        performance: Vec<PerformanceEvidence>,
        projector: ProjectorVerdict,
        drafter: DrafterVerdict,
    },
    #[serde(rename_all = "camelCase")]
    DoesNotFit {
        profile: ServingProfile,
        assessment_id: ModelAssessmentId,
        memory: Vec<MemoryAssessment>,
        limiting_resource: String,
        deficit_bytes: u64,
        projector: ProjectorVerdict,
        drafter: DrafterVerdict,
    },
    /// The engine cannot execute a custom model, with every reason it found. A catalog model is
    /// never unsupported: the pool drops a catalog target the engine cannot execute as a release
    /// defect. An unreadable file has no serving profile.
    Unsupported {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[cfg_attr(feature = "openapi", schema(nullable = false))]
        profile: Option<ServingProfile>,
        reasons: Vec<UnsupportedReason>,
        detail: String,
    },
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase")]
pub enum ModelAssessmentEntryState {
    Assessing,
    #[serde(rename_all = "camelCase")]
    Assessed {
        capabilities: ModelCapabilities,
        template_fingerprint: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[cfg_attr(feature = "openapi", schema(nullable = false))]
        fingerprint: Option<ModelFingerprint>,
        profiles: Vec<ModelAssessment>,
    },
    Dropped,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelAssessmentEntry {
    pub subject: ModelAssessmentSubject,
    pub state: ModelAssessmentEntryState,
}

/// One source's assessments. `Pending` until the source has been read (and, for custom models,
/// reconciled).
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub enum ModelAssessmentDomainSnapshot {
    #[serde(rename_all = "camelCase")]
    Pending { source_revision: u64 },
    #[serde(rename_all = "camelCase")]
    Available {
        source_revision: u64,
        entries: Vec<ModelAssessmentEntry>,
    },
}

/// Every model assessment in the service's one environment, established at service start.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelAssessmentsSnapshot {
    pub revision: u64,
    pub environment_id: AssessmentEnvironmentId,
    pub catalog: ModelAssessmentDomainSnapshot,
    pub custom: ModelAssessmentDomainSnapshot,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelAssessmentsInvalidation {
    pub revision: u64,
}

/// A bundle to download; `for_model` names the model the download serves, such as the model a
/// drafter is fetched for, and is echoed in every `ModelDownload` it produces.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct StartModelDownloadRequest {
    pub bundle: ServableModelBundle,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(nullable = false))]
    pub for_model: Option<ModelId>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(nullable = false))]
    pub for_model: Option<ModelId>,
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
    /// The engine-resolved supported maximum context: the serving ceiling.
    pub context_window_tokens: u32,
    /// The load's startup peak claim in the selected device's allocation domain (system RAM or
    /// dedicated VRAM). It excludes future context growth, which is claimed as history grows.
    pub required_memory_bytes: u64,
    pub device: ModelLoadDevice,
}

/// The execution device a model load uses.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelLoadDevice {
    pub id: crate::HardwareDeviceId,
    pub backend: crate::ExecutionBackend,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelLoadStage {
    /// Waiting for memory admission.
    Queued,
    /// Resolving, planning, opening the device and preparing programs.
    Preparing,
    /// Tuning kernels for the device; only a load that tunes enters it.
    Optimizing,
    LoadingWeights,
    /// State allocation, warm-up and readiness verification.
    Finalizing,
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
    /// Fetches only `component` of an installed catalog model whose target is found elsewhere.
    fn install_catalog_component(
        &self,
        model_id: &ModelId,
        component: CatalogComponent,
    ) -> BoxFuture<'_, Result<CatalogInstallationAdmission, InventoryError>>;
    fn remove_catalog_model_installation(
        &self,
        model_id: &ModelId,
    ) -> BoxFuture<'_, Result<CatalogInstallationRemoval, InventoryError>>;
    fn watch_catalog(&self) -> BoxStream<'static, ModelDomainInvalidation>;
}

/// Custom models: the configuration ACN hands over, the models it yields, and rescans.
pub trait LocalModels: Send + Sync + 'static {
    fn configuration(&self) -> BoxFuture<'_, Result<LocalModelsConfiguration, InventoryError>>;
    fn configure(
        &self,
        configuration: LocalModelsConfiguration,
    ) -> BoxFuture<'_, Result<LocalModelsConfiguration, InventoryError>>;
    fn list_local(&self) -> BoxFuture<'_, Result<LocalModelsResponse, InventoryError>>;
    fn refresh_local(&self) -> BoxFuture<'_, Result<LocalModelsResponse, InventoryError>>;
    fn remove_local(
        &self,
        model_id: &ModelId,
    ) -> BoxFuture<'_, Result<LocalModelRemoval, InventoryError>>;
    /// Removes the Magnitude-owned drafter packages that neither an installed catalog model nor
    /// any custom model uses. Drafters in other locations are never touched.
    fn prune_drafters(&self) -> BoxFuture<'_, Result<DrafterPruning, InventoryError>>;
    fn watch_local(&self) -> BoxStream<'static, ModelDomainInvalidation>;
}

/// Files not yet added: inspection, drafter enumeration and assessment, all header-only.
pub trait ModelCandidates: Send + Sync + 'static {
    /// Inspects a candidate, reporting its progress to `progress` as it goes.
    fn inspect_with_progress(
        &self,
        request: CandidateInspectionRequest,
        progress: CandidateInspectionProgressSink,
    ) -> BoxFuture<'_, Result<CandidateInspection, InventoryError>>;
    fn inspect(
        &self,
        request: CandidateInspectionRequest,
    ) -> BoxFuture<'_, Result<CandidateInspection, InventoryError>> {
        self.inspect_with_progress(request, CandidateInspectionProgressSink::none())
    }
    fn drafters(
        &self,
        request: DrafterCandidatesRequest,
    ) -> BoxFuture<'_, Result<DrafterCandidatesResponse, InventoryError>>;
    fn assess_candidate(
        &self,
        candidate: ModelCandidate,
    ) -> BoxFuture<'_, Result<ModelAssessment, InventoryError>>;
    fn bundle(
        &self,
        candidate: ModelCandidate,
    ) -> BoxFuture<'_, Result<ServableModelBundle, InventoryError>>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelDomainInvalidation {
    pub revision: u64,
}

pub trait CatalogInstallations: Send + Sync + 'static {
    fn watch_catalog_installations(&self) -> BoxStream<'static, CatalogInstallationsInvalidation>;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CatalogInstallationsInvalidation {
    pub revision: u64,
}

/// Tunes a freshly installed catalog model's kernels for the device a load would select, so its
/// first load does not. Reports progress until it finishes; dropping the future stops the job.
/// A failure is the optimizer's to log: the model is installed either way.
pub trait CatalogModelOptimizer: Send + Sync + 'static {
    fn optimize_catalog_model(
        &self,
        model_id: ModelId,
        progress: Box<dyn Fn(CatalogOptimizationProgress) + Send + Sync>,
    ) -> BoxFuture<'static, ()>;
}

pub trait CatalogPackageRemover: Send + Sync + 'static {
    fn remove_catalog_packages(
        &self,
        package_ids: Vec<ModelPackageId>,
    ) -> BoxFuture<'_, Result<u64, InventoryError>>;
}

pub trait ModelAssessments: Send + Sync + 'static {
    fn snapshot(&self) -> BoxFuture<'_, Result<ModelAssessmentsSnapshot, InventoryError>>;
    fn watch(&self) -> BoxStream<'static, ModelAssessmentsInvalidation>;
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

#[cfg(test)]
mod custom_model_wire_tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn unsupported_reasons_are_pascal_tagged_with_camel_case_fields() {
        let reasons = vec![
            UnsupportedReason::Known(KnownUnsupportedReason::Architecture {
                architecture: "deepseek4".to_owned(),
            }),
            UnsupportedReason::Known(KnownUnsupportedReason::WeightTypes {
                types: vec![WeightTypeCount {
                    ggml_type: "Q2_K".to_owned(),
                    tensor_count: 2,
                }],
            }),
            UnsupportedReason::Known(KnownUnsupportedReason::Tokenizer {
                model: "gpt2".to_owned(),
                pre: None,
            }),
            UnsupportedReason::Known(KnownUnsupportedReason::KernelDomain {
                backend: "metal".to_owned(),
                entry: "matmul".to_owned(),
                statics: vec![KernelStatic {
                    name: "head_dim".to_owned(),
                    value: 80,
                }],
            }),
            UnsupportedReason::Known(KnownUnsupportedReason::MissingMetadata {
                key: "dflash.attention.causal".to_owned(),
            }),
            UnsupportedReason::Known(KnownUnsupportedReason::TargetMismatch {
                detail: "hidden size".to_owned(),
            }),
            UnsupportedReason::Known(KnownUnsupportedReason::TiedOutputHead),
            UnsupportedReason::Known(KnownUnsupportedReason::DraftArchitecture {
                architecture: "nemotron_h_moe".to_owned(),
            }),
            UnsupportedReason::Known(KnownUnsupportedReason::Unreadable),
            UnsupportedReason::Known(KnownUnsupportedReason::UnsupportedFormat),
        ];
        let encoded = serde_json::to_value(&reasons).expect("serialize");
        assert_eq!(
            encoded,
            json!([
                { "_tag": "Architecture", "architecture": "deepseek4" },
                { "_tag": "WeightTypes", "types": [{ "ggmlType": "Q2_K", "tensorCount": 2 }] },
                { "_tag": "Tokenizer", "model": "gpt2" },
                {
                    "_tag": "KernelDomain",
                    "backend": "metal",
                    "entry": "matmul",
                    "statics": [{ "name": "head_dim", "value": 80 }]
                },
                { "_tag": "MissingMetadata", "key": "dflash.attention.causal" },
                { "_tag": "TargetMismatch", "detail": "hidden size" },
                { "_tag": "TiedOutputHead" },
                { "_tag": "DraftArchitecture", "architecture": "nemotron_h_moe" },
                { "_tag": "Unreadable" },
                { "_tag": "UnsupportedFormat" }
            ])
        );
        assert_eq!(
            serde_json::from_value::<Vec<UnsupportedReason>>(encoded).expect("deserialize"),
            reasons
        );
    }

    #[test]
    fn unsupported_reason_tags_from_a_newer_engine_stay_readable() {
        let reason: UnsupportedReason = serde_json::from_value(json!({
            "_tag": "SlidingWindowPattern",
            "detail": "interleaved sliding window",
            "pattern": [1, 0, 1]
        }))
        .expect("deserialize");
        assert_eq!(
            reason,
            UnsupportedReason::Unrecognized(UnrecognizedUnsupportedReason {
                tag: "SlidingWindowPattern".to_owned(),
                detail: Some("interleaved sliding window".to_owned()),
            })
        );
        assert_eq!(
            serde_json::to_value(&reason).expect("serialize"),
            json!({ "_tag": "SlidingWindowPattern", "detail": "interleaved sliding window" })
        );
        assert_eq!(
            serde_json::from_value::<UnsupportedReason>(json!({ "_tag": "Unreadable" }))
                .expect("deserialize"),
            UnsupportedReason::Known(KnownUnsupportedReason::Unreadable)
        );
    }

    #[test]
    fn unsupported_assessment_carries_reasons_and_detail() {
        let assessment = ModelAssessment::Unsupported {
            profile: Some(ServingProfile {
                context_length: 4096,
            }),
            reasons: vec![UnsupportedReason::Known(
                KnownUnsupportedReason::UnrecognizedTensor {
                    name: "rope_freqs.weight".to_owned(),
                },
            )],
            detail: "unrecognized tensor".to_owned(),
        };
        assert_eq!(
            serde_json::to_value(&assessment).expect("serialize"),
            json!({
                "_tag": "Unsupported",
                "profile": { "contextLength": 4096 },
                "reasons": [{ "_tag": "UnrecognizedTensor", "name": "rope_freqs.weight" }],
                "detail": "unrecognized tensor"
            })
        );
    }

    #[test]
    fn unreadable_assessment_serializes_without_a_profile() {
        let assessment = ModelAssessment::Unsupported {
            profile: None,
            reasons: vec![UnsupportedReason::Known(KnownUnsupportedReason::Unreadable)],
            detail: "truncated".to_owned(),
        };
        let encoded = json!({
            "_tag": "Unsupported",
            "reasons": [{ "_tag": "Unreadable" }],
            "detail": "truncated"
        });
        assert_eq!(
            serde_json::to_value(&assessment).expect("serialize"),
            encoded
        );
        assert_eq!(
            serde_json::from_value::<ModelAssessment>(encoded).expect("deserialize"),
            assessment
        );
    }

    #[test]
    fn local_models_configuration_round_trips() {
        let digest = "ab".repeat(32);
        let configuration: LocalModelsConfiguration = serde_json::from_value(json!({
            "roots": [
                { "kind": "LmStudio", "path": "/home/u/.lmstudio/models", "enabled": true },
                { "kind": "Folder", "path": "/models", "enabled": false }
            ],
            "declarations": [
                {
                    "modelId": "hf:owner/repo/model-Q4_K_M.gguf",
                    "origin": { "_tag": "Added", "repository": "owner/repo", "commit": "c".repeat(40) },
                    "target": ["model-Q4_K_M.gguf"],
                    "projector": "mmproj.gguf"
                },
                {
                    "modelId": format!("local:{digest}"),
                    "origin": { "_tag": "Imported", "path": "/models/a.gguf" },
                    "target": ["/models/a.gguf"]
                }
            ],
            "hidden": [format!("local:{digest}")],
            "drafters": [
                { "modelId": "hf:owner/repo/model-Q4_K_M.gguf", "drafter": { "_tag": "Auto" } },
                {
                    "modelId": format!("local:{digest}"),
                    "drafter": {
                        "_tag": "Reference",
                        "reference": { "_tag": "Catalog", "modelId": "qwen3.6-35b-a3b:gguf:q4" },
                        "method": { "_tag": "DFlash" }
                    }
                }
            ]
        }))
        .expect("deserialize configuration");
        assert_eq!(configuration.roots[0].kind, ModelRootKind::LmStudio);
        assert!(matches!(
            configuration.declarations[1].origin,
            CustomModelDeclarationOrigin::Imported { .. }
        ));
        let encoded = serde_json::to_value(&configuration).expect("serialize");
        assert_eq!(
            serde_json::from_value::<LocalModelsConfiguration>(encoded).expect("round trip"),
            configuration
        );
    }

    #[test]
    fn custom_model_origins_name_their_found_source() {
        let origin = CustomModelOrigin::Found {
            source: FoundSource::Folder {
                path: "/models".into(),
            },
        };
        assert_eq!(
            serde_json::to_value(&origin).expect("serialize"),
            json!({ "_tag": "Found", "source": { "_tag": "Folder", "path": "/models" } })
        );
        assert_eq!(
            serde_json::to_value(CustomModelOrigin::Found {
                source: FoundSource::Ollama
            })
            .expect("serialize"),
            json!({ "_tag": "Found", "source": { "_tag": "Ollama" } })
        );
    }

    #[test]
    fn drafter_verdicts_and_subjects_use_the_custom_vocabulary() {
        assert_eq!(
            serde_json::to_value(DrafterVerdict::Bound {
                method: SpeculativeMethod::Mtp,
                memory_bytes: 10
            })
            .expect("serialize"),
            json!({ "_tag": "Bound", "method": { "_tag": "Mtp" }, "memoryBytes": 10 })
        );
        assert_eq!(
            serde_json::to_value(ProjectorVerdict::Rejected {
                reasons: vec![UnsupportedReason::Known(KnownUnsupportedReason::TargetMismatch {
                    detail: "no vision projector".to_owned()
                })],
                detail: "projector: no vision projector".to_owned()
            })
            .expect("serialize"),
            json!({
                "_tag": "Rejected",
                "reasons": [{ "_tag": "TargetMismatch", "detail": "no vision projector" }],
                "detail": "projector: no vision projector"
            })
        );
        let subject = ModelAssessmentSubject::Custom {
            model_id: format!("local:{}", "0".repeat(64))
                .parse()
                .expect("model ID"),
        };
        assert_eq!(
            serde_json::to_value(&subject).expect("serialize")["_tag"],
            "Custom"
        );
    }
}
