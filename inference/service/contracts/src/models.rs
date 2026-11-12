//! Transport-neutral local-model package, assessment, download, and residency contracts.

use std::fmt;
use std::num::NonZeroU64;
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
string_id!(LabId);

/// A model's identity. Two `local:` IDs are the same model exactly when their hashes match: the
/// label is for reading only, so equality, order and hashing ignore it.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema), schema(value_type = String))]
#[derive(Debug, Clone)]
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

/// The identity of a custom model with no Hugging Face identity: the first
/// [`LocalModelHash::LENGTH`] lowercase hex characters of its content digest.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema), schema(value_type = String))]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LocalModelHash(String);

/// The readable part of a `local:` ID (`<name>-<quant>`), which never identifies the model.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema), schema(value_type = String))]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LocalModelLabel(String);

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

impl LocalModelHash {
    /// 48 bits: about one chance in 560 million of two models clashing among 1,000 on a computer.
    /// Fixed, so an ID never depends on other files.
    pub const LENGTH: usize = 12;

    pub fn new(value: impl Into<String>) -> Result<Self, ModelIdError> {
        let value = value.into();
        if value.len() != Self::LENGTH
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(model_id_error(format!(
                "local model hash must be {} lowercase hex characters",
                Self::LENGTH
            )));
        }
        Ok(Self(value))
    }

    /// The hash of a model's lowercase hex content digest.
    pub fn of_digest(digest: &str) -> Result<Self, ModelIdError> {
        Self::new(digest.get(..Self::LENGTH).unwrap_or(digest))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for LocalModelHash {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl LocalModelLabel {
    pub fn new(value: impl Into<String>) -> Result<Self, ModelIdError> {
        let value = value.into();
        if value.is_empty()
            || !value.bytes().all(|byte| {
                byte.is_ascii_lowercase() || byte.is_ascii_digit() || b"._-".contains(&byte)
            })
        {
            return Err(model_id_error(
                "local model label must be lowercase ASCII letters, digits, '.', '_' and '-'",
            ));
        }
        Ok(Self(value))
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for LocalModelLabel {
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
        label: LocalModelLabel,
        hash: LocalModelHash,
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
    pub fn local(label: &LocalModelLabel, hash: &LocalModelHash) -> Self {
        Self(format!("local:{}:{}", label.0, hash.0))
    }

    /// What names the model: a `local:` ID's hash, or any other ID whole.
    fn identity(&self) -> (bool, &str) {
        match self.0.strip_prefix("local:").and_then(|rest| rest.rsplit_once(':')) {
            Some((_, hash)) => (true, hash),
            None => (false, &self.0),
        }
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

    if let Some(rest) = value.strip_prefix("local:") {
        let (label, hash) = rest
            .rsplit_once(':')
            .ok_or_else(|| model_id_error("local model ID must be local:<label>:<hash>"))?;
        return Ok(ParsedModelId::Local {
            label: LocalModelLabel::new(label)?,
            hash: LocalModelHash::new(hash)?,
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

impl PartialEq for ModelId {
    fn eq(&self, other: &Self) -> bool {
        self.identity() == other.identity()
    }
}

impl Eq for ModelId {}

impl PartialOrd for ModelId {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ModelId {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.identity().cmp(&other.identity())
    }
}

impl std::hash::Hash for ModelId {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.identity().hash(state);
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
impl_validated_string!(LocalModelHash, LocalModelHash::new);
impl_validated_string!(LocalModelLabel, LocalModelLabel::new);

#[cfg(test)]
mod model_id_tests {
    use std::str::FromStr;

    use super::{
        CatalogBaseId, CatalogVariantId, LocalModelHash, LocalModelLabel, ModelId, ParsedModelId,
    };

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
        let label = LocalModelLabel::new("gemma-4-12b-instruct-q8_0").expect("label");
        let hash = LocalModelHash::of_digest(&"41ad09c2e7f3".repeat(6)).expect("hash");
        let id = ModelId::local(&label, &hash);
        assert_eq!(id.as_str(), "local:gemma-4-12b-instruct-q8_0:41ad09c2e7f3");
        assert_eq!(id.parsed(), ParsedModelId::Local { label, hash });
        let encoded = serde_json::to_string(&id).expect("serialize");
        assert_eq!(
            serde_json::from_str::<ModelId>(&encoded).expect("deserialize"),
            id
        );
    }

    #[test]
    fn local_identities_with_the_same_hash_are_the_same_model_whatever_their_labels() {
        let old = ModelId::from_str("local:my-finetune:9f2c1a7b3e04").expect("old");
        let current = ModelId::from_str("local:my-finetune-q4_k_m:9f2c1a7b3e04").expect("current");
        let other = ModelId::from_str("local:my-finetune-q4_k_m:9f2c1a7b3e05").expect("other");
        assert_eq!(old, current);
        assert_ne!(current, other);
        assert_eq!(
            std::collections::BTreeSet::from([old.clone(), current.clone()]).len(),
            1
        );
        assert_eq!(old.as_str(), "local:my-finetune:9f2c1a7b3e04");
    }

    #[test]
    fn rejects_non_canonical_local_identities() {
        for value in [
            "local:",
            "local:41ad09c2e7f3",
            "local::41ad09c2e7f3",
            "local:model:41AD09C2E7F3",
            "local:model:41ad09c2e7f",
            "local:model:41ad09c2e7f3a",
            "local:model:41ad09c2e7g3",
            "local:Model:41ad09c2e7f3",
            "local:my model:41ad09c2e7f3",
        ] {
            assert!(ModelId::from_str(value).is_err(), "accepted {value}");
        }
        // Only the lowercase prefix makes a local ID; this is a catalog ID's shape.
        assert!(matches!(
            ModelId::from_str("LOCAL:model:41ad09c2e7f3").unwrap().parsed(),
            ParsedModelId::Catalog { .. }
        ));
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
    /// The places granted to the load's prompt groups at the last observation.
    pub prompt_group_places: PromptGroupPlaces,
    /// Where an elastic expert bank's experts are; absent for a load without one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(nullable = false))]
    pub streaming: Option<ModelStreamingStatus>,
}

/// A contract bound a value breaks: the rule, in words.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutOfBounds(pub &'static str);

impl fmt::Display for OutOfBounds {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "out of the contract's bounds: {}", self.0)
    }
}

impl std::error::Error for OutOfBounds {}

impl OutOfBounds {
    /// `rule` holds, or the bound `statement` states is broken.
    pub fn check(rule: bool, statement: &'static str) -> Result<(), Self> {
        rule.then_some(()).ok_or(Self(statement))
    }
}

/// The places a prompt group holds now, of the most it may take: each place runs one chunk of
/// a prompt in the group's pass. It holds at most its capacity.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields, try_from = "PromptGroupPlacesParts")]
pub struct PromptGroupPlaces {
    #[cfg_attr(feature = "openapi", schema(value_type = u64, minimum = 1))]
    held: NonZeroU64,
    #[cfg_attr(feature = "openapi", schema(value_type = u64, minimum = 1))]
    capacity: NonZeroU64,
}

/// [`PromptGroupPlaces`]' fields, before their bound is checked.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PromptGroupPlacesParts {
    pub held: NonZeroU64,
    pub capacity: NonZeroU64,
}

impl TryFrom<PromptGroupPlacesParts> for PromptGroupPlaces {
    type Error = OutOfBounds;

    fn try_from(parts: PromptGroupPlacesParts) -> Result<Self, OutOfBounds> {
        let PromptGroupPlacesParts { held, capacity } = parts;
        OutOfBounds::check(held <= capacity, "a prompt group holds at most its capacity")?;
        Ok(Self { held, capacity })
    }
}

/// Where an elastic expert bank's routed experts are at the last observation: the experts of
/// each routed layer it holds (its slots), between the floor it never shrinks below and the
/// ceiling it never grows past, at most every expert; its bytes in each memory domain; the
/// host-RAM tier holding cold experts beside it; the bytes of the experts in neither, read
/// from the model file on demand; the stored tables it holds; and its experts' last resize.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields, try_from = "ModelStreamingStatusParts")]
pub struct ModelStreamingStatus {
    /// Each routed layer's experts.
    #[cfg_attr(feature = "openapi", schema(value_type = u64, minimum = 1))]
    experts: NonZeroU64,
    slots: u64,
    floor_slots: u64,
    ceiling_slots: u64,
    /// The memory domains holding the bank, device memory first.
    tiers: Vec<StreamingTierStatus>,
    /// The host-RAM tier a discrete device's bank keeps its cold experts in,
    /// exclusive of the bank; absent on unified memory.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(nullable = false))]
    host: Option<StreamingHostStatus>,
    /// The experts in neither the bank nor the host tier, as the model file
    /// stores them.
    disk_bytes: u64,
    /// The stored tables (n-gram tables) the bank holds beside its experts.
    tables: Vec<StreamingTableStatus>,
    /// The experts' last resize; absent before their first.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(nullable = false))]
    last_resize: Option<StreamingResize>,
}

/// [`ModelStreamingStatus`]' fields, before their bounds are checked.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelStreamingStatusParts {
    pub experts: NonZeroU64,
    pub slots: u64,
    pub floor_slots: u64,
    pub ceiling_slots: u64,
    pub tiers: Vec<StreamingTierStatus>,
    #[serde(default)]
    pub host: Option<StreamingHostStatus>,
    pub disk_bytes: u64,
    pub tables: Vec<StreamingTableStatus>,
    #[serde(default)]
    pub last_resize: Option<StreamingResize>,
}

impl TryFrom<ModelStreamingStatusParts> for ModelStreamingStatus {
    type Error = OutOfBounds;

    fn try_from(parts: ModelStreamingStatusParts) -> Result<Self, OutOfBounds> {
        let ModelStreamingStatusParts {
            experts,
            slots,
            floor_slots,
            ceiling_slots,
            tiers,
            host,
            disk_bytes,
            tables,
            last_resize,
        } = parts;
        OutOfBounds::check(
            floor_slots <= slots && slots <= ceiling_slots && ceiling_slots <= experts.get(),
            "an expert bank holds between its floor and its ceiling, at most every expert",
        )?;
        Ok(Self { experts, slots, floor_slots, ceiling_slots, tiers, host, disk_bytes, tables, last_resize })
    }
}

/// A stored table an expert bank holds: the records its slots use now, between the floor and
/// ceiling it resizes within; the bytes they hold; and its last resize.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields, try_from = "StreamingTableStatusParts")]
pub struct StreamingTableStatus {
    records: u64,
    floor_records: u64,
    #[cfg_attr(feature = "openapi", schema(value_type = u64, minimum = 1))]
    ceiling_records: NonZeroU64,
    bytes: u64,
    /// Absent before the table's first resize.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(nullable = false))]
    last_resize: Option<StreamingResize>,
}

/// [`StreamingTableStatus`]' fields, before their bounds are checked.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct StreamingTableStatusParts {
    pub records: u64,
    pub floor_records: u64,
    pub ceiling_records: NonZeroU64,
    pub bytes: u64,
    #[serde(default)]
    pub last_resize: Option<StreamingResize>,
}

impl TryFrom<StreamingTableStatusParts> for StreamingTableStatus {
    type Error = OutOfBounds;

    fn try_from(parts: StreamingTableStatusParts) -> Result<Self, OutOfBounds> {
        let StreamingTableStatusParts { records, floor_records, ceiling_records, bytes, last_resize } = parts;
        OutOfBounds::check(
            floor_records <= records && records <= ceiling_records.get(),
            "a table holds between its floor and its ceiling",
        )?;
        Ok(Self { records, floor_records, ceiling_records, bytes, last_resize })
    }
}

/// A discrete device's host-RAM tier, in its memory domain: its host slots now (each an expert
/// of every streamed layer) up to its ceiling, the host memory its records take, and its last
/// resize.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields, try_from = "StreamingHostStatusParts")]
pub struct StreamingHostStatus {
    memory_domain_id: crate::MemoryDomainId,
    slots: u64,
    /// At most every expert; none beside the landing reserve where a
    /// weight cap leaves the tier no slots.
    ceiling_slots: u64,
    bytes: u64,
    /// Absent before the tier's first resize.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(nullable = false))]
    last_resize: Option<StreamingResize>,
}

/// [`StreamingHostStatus`]' fields, before their bound is checked.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct StreamingHostStatusParts {
    pub memory_domain_id: crate::MemoryDomainId,
    pub slots: u64,
    pub ceiling_slots: u64,
    pub bytes: u64,
    #[serde(default)]
    pub last_resize: Option<StreamingResize>,
}

impl TryFrom<StreamingHostStatusParts> for StreamingHostStatus {
    type Error = OutOfBounds;

    fn try_from(parts: StreamingHostStatusParts) -> Result<Self, OutOfBounds> {
        let StreamingHostStatusParts { memory_domain_id, slots, ceiling_slots, bytes, last_resize } = parts;
        OutOfBounds::check(slots <= ceiling_slots, "a host tier holds at most its ceiling")?;
        Ok(Self { memory_domain_id, slots, ceiling_slots, bytes, last_resize })
    }
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct StreamingTierStatus {
    pub memory_domain_id: crate::MemoryDomainId,
    pub bytes: u64,
}

/// A holding's finished resize, in its units (slots, records or host slots), at `atUnixMs`
/// (milliseconds since the Unix epoch): shrunk by `by` to `to` as memory grew short, or grown
/// by `by` from `from` into free memory. Each changes its size by at least one.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub enum StreamingResize {
    #[serde(rename_all = "camelCase")]
    Shrank {
        to: u64,
        #[cfg_attr(feature = "openapi", schema(value_type = u64, minimum = 1))]
        by: NonZeroU64,
        at_unix_ms: u64,
    },
    #[serde(rename_all = "camelCase")]
    Grew {
        from: u64,
        #[cfg_attr(feature = "openapi", schema(value_type = u64, minimum = 1))]
        by: NonZeroU64,
        at_unix_ms: u64,
    },
}

#[cfg(test)]
mod model_streaming_status_tests {
    use super::*;

    fn allocation(streaming: Option<ModelStreamingStatus>) -> ModelInstanceAllocation {
        ModelInstanceAllocation {
            context_window_tokens: 4_096,
            memory_domains: Vec::new(),
            prompt_group_places: PromptGroupPlacesParts {
                held: NonZeroU64::MIN,
                capacity: NonZeroU64::new(8).unwrap(),
            }
            .try_into()
            .unwrap(),
            streaming,
        }
    }

    #[test]
    fn an_allocation_without_a_bank_omits_streaming() {
        let json = serde_json::to_value(allocation(None)).expect("serialize");
        assert_eq!(
            json,
            serde_json::json!({
                "contextWindowTokens": 4096,
                "memoryDomains": [],
                "promptGroupPlaces": { "held": 1, "capacity": 8 }
            })
        );
        let decoded: ModelInstanceAllocation =
            serde_json::from_value(json).expect("deserialize resident allocation");
        assert_eq!(decoded, allocation(None));
    }

    #[test]
    fn a_streamed_allocation_round_trips_in_its_wire_shape() {
        let json = serde_json::json!({
            "contextWindowTokens": 4096,
            "memoryDomains": [],
            "promptGroupPlaces": { "held": 3, "capacity": 8 },
            "streaming": {
                "experts": 256,
                "slots": 64,
                "floorSlots": 26,
                "ceilingSlots": 256,
                "tiers": [{ "memoryDomainId": "cuda-0", "bytes": 300 }, { "memoryDomainId": "system", "bytes": 400 }],
                "host": {
                    "memoryDomainId": "system",
                    "slots": 20,
                    "ceilingSlots": 256,
                    "bytes": 650,
                    "lastResize": { "_tag": "Grew", "from": 19, "by": 1, "atUnixMs": 1200 }
                },
                "diskBytes": 300,
                "tables": [{
                    "records": 40, "floorRecords": 1, "ceilingRecords": 42, "bytes": 500,
                    "lastResize": { "_tag": "Shrank", "to": 40, "by": 1, "atUnixMs": 1234 }
                }],
                "lastResize": { "_tag": "Shrank", "to": 64, "by": 8, "atUnixMs": 1100 }
            }
        });
        let decoded: ModelInstanceAllocation =
            serde_json::from_value(json.clone()).expect("deserialize streamed allocation");
        assert_eq!(serde_json::to_value(&decoded).expect("serialize"), json);
    }

    /// No value outside a bound decodes: each bounded quantity is refused out of order, a
    /// resize that changes nothing is refused, and no resize names a cause.
    #[test]
    fn values_outside_the_bounds_never_decode() {
        let status = |slots: u64, ceiling: u64| {
            serde_json::json!({
                "experts": 256, "slots": slots, "floorSlots": 26, "ceilingSlots": ceiling,
                "tiers": [], "diskBytes": 0, "tables": []
            })
        };
        assert!(serde_json::from_value::<ModelStreamingStatus>(status(64, 200)).is_ok());
        for (slots, ceiling) in [(20, 200), (201, 200), (64, 257)] {
            assert!(serde_json::from_value::<ModelStreamingStatus>(status(slots, ceiling)).is_err());
        }
        assert!(serde_json::from_value::<PromptGroupPlaces>(serde_json::json!({ "held": 9, "capacity": 8 })).is_err());
        assert!(serde_json::from_value::<StreamingTableStatus>(serde_json::json!({
            "records": 43, "floorRecords": 1, "ceilingRecords": 42, "bytes": 0
        }))
        .is_err());
        assert!(serde_json::from_value::<StreamingHostStatus>(serde_json::json!({
            "memoryDomainId": "system", "slots": 3, "ceilingSlots": 2, "bytes": 0
        }))
        .is_err());
        for resize in [
            serde_json::json!({ "_tag": "Grew", "from": 4, "by": 0, "atUnixMs": 1 }),
            serde_json::json!({ "_tag": "Grew", "from": 4, "by": 1, "cause": "reclaim", "atUnixMs": 1 }),
            serde_json::json!({ "_tag": "Shrank", "cause": "reclaim", "to": 4, "by": 1, "atUnixMs": 1 }),
        ] {
            assert!(serde_json::from_value::<StreamingResize>(resize).is_err());
        }
    }
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
        /// The device the worker opened and serves on.
        device: ModelLoadDevice,
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
    LiLiCorr,
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

/// What the release promises for a catalog model, with its `intelligence`: its Artificial
/// Analysis score as a percentage of the frontier's, which only a disabled model may lack. It is
/// independent of assessment, which says what one device can do with it.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "level", rename_all = "camelCase", deny_unknown_fields)]
pub enum CatalogSupport {
    /// Intended to be available as a supported catalog model.
    Supported { intelligence: u32 },
    /// Temporarily unavailable or not yet supported. Never installed or loaded.
    Disabled {
        reason: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        #[cfg_attr(feature = "openapi", schema(nullable = false))]
        intelligence: Option<u32>,
    },
    /// Never offered for new installs and never loaded. An installed copy stays identifiable and
    /// removable, and points at its replacement.
    #[serde(rename_all = "camelCase")]
    Deprecated {
        since: String,
        replacement: ModelId,
        reason: String,
        intelligence: u32,
    },
}

impl CatalogSupport {
    #[must_use]
    pub fn is_deprecated(&self) -> bool {
        matches!(self, Self::Deprecated { .. })
    }

    #[must_use]
    pub fn is_available(&self) -> bool {
        matches!(self, Self::Supported { .. })
    }
}

/// The lab that develops a model, from the catalog's lab table. `country` is its ISO 3166-1
/// alpha-2 code.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelDeveloper {
    pub lab: LabId,
    pub name: String,
    pub country: String,
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
    pub developer: ModelDeveloper,
    pub support: CatalogSupport,
    pub quant: CatalogQuant,
    pub quantization_aware: bool,
}

/// The quant a catalog model is: the repository that publishes it and its file quant name (for
/// example `UD-Q4_K_XL`), which states its bit width.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CatalogQuant {
    pub repository: String,
    pub name: String,
}

/// A catalog model's separately packaged drafter: its method, the repository that publishes it
/// and the bytes it adds to the download.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CatalogDrafter {
    pub method: SpeculativeMethod,
    pub repository: String,
    pub size_bytes: u64,
}

/// Whether Magnitude runs a model family now or will.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelFamilyStatus {
    Supported,
    ComingSoon,
}

/// A family of models Magnitude runs, by the GGUF architectures its models declare; fine-tunes and
/// re-uploads of these architectures run too.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelFamily {
    pub name: String,
    pub developer: ModelDeveloper,
    pub architectures: Vec<String>,
    pub status: ModelFamilyStatus,
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

/// A catalog model's state here. `stored_components`: the parts
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
    },
}

/// A catalog variant and its state on this computer. `desired`: the model installing it fetches
/// and serves here. `drafter_withheld`: it fits this computer only without its catalog drafter (or
/// not at all), so it is fetched, served and assessed without it, installed or not: it loads when
/// it fits without it, and otherwise is refused with its own shortfall. `aliases` are the `hf:` IDs
/// its target's files found in an external location (a Hugging Face cache) would have as custom
/// models, which clients resolve to this model, so references made before the file was attributed
/// keep working.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CatalogModel {
    pub id: ModelId,
    pub desired: ReadyModel,
    pub drafter_withheld: bool,
    pub display_name: String,
    pub variant_label: String,
    pub description: String,
    pub release_date: ModelReleaseDate,
    pub license: String,
    pub source_urls: Vec<String>,
    pub parameterization: ModelParameterization,
    pub developer: ModelDeveloper,
    pub support: CatalogSupport,
    pub quant: CatalogQuant,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(nullable = false))]
    pub drafter: Option<CatalogDrafter>,
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
    pub families: Vec<ModelFamily>,
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
    /// How the engine binds it; a drafter it rejects has no method.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(nullable = false))]
    pub method: Option<SpeculativeMethod>,
    pub state: AttachedDrafterState,
    pub update_available: bool,
}

/// The drafter a custom model should use: the recommended one, none, or an exact reference, whose
/// method the engine works out from the drafter and the model.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub enum DrafterChoice {
    Auto,
    None,
    Reference { reference: DrafterReference },
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(nullable = false))]
    pub developer: Option<ModelDeveloper>,
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

/// A model added from Hugging Face: its repository and the path of its first shard, which its
/// `hf:` ID names (the other shards of a split model follow from that shard's name), the pinned
/// commit, and a projector path in the repository at it.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AddedModel {
    pub repository: HuggingFaceRepositoryId,
    pub path: HuggingFaceArtifactSelector,
    pub revision: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(nullable = false))]
    pub projector: Option<String>,
}

impl AddedModel {
    #[must_use]
    pub fn model_id(&self) -> ModelId {
        ModelId::hugging_face(&self.repository, &self.path)
    }
}

/// The whole local-models configuration ACN owns and hands to ICN, at spawn and after every
/// change. Applying it is idempotent.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct LocalModelsConfiguration {
    pub roots: Vec<ModelRoot>,
    pub added: Vec<AddedModel>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(nullable = false))]
    pub developer: Option<ModelDeveloper>,
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
    Waiting,
    Preparing,
    Tuning,
}

/// Kernel tuning after a verified download, in tuning work units. `Waiting` has not finished and
/// waits for the device, which a resident or loading model holds (0 of 0); `Preparing` opens the
/// device and counts the units (0 of 0); `device` is the device a load would select, once
/// previewed. A load of the model whose own Prepare tunes reports that Prepare's progress.
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
    pub const fn waiting(device: Option<ModelLoadDevice>) -> Self {
        Self {
            stage: CatalogOptimizationStage::Waiting,
            completed: 0,
            total: 0,
            device,
        }
    }

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

/// What a fitting model's required memory holds across every domain: its weights, the
/// conversation's state at the served context, and everything else the load holds.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct MemoryBreakdown {
    pub weights_bytes: u64,
    pub context_bytes: u64,
    pub overhead_bytes: u64,
}

/// Where a fitting model's weights are held as it runs, as its files store them, so the four sum
/// to its weights: `device_bytes` in the serving device's memory (GPU memory, or unified memory on
/// Apple silicon); `host_bytes` in host RAM beside a discrete device; `disk_bytes` the routed
/// experts neither holds, streamed from the model file as steps miss them; `lookup_disk_bytes` the
/// rows of tables read by row (an embedding) that host RAM has no room to cache beside the model,
/// which each token looks up a few rows of on disk. What it holds beside its weights is its memory
/// breakdown's.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelPlacement {
    pub device_bytes: u64,
    pub host_bytes: u64,
    pub disk_bytes: u64,
    pub lookup_disk_bytes: u64,
}

/// An estimated decode speed at one context depth, in tokens per second per conversation, from
/// the costs measured of this machine's selected executor: what the engine's own kernels achieve
/// on it.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PerformanceEvidence {
    pub context_tokens: u32,
    pub estimated_tokens_per_second: f64,
}

/// A fitting model's speed estimates from this machine's measurements.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub enum ModelPerformance {
    /// This machine's measurements do not exist yet: the measurement job has not stored them.
    Pending,
    /// A measurement the estimate needs could not be taken on this machine.
    Unavailable { failure: ModelFailure },
    /// One decode-speed estimate per requested depth, ascending.
    Estimated { samples: Vec<PerformanceEvidence> },
}

/// How a fitting model holds its weights, and its decode speed held that way.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub enum ModelResidency {
    /// Every weight in memory.
    Resident { performance: ModelPerformance },
    /// The routed experts streamed through an expert bank (`bank`). `performance` is at the
    /// bank's slots; `floor_performance` is at its floor.
    #[serde(rename_all = "camelCase")]
    Streamed {
        bank: StreamedBank,
        performance: ModelPerformance,
        floor_performance: ModelPerformance,
    },
}

/// A streamed fit's expert bank: of each routed layer's `experts`, at least `floorSlots` in
/// memory, growing to `slots` where this machine's memory allows, fewer than every expert; and
/// where the experts it lacks are kept (`coldExperts`).
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields, try_from = "StreamedBankParts")]
pub struct StreamedBank {
    #[cfg_attr(feature = "openapi", schema(value_type = u64, minimum = 1))]
    experts: NonZeroU64,
    slots: u64,
    floor_slots: u64,
    cold_experts: ColdExperts,
}

/// [`StreamedBank`]'s fields, before their bounds are checked.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct StreamedBankParts {
    pub experts: NonZeroU64,
    pub slots: u64,
    pub floor_slots: u64,
    pub cold_experts: ColdExperts,
}

impl TryFrom<StreamedBankParts> for StreamedBank {
    type Error = OutOfBounds;

    fn try_from(parts: StreamedBankParts) -> Result<Self, OutOfBounds> {
        let StreamedBankParts { experts, slots, floor_slots, cold_experts } = parts;
        OutOfBounds::check(
            floor_slots <= slots && slots < experts.get(),
            "a streamed bank holds at least its floor and fewer than every expert",
        )?;
        Ok(Self { experts, slots, floor_slots, cold_experts })
    }
}

/// Where a streamed fit keeps the experts its bank lacks.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub enum ColdExperts {
    /// Read from the model file on demand (unified memory).
    Disk,
    /// In a host-RAM tier as host memory lets it grow, the rest read from the model file (a
    /// discrete device). `cache_bytes` is what the tier holds past its floor: cache the load
    /// grows into free host RAM, which no memory domain's required bytes count.
    #[serde(rename_all = "camelCase")]
    HostRam { cache_bytes: u64 },
}

#[cfg(test)]
mod model_residency_tests {
    use super::*;

    #[test]
    fn residencies_round_trip_in_their_wire_shape() {
        let estimated = |tokens_per_second: f64| ModelPerformance::Estimated {
            samples: vec![PerformanceEvidence {
                context_tokens: 4_096,
                estimated_tokens_per_second: tokens_per_second,
            }],
        };
        let wire = |tokens_per_second: f64| {
            serde_json::json!({
                "_tag": "Estimated",
                "samples": [{ "contextTokens": 4096, "estimatedTokensPerSecond": tokens_per_second }]
            })
        };
        let bank = |cold_experts| {
            StreamedBank::try_from(StreamedBankParts {
                experts: NonZeroU64::new(256).unwrap(),
                slots: 154,
                floor_slots: 26,
                cold_experts,
            })
            .unwrap()
        };
        let wire_bank = |cold: serde_json::Value| {
            serde_json::json!({ "experts": 256, "slots": 154, "floorSlots": 26, "coldExperts": cold })
        };
        for (residency, json) in [
            (
                ModelResidency::Resident {
                    performance: ModelPerformance::Pending,
                },
                serde_json::json!({ "_tag": "Resident", "performance": { "_tag": "Pending" } }),
            ),
            (
                ModelResidency::Streamed {
                    bank: bank(ColdExperts::HostRam { cache_bytes: 9_000_000_000 }),
                    performance: estimated(20.0),
                    floor_performance: estimated(5.5),
                },
                serde_json::json!({
                    "_tag": "Streamed",
                    "bank": wire_bank(serde_json::json!({ "_tag": "HostRam", "cacheBytes": 9_000_000_000u64 })),
                    "performance": wire(20.0),
                    "floorPerformance": wire(5.5)
                }),
            ),
            (
                ModelResidency::Streamed {
                    bank: bank(ColdExperts::Disk),
                    performance: ModelPerformance::Pending,
                    floor_performance: ModelPerformance::Pending,
                },
                serde_json::json!({
                    "_tag": "Streamed",
                    "bank": wire_bank(serde_json::json!({ "_tag": "Disk" })),
                    "performance": { "_tag": "Pending" },
                    "floorPerformance": { "_tag": "Pending" }
                }),
            ),
        ] {
            assert_eq!(serde_json::to_value(&residency).expect("serialize"), json);
            assert_eq!(
                serde_json::from_value::<ModelResidency>(json).expect("deserialize"),
                residency
            );
        }
        // A streamed bank holds at least its floor, and fewer than every expert.
        for (slots, floor_slots) in [(20, 26), (256, 26)] {
            assert!(serde_json::from_value::<StreamedBank>(serde_json::json!({
                "experts": 256, "slots": slots, "floorSlots": floor_slots, "coldExperts": { "_tag": "Disk" }
            }))
            .is_err());
        }
    }
}

/// The state of the measurement job, which measures this machine's selected executor in a
/// worker process of its own, so that speed estimates can be made. It runs while no model is
/// loading or being optimized, and while no request holds a loaded model, beside which it runs
/// once that model has made room for it. It moves from `Unmeasured` to `Measuring`, then to
/// `Measured`, or to `Failed` and back to `Measuring` at its retry, or back to `Unmeasured` when
/// a load or a request preempts the job.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
pub enum MeasurementState {
    /// Measurements are missing and no job has run yet: it waits for the device to be free.
    Unmeasured,
    /// A job is measuring this machine.
    Measuring,
    /// Every measurement that estimates read is stored, or recorded as unobtainable on this
    /// machine.
    Measured,
    /// The last job did not store every measurement; the next starts at `retry_at`
    /// (milliseconds since the Unix epoch).
    #[serde(rename_all = "camelCase")]
    Failed { retry_at: u64 },
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase")]
pub enum ModelAssessment {
    /// A fitting model at `served`, the largest context tier within `profile` at which its
    /// standard workload fits this computer, where it is placed by tier, how it holds its weights,
    /// and its decode-speed estimates held that way up to `served`.
    #[serde(rename_all = "camelCase")]
    Fits {
        profile: ServingProfile,
        served: ServingProfile,
        assessment_id: ModelAssessmentId,
        memory: Vec<MemoryAssessment>,
        memory_breakdown: MemoryBreakdown,
        placement: ModelPlacement,
        residency: ModelResidency,
        projector: ProjectorVerdict,
        drafter: DrafterVerdict,
    },
    /// A model that fits at no context tier: `deficit_bytes` is its shortfall at the smallest,
    /// `shortfall`.
    #[serde(rename_all = "camelCase")]
    DoesNotFit {
        profile: ServingProfile,
        shortfall: ServingProfile,
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

/// Every model assessment in the service's current environment: its structure is established at
/// service start, and its identity changes when the measurement job stores this machine's
/// measurements.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelAssessmentsSnapshot {
    pub revision: u64,
    pub environment_id: AssessmentEnvironmentId,
    pub catalog: ModelAssessmentDomainSnapshot,
    pub custom: ModelAssessmentDomainSnapshot,
    /// Whether this machine's measurements, which speed estimates read, exist.
    pub measurement: MeasurementState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelAssessmentsInvalidation {
    pub revision: u64,
}

/// A download of a custom model's `part`, by intent: the service works out its files from the
/// local-models configuration and keeps the request until it ends, resolving it again after a
/// restart. Every `ModelDownload` it produces is for `model_id`.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct StartModelDownloadRequest {
    pub model_id: ModelId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "openapi", schema(nullable = false))]
    pub part: Option<CustomModelPart>,
}

/// Which part of a custom model a download fetches: without one, an added model's files at its
/// pinned commit with the drafter it downloads with; `Drafter`, only the drafter its choice
/// resolves to; `Update`, an added model's files at the commit its repository has moved to.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum CustomModelPart {
    Drafter,
    Update,
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
    pub for_model: ModelId,
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
    /// Removes Magnitude's copies of `package_ids` as one removal, returning the bytes freed: a
    /// file another installed package is served from stays.
    fn remove_installed_packages(
        &self,
        package_ids: &[ModelPackageId],
    ) -> BoxFuture<'_, Result<u64, InventoryError>>;
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
/// first load does not. Reports progress until it finishes, `Waiting` while the job waits for the
/// device; the future ends once the model is tuned, or tuning is unneeded or cannot finish, and
/// dropping it stops the job. A failure is the optimizer's to log: the model is installed either
/// way.
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
        let local = "local:my-finetune-q4_k_m:9f2c1a7b3e04";
        let configuration: LocalModelsConfiguration = serde_json::from_value(json!({
            "roots": [
                { "kind": "LmStudio", "path": "/home/u/.lmstudio/models", "enabled": true },
                { "kind": "Folder", "path": "/models/finetune.gguf", "enabled": true }
            ],
            "added": [
                {
                    "repository": "owner/repo",
                    "path": "model-Q4_K_M.gguf",
                    "revision": "c".repeat(40),
                    "projector": "mmproj.gguf"
                }
            ],
            "hidden": [local],
            "drafters": [
                { "modelId": "hf:owner/repo/model-Q4_K_M.gguf", "drafter": { "_tag": "None" } },
                {
                    "modelId": local,
                    "drafter": {
                        "_tag": "Reference",
                        "reference": { "_tag": "Catalog", "modelId": "qwen3.6-35b-a3b:gguf:q4" }
                    }
                }
            ]
        }))
        .expect("deserialize configuration");
        assert_eq!(configuration.roots[0].kind, ModelRootKind::LmStudio);
        assert_eq!(
            configuration.added[0].model_id().as_str(),
            "hf:owner/repo/model-Q4_K_M.gguf"
        );
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
            model_id: "local:model:000000000000"
                .parse()
                .expect("model ID"),
        };
        assert_eq!(
            serde_json::to_value(&subject).expect("serialize")["_tag"],
            "Custom"
        );
    }
}
