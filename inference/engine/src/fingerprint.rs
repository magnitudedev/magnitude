//! A target's shape fingerprint, from its GGUF header alone.
//!
//! Requantizations, finetunes and abliterations keep the architecture, the
//! decoder's dimensions, the vocabulary and the tokenizer's token table, and
//! drafters bind to exactly these, so equal fingerprints match a model to the
//! catalog target it derives from when no lineage metadata does.

use magnitude_artifacts::{
    gguf::{Directory, Scalar, Value},
    PackageHeaders,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{fmt, path::Path};

/// Versions the token-table digest: a change to what it hashes changes the tag.
const TOKENIZER_DIGEST_TAG: &[u8] = b"magnitude.tokenizer-digest.v1";

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelFingerprint {
    pub architecture: String,
    pub block_count: u32,
    pub embedding_length: u32,
    pub head_count: u32,
    pub head_count_kv: u32,
    pub key_length: u32,
    pub vocab_size: u32,
    pub expert_count: Option<u32>,
    pub tokenizer_digest: String,
}

/// Why a header has no fingerprint: a fact it needs is absent or malformed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FingerprintError {
    Artifact(String),
    Missing(String),
    Invalid(String),
}

impl fmt::Display for FingerprintError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Artifact(error) => write!(formatter, "header: {error}"),
            Self::Missing(key) => write!(formatter, "missing {key}"),
            Self::Invalid(key) => write!(formatter, "{key} is outside the fingerprint's domain"),
        }
    }
}

impl std::error::Error for FingerprintError {}

impl ModelFingerprint {
    /// The fingerprint of the GGUF component at `path`: the file itself, or
    /// every shard when it is the first shard of a split GGUF. Reads headers
    /// only.
    pub fn from_path(path: &Path) -> Result<Self, FingerprintError> {
        let headers = PackageHeaders::open(path, None)
            .map_err(|error| FingerprintError::Artifact(error.to_string()))?;
        Self::from_directory(headers.target())
    }

    /// The fingerprint of a target's header directory. Its block count is the decoder's, without
    /// MTP prediction layers (`nextn_predict_layers`). A per-layer head
    /// count is its largest layer's (recurrent layers state zero key-value
    /// heads); an absent key-value count is the head count and an absent key
    /// length is the embedding length per head, as llama.cpp reads them. A
    /// zero expert count is a dense model.
    pub fn from_directory(directory: &Directory) -> Result<Self, FingerprintError> {
        let architecture = directory
            .value("general.architecture")
            .and_then(Value::string)
            .ok_or_else(|| FingerprintError::Missing("general.architecture".into()))?
            .to_owned();
        let key = |name: &str| format!("{architecture}.{name}");
        let count = |name: &str| -> Result<Option<u64>, FingerprintError> {
            let key = key(name);
            match directory.value(&key) {
                None => Ok(None),
                Some(Value::Array(values)) => values
                    .iter()
                    .map(|value| {
                        unsigned(value).ok_or_else(|| FingerprintError::Invalid(key.clone()))
                    })
                    .try_fold(None, |largest: Option<u64>, value| {
                        Ok(Some(largest.unwrap_or(0).max(value?)))
                    }),
                Some(value) => value
                    .unsigned()
                    .map(Some)
                    .ok_or_else(|| FingerprintError::Invalid(key.clone())),
            }
        };
        let required = |name: &str| {
            count(name)?
                .filter(|value| *value > 0)
                .ok_or_else(|| FingerprintError::Missing(key(name)))
        };
        let narrow = |name: &str, value: u64| {
            u32::try_from(value).map_err(|_| FingerprintError::Invalid(name.to_owned()))
        };
        // A file carrying MTP heads counts them as extra blocks; the decoder is the same with or
        // without them, so a conversion that dropped the heads keeps the fingerprint.
        let prediction_layers = count("nextn_predict_layers")?.unwrap_or(0);
        let block_count = required("block_count")?
            .checked_sub(prediction_layers)
            .filter(|blocks| *blocks > 0)
            .ok_or_else(|| FingerprintError::Invalid(key("nextn_predict_layers")))?;
        let embedding_length = required("embedding_length")?;
        let head_count = required("attention.head_count")?;
        let head_count_kv = count("attention.head_count_kv")?
            .filter(|value| *value > 0)
            .unwrap_or(head_count);
        let key_length = match count("attention.key_length")?.filter(|value| *value > 0) {
            Some(length) => length,
            None => embedding_length / head_count,
        };
        let vocab_size = directory
            .tensor("token_embd.weight")
            .and_then(|tensor| tensor.shape.first().copied())
            .ok_or_else(|| FingerprintError::Missing("token_embd.weight".into()))?;
        let expert_count = count("expert_count")?.filter(|value| *value > 0);
        Ok(Self {
            block_count: narrow("block_count", block_count)?,
            embedding_length: narrow("embedding_length", embedding_length)?,
            head_count: narrow("attention.head_count", head_count)?,
            head_count_kv: narrow("attention.head_count_kv", head_count_kv)?,
            key_length: narrow("attention.key_length", key_length)?,
            vocab_size: narrow("token_embd.weight", vocab_size)?,
            expert_count: expert_count
                .map(|count| narrow("expert_count", count))
                .transpose()?,
            tokenizer_digest: tokenizer_digest(directory)?,
            architecture,
        })
    }
}

/// The lowercase hex sha256 of the version tag and every entry of
/// `tokenizer.ggml.tokens` in order, each length-prefixed (u64
/// little-endian) so no two tables share a digest.
fn tokenizer_digest(directory: &Directory) -> Result<String, FingerprintError> {
    let key = "tokenizer.ggml.tokens";
    let Some(Value::Array(tokens)) = directory.value(key) else {
        return Err(FingerprintError::Missing(key.into()));
    };
    let mut digest = Sha256::new();
    length_prefixed(&mut digest, TOKENIZER_DIGEST_TAG);
    for token in tokens {
        let Scalar::String(token) = token else {
            return Err(FingerprintError::Invalid(key.into()));
        };
        length_prefixed(&mut digest, token.as_bytes());
    }
    Ok(digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn length_prefixed(digest: &mut Sha256, bytes: &[u8]) {
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
}

fn unsigned(value: &Scalar) -> Option<u64> {
    match value {
        Scalar::Unsigned(value) => Some(*value),
        Scalar::Signed(value) => u64::try_from(*value).ok(),
        _ => None,
    }
}
