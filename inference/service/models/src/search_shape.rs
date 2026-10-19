//! What Hugging Face search learns from a repository's model file: its architecture and shape.
//!
//! A result's model file is found by size, never by name: its GGUFs (shards counted together,
//! projectors excluded) largest first, skipping files whose header names no architecture, a
//! projector's or a separate drafter's, or declares fewer tensors than blocks (a piece of a
//! layer-split model), up to [`MAX_MODEL_FILE_READS`] files. Each is read from its
//! first [`SHAPE_PREFIX_BYTES`] with one ranged GET: the metadata up to the tokenizer's token list,
//! whose element count is the vocabulary size. When the prefix holds no shape (its writer put the
//! shape keys after the token list) or a near miss of a catalog model's (a key after the token
//! list, such as an appended MTP head count, may settle it), the whole header is read by the remote
//! header reader, header bytes only. The shape comes from metadata alone, never tensor types. It is
//! the engine's `ModelFingerprint` without the token table digest, and a result is a variant of the first
//! catalog model (in catalog order) whose fingerprint has the same shape. What a repository's
//! commit holds is cached, so it is read once.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use futures_util::future::BoxFuture;
use magnitude_artifacts::gguf::{Directory, PROJECTOR_ARCHITECTURE, Scalar, Value, read_header};
use magnitude_service_contracts::models::{ModelFingerprint, ModelId};
use magnitude_service_contracts::{HuggingFaceRepositoryRequest, InventoryError};
use reqwest::header::RANGE;
use serde::{Deserialize, Serialize};

use crate::cache::ModelIndexKind;
use crate::candidate_files::{CandidateGguf, RecognizedArchitectures, group_ggufs};
use crate::header_layout::{Layout, walk};
use crate::inventory::ManagedModelStore;

/// The bytes read from the start of a search result's GGUF.
pub const SHAPE_PREFIX_BYTES: usize = 64 * 1024;
/// The most GGUFs read to find a repository's model file.
pub const MAX_MODEL_FILE_READS: usize = 3;
/// How long finding and reading a repository's model file may take before its result is shown
/// without what it would tell.
const MODEL_READ_TIMEOUT: Duration = Duration::from_secs(15);
/// Versions the cached outcome: a change to how the model file is found drops older entries.
const CACHE_VERSION: &str = "model-file.v3";
const TOKENS_KEY: &str = "tokenizer.ggml.tokens";

/// Each catalog model's shape fingerprint, as its first variant's model ID, in catalog order.
pub trait CatalogFingerprints: Send + Sync {
    fn catalog_fingerprints(&self) -> BoxFuture<'_, Vec<(ModelId, ModelFingerprint)>>;
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelShape {
    architecture: String,
    block_count: u32,
    embedding_length: u32,
    head_count: u32,
    head_count_kv: u32,
    key_length: u32,
    vocab_size: u32,
    expert_count: Option<u32>,
}

/// What a repository's commit holds for search: the model file's architecture and shape (absent
/// when its header doesn't state one), headers read that hold no model file, or headers read that
/// are all drafters' (a drafter repository, never a model result).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum RepositoryModel {
    Model {
        architecture: String,
        shape: Option<ModelShape>,
    },
    Unreadable,
    Drafter,
}

/// The header metadata in a GGUF prefix: its complete entries, as a header of their own without
/// tensors, and the token list's element count when the prefix reaches it.
fn prefix_metadata(bytes: &[u8]) -> Option<(Directory, Option<u64>)> {
    let (entries, open) = match walk(bytes).ok()? {
        Layout::Complete { entries, .. } => (entries, None),
        Layout::Incomplete { entries, open, .. } => (entries, open),
    };
    let vocab_size = entries
        .iter()
        .find(|entry| entry.key == TOKENS_KEY)
        .map(|entry| entry.head_end)
        .or_else(|| {
            open.filter(|open| open.key == TOKENS_KEY)
                .map(|open| open.head_end)
        })
        .and_then(|head_end| {
            let head = bytes.get(head_end.checked_sub(16)?..head_end)?;
            let le_u32 =
                |at: usize| u32::from_le_bytes(head[at..at + 4].try_into().unwrap_or_default());
            (le_u32(0) == 9 && le_u32(4) == 8)
                .then(|| u64::from_le_bytes(head[8..16].try_into().unwrap_or_default()))
        });
    let mut metadata = bytes.get(..entries.last()?.end)?.to_vec();
    metadata
        .get_mut(8..16)?
        .copy_from_slice(&0_u64.to_le_bytes());
    metadata
        .get_mut(16..24)?
        .copy_from_slice(&u64::try_from(entries.len()).ok()?.to_le_bytes());
    Some((read_header(&metadata).ok()?, vocab_size))
}

impl ModelShape {
    /// The shape of a GGUF from any prefix that holds its token list's element count and the
    /// metadata entries the shape needs (writers put them before the tokenizer, but not all do),
    /// read the way the engine reads its fingerprint: the decoder's block count without MTP prediction
    /// layers, a per-layer count as its largest layer's, an absent key-value head count as the
    /// head count, an absent key length as the embedding length per head, and a zero expert count
    /// as a dense model.
    pub fn from_header_prefix(bytes: &[u8]) -> Option<Self> {
        let (directory, vocab_size) = prefix_metadata(bytes)?;
        Self::from_directory(&directory, vocab_size?)
    }

    fn from_directory(directory: &Directory, vocab_size: u64) -> Option<Self> {
        let architecture = directory
            .value("general.architecture")?
            .string()?
            .to_owned();
        let count = |name: &str| -> Option<Option<u64>> {
            match directory.value(&format!("{architecture}.{name}")) {
                None => Some(None),
                Some(Value::Array(values)) => {
                    values.iter().try_fold(None, |largest: Option<u64>, value| {
                        Some(Some(largest.unwrap_or(0).max(unsigned(value)?)))
                    })
                }
                Some(value) => value.unsigned().map(Some),
            }
        };
        let positive = |name: &str| Some(count(name)?.filter(|value| *value > 0));
        let block_count = positive("block_count")??
            .checked_sub(count("nextn_predict_layers")?.unwrap_or(0))
            .filter(|blocks| *blocks > 0)?;
        let embedding_length = positive("embedding_length")??;
        let head_count = positive("attention.head_count")??;
        let head_count_kv = positive("attention.head_count_kv")?.unwrap_or(head_count);
        let key_length = positive("attention.key_length")?.unwrap_or(embedding_length / head_count);
        let narrow = |value: u64| u32::try_from(value).ok();
        let expert_count = match positive("expert_count")? {
            Some(count) => Some(narrow(count)?),
            None => None,
        };
        Some(Self {
            block_count: narrow(block_count)?,
            embedding_length: narrow(embedding_length)?,
            head_count: narrow(head_count)?,
            head_count_kv: narrow(head_count_kv)?,
            key_length: narrow(key_length)?,
            vocab_size: narrow(vocab_size).filter(|size| *size > 0)?,
            expert_count,
            architecture,
        })
    }

    /// The shape of an engine fingerprint: every dimension but the token table.
    #[must_use]
    pub fn of(fingerprint: &ModelFingerprint) -> Self {
        Self {
            architecture: fingerprint.architecture.clone(),
            block_count: fingerprint.block_count,
            embedding_length: fingerprint.embedding_length,
            head_count: fingerprint.head_count,
            head_count_kv: fingerprint.head_count_kv,
            key_length: fingerprint.key_length,
            vocab_size: fingerprint.vocab_size,
            expert_count: fingerprint.expert_count,
        }
    }

    /// The catalog model a model of this shape is a variant of: the first in `catalog` (each
    /// catalog model's fingerprint, in catalog order) of the same shape. Names and declared
    /// lineage never decide; search and inspection both answer by this alone.
    #[must_use]
    pub fn variant_of(&self, catalog: &[(ModelId, ModelFingerprint)]) -> Option<ModelId> {
        catalog
            .iter()
            .find(|(_, fingerprint)| self.matches(fingerprint))
            .map(|(model_id, _)| model_id.clone())
    }

    /// Whether `fingerprint` is of this shape's architecture, width and vocabulary: the dimensions
    /// no key that may follow the token list changes.
    fn resembles(&self, fingerprint: &ModelFingerprint) -> bool {
        self.architecture == fingerprint.architecture
            && self.embedding_length == fingerprint.embedding_length
            && self.vocab_size == fingerprint.vocab_size
    }

    /// Whether `fingerprint` has this shape: every dimension but the token table.
    pub fn matches(&self, fingerprint: &ModelFingerprint) -> bool {
        self.architecture == fingerprint.architecture
            && self.block_count == fingerprint.block_count
            && self.embedding_length == fingerprint.embedding_length
            && self.head_count == fingerprint.head_count
            && self.head_count_kv == fingerprint.head_count_kv
            && self.key_length == fingerprint.key_length
            && self.vocab_size == fingerprint.vocab_size
            && self.expert_count == fingerprint.expert_count
    }
}

#[cfg(test)]
impl ModelShape {
    /// A catalog fingerprint of this shape, for tests.
    pub(crate) fn fingerprint(&self) -> ModelFingerprint {
        ModelFingerprint {
            architecture: self.architecture.clone(),
            block_count: self.block_count,
            embedding_length: self.embedding_length,
            head_count: self.head_count,
            head_count_kv: self.head_count_kv,
            key_length: self.key_length,
            vocab_size: self.vocab_size,
            expert_count: self.expert_count,
            tokenizer_digest: String::new(),
        }
    }
}

fn unsigned(value: &Scalar) -> Option<u64> {
    match value {
        Scalar::Unsigned(value) => Some(*value),
        Scalar::Signed(value) => u64::try_from(*value).ok(),
        _ => None,
    }
}

/// What a GGUF read from its prefix is for search.
#[derive(Clone, Copy, PartialEq, Eq)]
enum FileKind {
    Model,
    Drafter,
    Other,
}

/// A single file declaring fewer tensors than its blocks can't hold the decoder: a prediction head
/// alone when it declares prediction layers (an MTP block, a drafter), else a piece of a
/// layer-split model, which repeats the whole model's metadata over a few tensors (one layer, the
/// output head). A split set's shards hold part of the set by design and are never asked (they are
/// grouped by the listing, since `split.count` comes after the tokenizer where writers append it);
/// a block count the prefix doesn't reach says nothing.
fn single_file_kind(directory: &Directory, prefix: &[u8], architecture: &str) -> FileKind {
    let count = |key: &str| {
        directory
            .value(&format!("{architecture}.{key}"))
            .and_then(Value::unsigned)
    };
    let split = directory
        .value("split.count")
        .and_then(Value::unsigned)
        .is_some_and(|count| count > 1);
    let tensors = prefix
        .get(8..16)
        .and_then(|bytes| bytes.try_into().ok())
        .map(u64::from_le_bytes);
    let partial = !split
        && count("block_count")
            .zip(tensors)
            .is_some_and(|(blocks, tensors)| tensors < blocks);
    match (partial, count("nextn_predict_layers").unwrap_or(0) > 0) {
        (false, _) => FileKind::Model,
        (true, true) => FileKind::Drafter,
        (true, false) => FileKind::Other,
    }
}

/// How many versions a repository's GGUF paths offer: its complete GGUFs, split shards counted
/// once, without projectors. Names alone cannot tell a drafter from a model, so this is
/// approximate. `None` when it offers none.
pub(crate) fn version_count<'a>(paths: impl IntoIterator<Item = &'a Path>) -> Option<u32> {
    let files = paths
        .into_iter()
        .filter(|path| !crate::catalog::is_projector_path(path))
        .map(|path| CandidateGguf {
            path: path.to_path_buf(),
            size_bytes: 0,
        })
        .collect::<Vec<_>>();
    u32::try_from(group_ggufs(&files).len())
        .ok()
        .filter(|count| *count > 0)
}

/// What `repository` at `commit` holds for search: cached, else found and read (see the module
/// documentation). `None` is a failed or slow read (a refused gated file among them), which is
/// tried again next time. A `restricted` (gated or private) repository's read is cached for the
/// credential that made it only.
pub(crate) async fn repository_model(
    models: &Arc<ManagedModelStore>,
    architectures: &RecognizedArchitectures,
    catalog: &[(ModelId, ModelFingerprint)],
    repository: &str,
    commit: &str,
    restricted: bool,
) -> Option<RepositoryModel> {
    let key = format!(
        "{CACHE_VERSION}:{}:{repository}@{commit}{}",
        models.client.endpoint(),
        models.restricted_access_scope(restricted)
    );
    if let Some(cached) = models
        .cache
        .read_index::<RepositoryModel>(ModelIndexKind::HuggingFaceSearchShape, &key)
    {
        return Some(cached);
    }
    let read = read_repository_model(models, architectures, catalog, repository, commit);
    let model = match tokio::time::timeout(MODEL_READ_TIMEOUT, read).await {
        Ok(Ok(model)) => model,
        Ok(Err(error)) => {
            tracing::debug!(%repository, %error, "search model file read failed");
            return None;
        }
        Err(_) => {
            tracing::debug!(%repository, "search model file read timed out");
            return None;
        }
    };
    models
        .cache
        .write_index(ModelIndexKind::HuggingFaceSearchShape, &key, &model);
    Some(model)
}

async fn read_repository_model(
    models: &Arc<ManagedModelStore>,
    architectures: &RecognizedArchitectures,
    catalog: &[(ModelId, ModelFingerprint)],
    repository: &str,
    commit: &str,
) -> Result<RepositoryModel, InventoryError> {
    let snapshot = match crate::preview::pinned_hugging_face_repository(
        models,
        HuggingFaceRepositoryRequest {
            repository: repository.to_owned(),
            revision: commit.to_owned(),
        },
    )
    .await
    {
        Ok(snapshot) => snapshot,
        Err(error) => return Err(error),
    };
    let files = snapshot
        .gguf_files
        .iter()
        .filter(|file| !crate::catalog::is_projector_path(&file.path))
        .map(|file| CandidateGguf {
            path: file.path.clone(),
            size_bytes: file.size_bytes,
        })
        .collect::<Vec<_>>();
    let mut groups = group_ggufs(&files);
    groups.sort_by_key(|group| std::cmp::Reverse(group.size_bytes()));
    let mut read_any = false;
    let mut drafters_only = true;
    for group in groups.iter().take(MAX_MODEL_FILE_READS) {
        let primary = group.primary();
        let prefix = models
            .read_prefix(repository, commit, &primary.path)
            .await?;
        read_any = true;
        let metadata = prefix_metadata(&prefix).map(|(directory, _)| directory);
        let architecture = metadata.as_ref().and_then(|directory| {
            directory
                .value("general.architecture")
                .and_then(Value::string)
                .map(ToOwned::to_owned)
        });
        let file = match (&architecture, &metadata) {
            (Some(architecture), _) if architecture == PROJECTOR_ARCHITECTURE => FileKind::Other,
            (Some(architecture), _) if architectures.is_separate_drafter(architecture) => {
                FileKind::Drafter
            }
            (Some(_), _) if group.files.len() > 1 => FileKind::Model,
            (Some(architecture), Some(directory)) => {
                single_file_kind(directory, &prefix, architecture)
            }
            (Some(_), None) => FileKind::Model,
            (None, _) => FileKind::Other,
        };
        let Some(architecture) = architecture.filter(|_| file == FileKind::Model) else {
            drafters_only &= file == FileKind::Drafter;
            continue;
        };
        // Keys a writer put after the token list (some put the shape's, or an appended MTP head
        // count) are seen only in the whole header. A prefix's shape stands when the header ends
        // in the prefix, when it is a catalog model's, or when no catalog model is even of its
        // architecture, width and vocabulary; otherwise (no shape yet, or a near miss) the whole
        // header is read.
        let prefix_shape = ModelShape::from_header_prefix(&prefix);
        let settled = match walk(&prefix) {
            Ok(Layout::Incomplete { .. }) => prefix_shape.as_ref().is_some_and(|shape| {
                catalog
                    .iter()
                    .any(|(_, fingerprint)| shape.matches(fingerprint))
                    || !catalog
                        .iter()
                        .any(|(_, fingerprint)| shape.resembles(fingerprint))
            }),
            _ => true,
        };
        let shape = match settled {
            true => prefix_shape,
            false => {
                // The header's metadata only: the remote reader walks the tensor directory's
                // layout but never judges its tensor types, which inspection decides. The
                // repository's cached headers are references for its tokenizer.
                let references = snapshot
                    .gguf_files
                    .iter()
                    .filter(|other| other.path != primary.path)
                    .filter_map(|other| {
                        models.hugging_face_reference_header(repository, commit, other)
                    })
                    .collect::<Vec<_>>();
                let references = references.iter().map(Vec::as_slice).collect::<Vec<_>>();
                let token = models.hub_token();
                let mut remote = crate::preview::RemoteRanges::new(
                    &models.http,
                    token.as_deref(),
                    models.client.endpoint(),
                    repository,
                    commit,
                    &primary.path,
                    primary.size_bytes,
                )?;
                match crate::remote_header::read_header(&mut remote, &references).await {
                    Ok(header) => ModelShape::from_header_prefix(&header),
                    Err(InventoryError::Integrity(_)) => prefix_shape,
                    Err(error) => return Err(error),
                }
            }
        };
        return Ok(RepositoryModel::Model {
            architecture,
            shape,
        });
    }
    // Only headers actually read and naming no model architecture make a repository unreadable;
    // without a model file to read, nothing is known.
    if read_any && drafters_only {
        Ok(RepositoryModel::Drafter)
    } else if read_any {
        Ok(RepositoryModel::Unreadable)
    } else {
        Err(InventoryError::NotFound(format!(
            "{repository} lists no model GGUF"
        )))
    }
}

impl ManagedModelStore {
    /// At most the first [`SHAPE_PREFIX_BYTES`] of a repository file, in one ranged GET.
    async fn read_prefix(
        &self,
        repository: &str,
        commit: &str,
        path: &Path,
    ) -> Result<Vec<u8>, InventoryError> {
        let url = crate::preview::resolve_url(self.client.endpoint(), repository, commit, path)?;
        let range = format!("bytes=0-{}", SHAPE_PREFIX_BYTES - 1);
        let token = self.hub_token();
        let (mut response, _) =
            crate::preview::send_with_credential_fallback(token.as_deref(), || {
                self.http.get(url.clone()).header(RANGE, &range)
            })
            .await?;
        if !response.status().is_success() {
            return Err(InventoryError::Upstream(format!(
                "GGUF prefix request returned HTTP {}",
                response.status()
            )));
        }
        let mut bytes = Vec::with_capacity(SHAPE_PREFIX_BYTES);
        while bytes.len() < SHAPE_PREFIX_BYTES
            && let Some(chunk) = response
                .chunk()
                .await
                .map_err(|error| InventoryError::Upstream(error.to_string()))?
        {
            let take = chunk.len().min(SHAPE_PREFIX_BYTES - bytes.len());
            bytes.extend_from_slice(&chunk[..take]);
        }
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_gguf::fixture_header;

    #[test]
    fn version_count_groups_shards_and_excludes_projectors() {
        let paths = [
            "Model-Q4_K_M.gguf",
            "Model-Q8_0.gguf",
            "BF16/Model-BF16-00001-of-00003.gguf",
            "BF16/Model-BF16-00002-of-00003.gguf",
            "BF16/Model-BF16-00003-of-00003.gguf",
            "mmproj-Model-F16.gguf",
            "Model-mmproj-BF16.gguf",
        ]
        .map(Path::new);
        assert_eq!(version_count(paths), Some(3));
        assert_eq!(version_count([Path::new("mmproj-F16.gguf")]), None);
    }

    #[test]
    fn a_prefix_ending_inside_the_token_list_has_the_whole_header_s_shape() {
        for name in [
            "gemma-4-E2B-it-qat-UD-Q4_K_XL",
            "Qwen3.5-9B-heretic-Q4_K_M",
            "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004",
        ] {
            let header = fixture_header(name);
            let whole = ModelShape::from_header_prefix(&header).expect(name);
            assert_eq!(
                ModelShape::from_header_prefix(&header[..SHAPE_PREFIX_BYTES]).as_ref(),
                Some(&whole),
                "{name}"
            );
        }
    }

    #[test]
    fn finetunes_share_their_base_s_shape_and_other_models_do_not() {
        let shape = |name: &str| {
            ModelShape::from_header_prefix(&fixture_header(name)[..SHAPE_PREFIX_BYTES]).unwrap()
        };
        let gemma = shape("gemma-4-E2B-it-qat-UD-Q4_K_XL");
        assert_eq!(
            shape("Gemma-4-E2B-Uncensored-HauhauCS-Aggressive-Q2_K_P"),
            gemma
        );
        assert_ne!(shape("Qwen3.5-4B-Q3_K_M"), gemma);
        assert_ne!(
            shape("Qwen3.5-4B-Q3_K_M"),
            shape("Qwen3.5-9B-heretic-Q4_K_M")
        );
    }

    #[test]
    fn a_prefix_that_ends_before_the_token_count_has_no_shape() {
        let header = fixture_header("gemma-4-E2B-it-qat-UD-Q4_K_XL");
        assert_eq!(ModelShape::from_header_prefix(&header[..512]), None);
        assert_eq!(ModelShape::from_header_prefix(b"not a gguf"), None);
    }
}
