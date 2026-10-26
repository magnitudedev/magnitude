//! The release catalog's component headers, exported for header-only tests,
//! and the headers of the few artifacts outside the catalog those tests read
//! ([`artifacts`]).
//!
//! `magnitude-service-catalog-tool headers` writes, from the catalog lock, every
//! component's exact GGUF header extended sparsely to the file's size, and each
//! artifact's at its pinned source, at `<repository>/<revision>/<path>` under
//! one directory, and this index beside them as [`INDEX`]. Each header opens as
//! the file it was read from ([`crate::PackageHeaders`]); the index names the
//! lock and the artifacts it was exported from, so a reader can refuse an
//! export of another lock or of other artifacts.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

/// The index's file name in an export directory.
pub const INDEX: &str = "index.json";

/// Every exported component of one catalog lock.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct HeaderIndex {
    /// [`lock_digest`] of the lock the headers were exported from.
    pub lock: String,
    pub components: Vec<CatalogComponent>,
    /// The [`artifacts`] the export holds.
    pub artifacts: Vec<ArtifactHeader>,
}

/// One component of a catalog model's variant: the file a load opens (the
/// first shard of a split GGUF, whose other shards are exported beside it) at
/// its locked source.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CatalogComponent {
    /// Catalog model ID, `<model>:<variant>` (`qwen3.5-4b:gguf:q4`).
    pub model: String,
    pub role: ComponentRole,
    pub repository: String,
    pub revision: String,
    pub path: PathBuf,
}

/// A component's role in its catalog model's variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ComponentRole {
    Target,
    Draft,
    Projector,
}

impl std::fmt::Display for ComponentRole {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Target => "target",
            Self::Draft => "draft",
            Self::Projector => "projector",
        })
    }
}

impl CatalogComponent {
    /// The component's file under an export directory.
    pub fn file(&self, root: &Path) -> PathBuf {
        source_file(root, &self.repository, &self.revision, &self.path)
    }
}

/// An artifact outside the catalog whose header tests read, by its name in
/// tests, at its pinned source.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ArtifactHeader {
    pub name: String,
    pub repository: String,
    pub revision: String,
    pub path: PathBuf,
}

impl ArtifactHeader {
    /// The artifact's file under an export directory.
    pub fn file(&self, root: &Path) -> PathBuf {
        source_file(root, &self.repository, &self.revision, &self.path)
    }
}

/// The artifacts outside the catalog whose headers tests read: the Qwen3.8
/// 27B and Nemotron 3.5 Bolt and Super VL DSpark drafts, the Qwen3.6 27B
/// target the Qwen3.6 27B DFlash draft's publication binds to and
/// bartowski's Gemma 4 E2B Q4_K_M (which keeps the sharing layers'
/// projections the catalog's file omits).
pub fn artifacts() -> Vec<ArtifactHeader> {
    [
        (
            "qwen3.8-27b-dspark",
            "magnitudedev/Qwen3.8-27B-DSpark-GGUF",
            "8741336fde55cf8fb01e0dd004a75b688aa0ed71",
            "Qwen3.8-27B-DSpark-Q8_0.gguf",
        ),
        (
            "nemotron-3.5-bolt-74b-a9b-dspark",
            "magnitudedev/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-DSpark-GGUF",
            "ca4ba8b87c9db0250c831e0f661c70ba5885d303",
            "dspark-NVIDIA-Nemotron-3.5-Bolt-74B-A9B-NVFP4.gguf",
        ),
        (
            "nemotron-3.5-super-vl-120b-a12b-dspark",
            "magnitudedev/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-DSpark-GGUF",
            "1e4dc982ea54dc02c678d9e4943ed77152778d26",
            "NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-DSpark-Q8_0.gguf",
        ),
        (
            "qwen3.6-27b-q4",
            "unsloth/Qwen3.6-27B-MTP-GGUF",
            "5cb35eb3dcbf52dbce5f87dbc64df6aaffadcace",
            "Qwen3.6-27B-UD-Q4_K_XL.gguf",
        ),
        (
            "gemma-4-e2b-it-bartowski-q4_k_m",
            "bartowski/google_gemma-4-E2B-it-GGUF",
            "81012ba3538e061d5ee003f11f25335b17f82e2d",
            "google_gemma-4-E2B-it-Q4_K_M.gguf",
        ),
    ]
    .into_iter()
    .map(|(name, repository, revision, path)| ArtifactHeader {
        name: name.to_owned(),
        repository: repository.to_owned(),
        revision: revision.to_owned(),
        path: PathBuf::from(path),
    })
    .collect()
}

fn source_file(root: &Path, repository: &str, revision: &str, path: &Path) -> PathBuf {
    root.join(repository).join(revision).join(path)
}

/// The identity of a catalog lock: the SHA-256 of its file's bytes, in hex.
pub fn lock_digest(lock: &[u8]) -> String {
    Sha256::digest(lock)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
