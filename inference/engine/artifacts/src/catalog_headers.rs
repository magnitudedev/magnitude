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
/// target the Qwen3.6 27B DFlash draft's publication binds to, Gemma 4 E2B
/// QAT with its projector and bartowski's Gemma 4 E2B Q4_K_M (which keeps the
/// sharing layers' projections the QAT file omits), and the releases of
/// architectures and drafts the engine supports but the catalog does not
/// list: LFM2.5 2.6B and 8B-A1B with their DSpark drafts, LFM3-VL 2B,
/// MiniCPM5 2B with its DSpark draft, MiniCPM-V 4.7 1B and 35B-A3B, and
/// Nemotron 3.5 Lightning 30B-A3B with its DFlash draft.
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
        (
            "gemma-4-e2b-it-qat-q4",
            "unsloth/gemma-4-E2B-it-qat-GGUF",
            "66a399f68ddd113b06dff02fca9523e55465d11d",
            "gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf",
        ),
        (
            "gemma-4-e2b-it-qat-mmproj",
            "unsloth/gemma-4-E2B-it-qat-GGUF",
            "66a399f68ddd113b06dff02fca9523e55465d11d",
            "mmproj-BF16.gguf",
        ),
        (
            "lfm2.5-2.6b-q4",
            "LiquidAI/LFM2.5-2.6B-GGUF",
            "b421ad1d549afeda6a0fb2ad3a697cb5a7879adc",
            "LFM2.5-2.6B-Q4_K_M.gguf",
        ),
        (
            "lfm2.5-2.6b-dspark",
            "magnitudedev/LFM2.5-2.6B-DSpark-GGUF",
            "60103108e36ef66c2591c2b1ea3e6530a4e2c3b0",
            "LFM2.5-2.6B-DSpark-Q8_0.gguf",
        ),
        (
            "lfm2.5-8b-a1b-q4",
            "LiquidAI/LFM2.5-8B-A1B-GGUF",
            "dfd5fdcad7a1c0d31473fb4ca443b8befbacddf0",
            "LFM2.5-8B-A1B-Q4_K_M.gguf",
        ),
        (
            "lfm2.5-8b-a1b-dspark",
            "magnitudedev/LFM2.5-8B-A1B-DSpark-GGUF",
            "0e2ce447c266c98603073cf9927a56130b9dca43",
            "LFM2.5-8B-A1B-DSpark-Q8_0.gguf",
        ),
        (
            "lfm3-vl-2b-q4-qat",
            "LiquidAI/LFM3-VL-2B-GGUF",
            "7c3b4a43c9023e80bf234fec5e826a3365c8dda8",
            "LFM3-VL-2B-QAD-Q4_0.gguf",
        ),
        (
            "lfm3-vl-2b-mmproj",
            "LiquidAI/LFM3-VL-2B-GGUF",
            "7c3b4a43c9023e80bf234fec5e826a3365c8dda8",
            "mmproj-LFM3-VL-2B-BF16.gguf",
        ),
        (
            "minicpm5-2b-q4",
            "openbmb/MiniCPM5-2B-GGUF",
            "2079a22f3beaa4e306449978533478fe0522f4b3",
            "MiniCPM5-2B-Q4_K_M.gguf",
        ),
        (
            "minicpm5-2b-dspark",
            "magnitudedev/MiniCPM5-2B-DSpark-GGUF",
            "fbc306afa97f19283820af3089f9306f7b2051cb",
            "MiniCPM5-2.6B-DSpark.gguf",
        ),
        (
            "minicpm-v-4.7-1b-q4",
            "openbmb/MiniCPM-V-4.7-1B-gguf",
            "59cd286146b6ce70f818508ccf512e11438d8957",
            "MiniCPM-V-4_7-1B-Q4_K_M.gguf",
        ),
        (
            "minicpm-v-4.7-1b-mmproj",
            "openbmb/MiniCPM-V-4.7-1B-gguf",
            "59cd286146b6ce70f818508ccf512e11438d8957",
            "mmproj-model-f16-4x.gguf",
        ),
        (
            "minicpm-v-4.7-35b-a3b-q4",
            "openbmb/MiniCPM-V-4.7-35B-A3B-gguf",
            "3c3f7e84f14a760ff26d17ab5932d4aa996227c0",
            "MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf",
        ),
        (
            "minicpm-v-4.7-35b-a3b-mmproj",
            "openbmb/MiniCPM-V-4.7-35B-A3B-gguf",
            "3c3f7e84f14a760ff26d17ab5932d4aa996227c0",
            "mmproj-model-f16-4x.gguf",
        ),
        (
            "nemotron-3.5-lightning-30b-a3b-nvfp4",
            "ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF",
            "88d7ce0b0fa385c5108866ce5d33690927531a37",
            "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4.gguf",
        ),
        (
            "nemotron-3.5-lightning-30b-a3b-q4",
            "ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF",
            "88d7ce0b0fa385c5108866ce5d33690927531a37",
            "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q4_K_M.gguf",
        ),
        (
            "nemotron-3.5-lightning-30b-a3b-q8",
            "ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF",
            "88d7ce0b0fa385c5108866ce5d33690927531a37",
            "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q8_0.gguf",
        ),
        (
            "nemotron-3.5-lightning-30b-a3b-dflash",
            "magnitudedev/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash-GGUF",
            "4f5b414b9388a4543dcc4b24758e09d68140a6fe",
            "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash.gguf",
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
