//! A catalog model's resolved local paths, as catalog acquisition reports
//! them (contract "Model reference"). The drafter and projector are always
//! the catalog's own.

use serde::Deserialize;
use serde_json::Value;
use std::path::{Path, PathBuf};

/// The fields of the acquisition object the harness loads from; the record
/// keeps the whole object.
#[derive(Clone, Debug, Deserialize)]
pub struct Artifacts {
    pub model: String,
    pub target: PathBuf,
    /// All target GGUF shard paths in shard order; defaults to `[target]` when
    /// absent (older acquisition objects).
    #[serde(default)]
    pub target_files: Vec<PathBuf>,
    pub target_sha256: String,
    /// The catalog drafter, `null` when the catalog declares none.
    pub drafter: Option<Drafter>,
}

impl Artifacts {
    /// The target GGUF shard paths in shard order (always non-empty).
    pub fn target_shards(&self) -> Vec<PathBuf> {
        if self.target_files.is_empty() {
            vec![self.target.clone()]
        } else {
            self.target_files.clone()
        }
    }

    /// The separate draft GGUF path, or `None` when the catalog declares no
    /// drafter or an embedded drafter (`drafter.path == target`).
    pub fn separate_drafter(&self) -> Option<&Path> {
        self.drafter
            .as_ref()
            .map(|drafter| drafter.path.as_path())
            .filter(|path| *path != self.target.as_path())
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct Drafter {
    pub path: PathBuf,
}

/// The acquisition object as read, kept verbatim for the result record.
pub struct ArtifactsFile {
    pub artifacts: Artifacts,
    pub raw: Value,
}

impl ArtifactsFile {
    pub fn read(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|error| format!("artifacts {}: {error}", path.display()))?;
        let raw: Value = serde_json::from_str(&text)
            .map_err(|error| format!("artifacts {}: {error}", path.display()))?;
        let artifacts = serde_json::from_value(raw.clone())
            .map_err(|error| format!("artifacts {}: {error}", path.display()))?;
        Ok(Self { artifacts, raw })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn target_shards_defaults_to_target_when_target_files_is_absent() {
        let artifacts: Artifacts = serde_json::from_value(json!({
            "model": "qwen3.5-4b:gguf:q4",
            "target": "/store/target.gguf",
            "target_sha256": "aa",
            "drafter": null,
        }))
        .expect("legacy artifacts without target_files deserialize");
        assert_eq!(
            artifacts.target_shards(),
            vec![PathBuf::from("/store/target.gguf")]
        );
    }

    #[test]
    fn target_shards_preserves_explicit_sharded_and_engram_paths() {
        let artifacts: Artifacts = serde_json::from_value(json!({
            "model": "deepseek-v4.1-flash:gguf:iq2-xxs",
            "target": "/store/DeepSeek-V4.1-Flash-00001-of-00006.gguf",
            "target_files": [
                "/store/DeepSeek-V4.1-Flash-00001-of-00006.gguf",
                "/store/DeepSeek-V4.1-Flash-00002-of-00006.gguf",
                "/store/DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf",
            ],
            "target_sha256": "aa",
            "drafter": null,
        }))
        .expect("sharded artifacts with target_files deserialize");
        assert_eq!(
            artifacts.target_shards(),
            vec![
                PathBuf::from("/store/DeepSeek-V4.1-Flash-00001-of-00006.gguf"),
                PathBuf::from("/store/DeepSeek-V4.1-Flash-00002-of-00006.gguf"),
                PathBuf::from("/store/DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf"),
            ]
        );
    }

    #[test]
    fn separate_drafter_excludes_embedded_drafter_pointing_to_target() {
        let embedded: Artifacts = serde_json::from_value(json!({
            "model": "glm-5.3-flash:gguf:iq2",
            "target": "/store/glm-5.3-00001-of-00004.gguf",
            "target_sha256": "aa",
            "drafter": { "path": "/store/glm-5.3-00001-of-00004.gguf" },
        }))
        .expect("embedded drafter artifacts deserialize");
        assert_eq!(embedded.separate_drafter(), None);

        let separate: Artifacts = serde_json::from_value(json!({
            "model": "qwen3.8-flash-next:gguf:q2",
            "target": "/store/qwen3.8.gguf",
            "target_sha256": "aa",
            "drafter": { "path": "/store/mtp-qwen3.8.gguf" },
        }))
        .expect("separate drafter artifacts deserialize");
        assert_eq!(
            separate.separate_drafter(),
            Some(Path::new("/store/mtp-qwen3.8.gguf"))
        );
    }
}
