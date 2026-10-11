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
    pub target_sha256: String,
    /// The catalog drafter, `null` when the catalog declares none.
    pub drafter: Option<Drafter>,
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
