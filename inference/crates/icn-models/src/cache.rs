use std::path::{Path, PathBuf};
use std::{fs, io};

use getrandom::fill;
use icn_contracts::{ContentId, HardwareAssessment, ModelExecutionAssessment};
use icn_utils::file_cache::{
    read_bytes, read_json, read_object, write_bytes_atomic, write_json_atomic,
};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};

const MAX_INDEX_BYTES: usize = 64 * 1024 * 1024;
const MAX_BLOB_BYTES: usize = 256 * 1024 * 1024;

#[derive(Clone, Copy, Debug)]
pub enum ModelIndexKind {
    Artifact,
    ArtifactInspection,
    HardwareAssessment,
    ExecutionAssessment,
    OfferingAssessment,
}

impl ModelIndexKind {
    fn relative(self) -> &'static str {
        match self {
            Self::Artifact => "artifacts",
            Self::ArtifactInspection => "inspections/artifacts",
            Self::HardwareAssessment => "assessments/hardware",
            Self::ExecutionAssessment => "assessments/execution",
            Self::OfferingAssessment => "assessments/offerings",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ModelBlobKind {
    GgufHeader,
}

impl ModelBlobKind {
    fn relative(self) -> &'static str {
        match self {
            Self::GgufHeader => "gguf-headers",
        }
    }
}

/// The sole filesystem namespace owner for recomputable model data.
#[derive(Clone, Debug)]
pub struct ModelCache {
    root: PathBuf,
}

pub struct ModelCacheWorkspace {
    path: PathBuf,
    temporary: Option<tempfile::TempDir>,
}

impl ModelCacheWorkspace {
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for ModelCacheWorkspace {
    fn drop(&mut self) {
        if self.temporary.is_none() {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

impl ModelCache {
    #[must_use]
    pub fn new(model_store_root: &Path) -> Self {
        Self {
            root: model_store_root.join("cache"),
        }
    }

    pub(crate) fn read_inventory(&self) -> Option<Map<String, Value>> {
        read_object(&self.inventory_path(), MAX_INDEX_BYTES)
    }

    pub(crate) fn write_inventory<T: Serialize>(&self, value: &T) {
        write_json_atomic(
            &self.inventory_path(),
            &self.lock_path("inventory"),
            value,
            MAX_INDEX_BYTES,
        );
    }

    pub fn read_index<T: DeserializeOwned>(
        &self,
        kind: ModelIndexKind,
        evidence: &str,
    ) -> Option<T> {
        read_json(&self.index_path(kind, evidence), MAX_INDEX_BYTES)
    }

    pub fn write_index<T: Serialize>(&self, kind: ModelIndexKind, evidence: &str, value: &T) {
        let digest = evidence_digest(evidence);
        write_json_atomic(
            &self.index_path_for_digest(kind, &digest),
            &self.lock_path(&format!("index-{digest}")),
            value,
            MAX_INDEX_BYTES,
        );
    }

    pub fn read_hardware_assessment(
        &self,
        content_id: &ContentId,
        hardware_evidence: &str,
    ) -> Option<HardwareAssessment> {
        self.read_index::<HardwareAssessment>(
            ModelIndexKind::HardwareAssessment,
            &hardware_assessment_evidence(content_id, hardware_evidence),
        )
        .filter(is_terminal_assessment)
    }

    pub fn write_hardware_assessment(
        &self,
        content_id: &ContentId,
        hardware_evidence: &str,
        assessment: &HardwareAssessment,
    ) {
        if is_terminal_assessment(assessment) {
            self.write_index(
                ModelIndexKind::HardwareAssessment,
                &hardware_assessment_evidence(content_id, hardware_evidence),
                assessment,
            );
        }
    }

    pub fn read_execution_assessment(
        &self,
        content_id: &ContentId,
        execution_evidence: &str,
    ) -> Option<ModelExecutionAssessment> {
        self.read_index::<ModelExecutionAssessment>(
            ModelIndexKind::ExecutionAssessment,
            &hardware_assessment_evidence(content_id, execution_evidence),
        )
        .filter(|assessment| is_terminal_assessment(&assessment.hardware))
    }

    pub fn write_execution_assessment(
        &self,
        content_id: &ContentId,
        execution_evidence: &str,
        assessment: &ModelExecutionAssessment,
    ) {
        if is_terminal_assessment(&assessment.hardware) {
            self.write_index(
                ModelIndexKind::ExecutionAssessment,
                &hardware_assessment_evidence(content_id, execution_evidence),
                assessment,
            );
        }
    }

    pub fn read_blob(&self, kind: ModelBlobKind, digest: &str) -> Option<Vec<u8>> {
        valid_digest(digest)
            .then(|| read_bytes(&self.blob_path(kind, digest), MAX_BLOB_BYTES))
            .flatten()
            .filter(|bytes| hex_sha256(bytes) == digest)
    }

    pub fn write_blob(&self, kind: ModelBlobKind, digest: &str, bytes: &[u8]) {
        if !valid_digest(digest) || hex_sha256(bytes) != digest {
            return;
        }
        write_bytes_atomic(
            &self.blob_path(kind, digest),
            &self.lock_path(&format!("blob-{digest}")),
            bytes,
            MAX_BLOB_BYTES,
        );
    }

    pub fn workspace(&self) -> io::Result<ModelCacheWorkspace> {
        let mut random = [0_u8; 16];
        fill(&mut random).map_err(io::Error::other)?;
        let name = random
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let path = self.root.join(".work").join(name);
        if fs::create_dir_all(&path).is_ok() {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                if fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).is_err() {
                    let _ = fs::remove_dir_all(&path);
                } else {
                    return Ok(ModelCacheWorkspace {
                        path,
                        temporary: None,
                    });
                }
            }
            #[cfg(not(unix))]
            return Ok(ModelCacheWorkspace {
                path,
                temporary: None,
            });
        }
        let temporary = tempfile::Builder::new()
            .prefix("magnitude-model-preview-")
            .tempdir()?;
        Ok(ModelCacheWorkspace {
            path: temporary.path().to_path_buf(),
            temporary: Some(temporary),
        })
    }

    fn inventory_path(&self) -> PathBuf {
        self.root.join("indexes/inventory.json")
    }

    fn index_path(&self, kind: ModelIndexKind, evidence: &str) -> PathBuf {
        self.index_path_for_digest(kind, &evidence_digest(evidence))
    }

    fn index_path_for_digest(&self, kind: ModelIndexKind, digest: &str) -> PathBuf {
        self.root
            .join("indexes")
            .join(kind.relative())
            .join(format!("{digest}.json"))
    }

    fn blob_path(&self, kind: ModelBlobKind, digest: &str) -> PathBuf {
        self.root.join("blobs").join(kind.relative()).join(digest)
    }

    fn lock_path(&self, name: &str) -> PathBuf {
        self.root.join(".locks").join(format!("{name}.lock"))
    }
}

fn evidence_digest(evidence: &str) -> String {
    hex_sha256(evidence.as_bytes())
}

fn hardware_assessment_evidence(content_id: &ContentId, hardware_evidence: &str) -> String {
    format!("{}:{hardware_evidence}", content_id.0)
}

fn is_terminal_assessment(assessment: &HardwareAssessment) -> bool {
    matches!(
        assessment,
        HardwareAssessment::Fits { .. }
            | HardwareAssessment::DoesNotFit { .. }
            | HardwareAssessment::InvalidArtifact { .. }
            | HardwareAssessment::IncompatibleArtifact { .. }
    )
}

fn hex_sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn namespaces_are_typed_and_fail_as_misses() {
        let directory = tempfile::tempdir().unwrap();
        let cache = ModelCache::new(directory.path());

        cache.write_index(
            ModelIndexKind::ArtifactInspection,
            "template evidence",
            &42_u64,
        );
        assert_eq!(
            cache.read_index::<u64>(ModelIndexKind::ArtifactInspection, "template evidence"),
            Some(42)
        );
        assert_eq!(
            cache.read_index::<u64>(ModelIndexKind::HardwareAssessment, "template evidence"),
            None
        );

        let content_id = ContentId("artifact".to_owned());
        let assessment = serde_json::from_value::<HardwareAssessment>(serde_json::json!({
            "type": "fits",
            "profile": {
                "context_length": 4096,
                "acceleration": "cpu",
                "device": "system"
            },
            "memory": {
                "required_bytes": 1,
                "available_bytes": 2,
                "headroom_bytes": 1,
                "domains": []
            },
            "recommendation": "recommended"
        }))
        .unwrap();
        cache.write_hardware_assessment(&content_id, "hardware", &assessment);
        assert_eq!(
            cache.read_hardware_assessment(&content_id, "hardware"),
            Some(assessment)
        );
        assert!(
            cache
                .read_hardware_assessment(&content_id, "different-hardware")
                .is_none()
        );

        let bytes = b"header";
        let digest = hex_sha256(bytes);
        cache.write_blob(ModelBlobKind::GgufHeader, &digest, bytes);
        assert_eq!(
            cache
                .read_blob(ModelBlobKind::GgufHeader, &digest)
                .as_deref(),
            Some(bytes.as_slice())
        );
        cache.write_blob(ModelBlobKind::GgufHeader, &"0".repeat(64), bytes);
        assert!(
            cache
                .read_blob(ModelBlobKind::GgufHeader, &"0".repeat(64))
                .is_none()
        );

        let index_path = cache.index_path(ModelIndexKind::ArtifactInspection, "template evidence");
        fs::write(&index_path, b"corrupt").unwrap();
        assert_eq!(
            cache.read_index::<u64>(ModelIndexKind::ArtifactInspection, "template evidence"),
            None
        );
        cache.write_index(
            ModelIndexKind::ArtifactInspection,
            "template evidence",
            &7_u64,
        );
        assert_eq!(
            cache.read_index::<u64>(ModelIndexKind::ArtifactInspection, "template evidence"),
            Some(7)
        );

        let blob_path = cache.blob_path(ModelBlobKind::GgufHeader, &digest);
        fs::write(&blob_path, b"corrupt").unwrap();
        assert!(
            cache
                .read_blob(ModelBlobKind::GgufHeader, &digest)
                .is_none()
        );
    }

    #[test]
    fn workspaces_are_private_and_removed_on_drop() {
        let directory = tempfile::tempdir().unwrap();
        let cache = ModelCache::new(directory.path());
        let path = {
            let workspace = cache.workspace().unwrap();
            let path = workspace.path().to_path_buf();
            assert!(path.is_dir());
            path
        };
        assert!(!path.exists());
    }

    #[test]
    fn workspace_falls_back_when_the_cache_root_is_unusable() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(directory.path().join("cache"), b"not a directory").unwrap();
        let cache = ModelCache::new(directory.path());
        let path = {
            let workspace = cache.workspace().unwrap();
            let path = workspace.path().to_path_buf();
            assert!(path.is_dir());
            assert!(!path.starts_with(directory.path()));
            path
        };
        assert!(!path.exists());
    }
}
