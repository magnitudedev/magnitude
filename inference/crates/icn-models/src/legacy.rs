use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use icn_contracts::{
    ComponentRelationship, ComponentRole, ContentIdentity, InventoryError, ModelComponent,
};
use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::identity::{content_id, model_id};
use crate::inventory::{hf_repo_dir, now};
use crate::manifest::{MANIFEST_VERSION, ManagedManifest};
use crate::validation::validate_relative_path;

const IMPORT_MARKER: &str = "legacy-v1-imported";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyManifest {
    version: u32,
    artifact: LegacyArtifact,
    files: Vec<LegacyCachedFile>,
    #[serde(rename = "installedAt")]
    _installed_at: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyArtifact {
    id: String,
    repository: String,
    #[serde(rename = "requestedRevision")]
    requested_revision: String,
    commit: String,
    files: Vec<LegacyArtifactFile>,
    relationships: Vec<LegacyRelationship>,
    #[serde(rename = "totalBytes")]
    total_bytes: u64,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct LegacyArtifactFile {
    path: PathBuf,
    role: LegacyRole,
    #[serde(rename = "shardIndex", default)]
    shard_index: Option<u32>,
    #[serde(rename = "sizeBytes")]
    size_bytes: u64,
    content: LegacyContent,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyCachedFile {
    path: PathBuf,
    role: LegacyRole,
    #[serde(rename = "shardIndex", default)]
    shard_index: Option<u32>,
    #[serde(rename = "sizeBytes")]
    size_bytes: u64,
    content: LegacyContent,
    #[serde(rename = "snapshotRelativePath")]
    snapshot_relative_path: PathBuf,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum LegacyRole {
    Primary,
    Shard,
    Projector,
    Auxiliary,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(tag = "_tag")]
enum LegacyContent {
    LfsSha256 { sha256: String },
    Xet { hash: String },
    Git { oid: String },
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LegacyRelationship {
    kind: String,
    #[serde(rename = "fromPath")]
    from_path: PathBuf,
    #[serde(rename = "toPath")]
    to_path: PathBuf,
}

/// Import valid TypeScript v1 installations without ever modifying the old store.
/// Invalid manifests are intentionally skipped; the marker records a complete scan, not that every
/// input was accepted.
pub(crate) fn import_v1_store(legacy: &Path, destination: &Path) -> Result<(), InventoryError> {
    let marker = destination.join(IMPORT_MARKER);
    if marker.is_file() || !legacy.is_dir() {
        return Ok(());
    }
    let lock_path = destination.join("locks/legacy-import.lock");
    let lock = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(lock_path)
        .map_err(io_error)?;
    fs2::FileExt::lock_exclusive(&lock).map_err(io_error)?;
    if marker.is_file() {
        return Ok(());
    }

    let cache = legacy.join("cache");
    let installations = legacy.join("installations");
    if installations.is_dir() {
        let mut entries = fs::read_dir(&installations)
            .map_err(io_error)?
            .filter_map(Result::ok)
            .collect::<Vec<_>>();
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            let path = entry.path();
            if path.extension().and_then(|value| value.to_str()) != Some("json") {
                continue;
            }
            let Some(manifest) = fs::read(&path)
                .ok()
                .and_then(|bytes| serde_json::from_slice::<LegacyManifest>(&bytes).ok())
            else {
                continue;
            };
            let _ = import_manifest(&cache, destination, manifest);
        }
    }
    atomic_write(&marker, b"1\n")?;
    fs2::FileExt::unlock(&lock).map_err(io_error)?;
    Ok(())
}

fn import_manifest(
    legacy_cache: &Path,
    destination: &Path,
    manifest: LegacyManifest,
) -> Result<(), InventoryError> {
    if manifest.version != 1
        || manifest.artifact.id.is_empty()
        || manifest.artifact.repository.split_once('/').is_none()
        || manifest.artifact.commit.is_empty()
        || manifest.files.is_empty()
        || manifest.files.len() != manifest.artifact.files.len()
        || manifest.artifact.total_bytes
            != manifest
                .artifact
                .files
                .iter()
                .map(|file| file.size_bytes)
                .sum::<u64>()
    {
        return Err(InventoryError::InvalidRequest(
            "invalid legacy installation manifest".to_owned(),
        ));
    }
    for expected in &manifest.artifact.files {
        validate_relative_path(&expected.path)?;
        let Some(actual) = manifest
            .files
            .iter()
            .find(|file| file.path == expected.path)
        else {
            return Err(InventoryError::InvalidRequest(
                "legacy installation file set does not match its artifact".to_owned(),
            ));
        };
        if actual.role != expected.role
            || actual.shard_index != expected.shard_index
            || actual.size_bytes != expected.size_bytes
            || actual.content != expected.content
        {
            return Err(InventoryError::InvalidRequest(
                "legacy installation file metadata does not match its artifact".to_owned(),
            ));
        }
    }

    let repo_root = destination
        .join("hub")
        .join(hf_repo_dir(&manifest.artifact.repository));
    let snapshot = repo_root.join("snapshots").join(&manifest.artifact.commit);
    let legacy_root = legacy_cache.canonicalize().map_err(io_error)?;
    let relationships = &manifest.artifact.relationships;
    let mut components = Vec::with_capacity(manifest.files.len());
    for file in &manifest.files {
        validate_relative_path(&file.path)?;
        validate_relative_path(&file.snapshot_relative_path)?;
        let source = legacy_cache.join(&file.snapshot_relative_path);
        let canonical_source = source.canonicalize().map_err(io_error)?;
        if !canonical_source.starts_with(&legacy_root) {
            return Err(InventoryError::InvalidRequest(
                "legacy snapshot target escapes its cache".to_owned(),
            ));
        }
        validate_file(&canonical_source, file.size_bytes, &file.content)?;
        let content = content_identity(&file.content);
        let blob = repo_root.join("blobs").join(content_key(&file.content));
        publish_blob(&canonical_source, &blob, file.size_bytes, &file.content)?;
        publish_snapshot_link(
            &blob,
            &snapshot.join(&file.path),
            file.size_bytes,
            &file.content,
        )?;
        components.push(ModelComponent {
            path: file.path.clone(),
            role: role(file.role, file.shard_index),
            size_bytes: file.size_bytes,
            content,
            shard_index: file.shard_index,
            relationship: relationships.iter().find_map(|relationship| {
                (relationship.kind == "projector-for"
                    && (relationship.from_path == file.path || relationship.to_path == file.path))
                    .then(|| ComponentRelationship::ProjectorFor {
                        projector: relationship.from_path.clone(),
                        model: relationship.to_path.clone(),
                    })
            }),
        });
    }
    let content_id = content_id(&components);
    let model_id = model_id("magnitude-cache", &snapshot, &content_id);
    let published = ManagedManifest {
        version: MANIFEST_VERSION,
        model_id: model_id.clone(),
        content_id,
        repository: manifest.artifact.repository,
        requested_revision: manifest.artifact.requested_revision,
        commit: manifest.artifact.commit,
        components,
        created_at: now(),
        ready_at: now(),
    };
    let bytes = serde_json::to_vec_pretty(&published)
        .map_err(|error| InventoryError::Internal(error.to_string()))?;
    atomic_write(
        &destination
            .join("installations")
            .join(format!("{}.json", model_id.0)),
        &bytes,
    )
}

fn role(role: LegacyRole, shard: Option<u32>) -> ComponentRole {
    match role {
        LegacyRole::Primary if shard.is_some() => ComponentRole::Shard,
        LegacyRole::Primary => ComponentRole::Weights,
        LegacyRole::Shard => ComponentRole::Shard,
        LegacyRole::Projector => ComponentRole::Projector,
        LegacyRole::Auxiliary => ComponentRole::Auxiliary,
    }
}

fn content_identity(content: &LegacyContent) -> ContentIdentity {
    match content {
        LegacyContent::LfsSha256 { sha256 } => ContentIdentity::Sha256 {
            value: sha256.to_ascii_lowercase(),
        },
        LegacyContent::Xet { hash } => ContentIdentity::Xet {
            value: hash.clone(),
        },
        LegacyContent::Git { oid } => ContentIdentity::GitOid { value: oid.clone() },
    }
}

fn content_key(content: &LegacyContent) -> &str {
    match content {
        LegacyContent::LfsSha256 { sha256 } => sha256,
        LegacyContent::Xet { hash } => hash,
        LegacyContent::Git { oid } => oid,
    }
}

fn validate_file(
    path: &Path,
    expected_size: u64,
    content: &LegacyContent,
) -> Result<(), InventoryError> {
    let metadata = path.metadata().map_err(io_error)?;
    if !metadata.is_file() || metadata.len() != expected_size {
        return Err(InventoryError::Io(
            "legacy cached file does not match its declared size".to_owned(),
        ));
    }
    if let LegacyContent::LfsSha256 { sha256 } = content {
        let mut file = fs::File::open(path).map_err(io_error)?;
        let mut digest = Sha256::new();
        let mut buffer = vec![0_u8; 1024 * 1024];
        loop {
            let count = file.read(&mut buffer).map_err(io_error)?;
            if count == 0 {
                break;
            }
            digest.update(&buffer[..count]);
        }
        let actual = format!("{:x}", digest.finalize());
        if actual != sha256.to_ascii_lowercase() {
            return Err(InventoryError::Io(
                "legacy cached file does not match its LFS digest".to_owned(),
            ));
        }
    }
    Ok(())
}

fn publish_blob(
    source: &Path,
    destination: &Path,
    expected_size: u64,
    content: &LegacyContent,
) -> Result<(), InventoryError> {
    if destination.is_file() {
        return validate_file(destination, expected_size, content);
    }
    let parent = destination
        .parent()
        .ok_or_else(|| InventoryError::Io("blob destination has no parent".to_owned()))?;
    fs::create_dir_all(parent).map_err(io_error)?;
    let temporary = destination.with_extension(format!("import-{}", std::process::id()));
    if fs::hard_link(source, &temporary).is_err() {
        fs::copy(source, &temporary).map_err(io_error)?;
    }
    if let Err(error) = validate_file(&temporary, expected_size, content) {
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }
    match fs::rename(&temporary, destination) {
        Ok(()) => Ok(()),
        Err(_) if destination.is_file() => {
            let _ = fs::remove_file(&temporary);
            validate_file(destination, expected_size, content)
        }
        Err(error) => {
            let _ = fs::remove_file(&temporary);
            Err(io_error(error))
        }
    }
}

fn publish_snapshot_link(
    blob: &Path,
    pointer: &Path,
    expected_size: u64,
    content: &LegacyContent,
) -> Result<(), InventoryError> {
    if pointer.exists() {
        return validate_file(pointer, expected_size, content);
    }
    fs::create_dir_all(
        pointer
            .parent()
            .ok_or_else(|| InventoryError::Io("snapshot pointer has no parent".to_owned()))?,
    )
    .map_err(io_error)?;
    fs::hard_link(blob, pointer).map_err(io_error)
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), InventoryError> {
    if path.is_file() {
        return Ok(());
    }
    let parent = path
        .parent()
        .ok_or_else(|| InventoryError::Io("publication path has no parent".to_owned()))?;
    fs::create_dir_all(parent).map_err(io_error)?;
    let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
    fs::write(&temporary, bytes).map_err(io_error)?;
    fs::rename(temporary, path).map_err(io_error)
}

fn io_error(error: impl std::fmt::Display) -> InventoryError {
    InventoryError::Io(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    fn layout(root: &Path) {
        for directory in ["hub", "installations", "locks"] {
            fs::create_dir_all(root.join(directory)).unwrap();
        }
    }

    #[test]
    fn imports_a_valid_v1_installation_once_without_removing_the_source() {
        let source = TempDir::new().unwrap();
        let destination = TempDir::new().unwrap();
        layout(destination.path());
        let bytes = b"GGUF-test-model";
        let digest = format!("{:x}", Sha256::digest(bytes));
        let relative = "models--owner--repo/snapshots/commit/model.gguf";
        let pointer = source.path().join("cache").join(relative);
        fs::create_dir_all(pointer.parent().unwrap()).unwrap();
        fs::write(&pointer, bytes).unwrap();
        let manifest = json!({
            "version": 1,
            "artifact": {
                "id": format!("hf_{}", "a".repeat(64)),
                "repository": "owner/repo",
                "requestedRevision": "main",
                "commit": "commit",
                "files": [{
                    "path": "model.gguf",
                    "role": "primary",
                    "sizeBytes": bytes.len(),
                    "content": {"_tag": "LfsSha256", "sha256": digest}
                }],
                "relationships": [],
                "totalBytes": bytes.len()
            },
            "files": [{
                "path": "model.gguf",
                "role": "primary",
                "sizeBytes": bytes.len(),
                "content": {"_tag": "LfsSha256", "sha256": digest},
                "snapshotRelativePath": relative
            }],
            "installedAt": "2026-07-18T00:00:00.000Z"
        });
        let installations = source.path().join("installations");
        fs::create_dir_all(&installations).unwrap();
        fs::write(
            installations.join("legacy.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();

        import_v1_store(source.path(), destination.path()).unwrap();
        import_v1_store(source.path(), destination.path()).unwrap();

        assert!(
            pointer.is_file(),
            "the importer must leave the old store intact"
        );
        assert!(destination.path().join(IMPORT_MARKER).is_file());
        assert_eq!(
            fs::read_dir(destination.path().join("installations"))
                .unwrap()
                .count(),
            1
        );
        assert_eq!(
            fs::read(
                destination
                    .path()
                    .join("hub/models--owner--repo/snapshots/commit/model.gguf")
            )
            .unwrap(),
            bytes
        );
    }

    #[test]
    fn skips_invalid_v1_manifests_but_completes_the_bounded_scan() {
        let source = TempDir::new().unwrap();
        let destination = TempDir::new().unwrap();
        layout(destination.path());
        fs::create_dir_all(source.path().join("installations")).unwrap();
        fs::write(source.path().join("installations/bad.json"), b"not-json").unwrap();

        import_v1_store(source.path(), destination.path()).unwrap();

        assert!(destination.path().join(IMPORT_MARKER).is_file());
        assert_eq!(
            fs::read_dir(destination.path().join("installations"))
                .unwrap()
                .count(),
            0
        );
    }
}
