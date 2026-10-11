use std::path::{Path, PathBuf};

use magnitude_service_contracts::{ContentId, ContentIdentity, InventoryEntryId, ModelComponent};
use sha2::{Digest, Sha256};

/// Canonicalize the deepest existing ancestor and append the rest, so a location has the same
/// identity before and after it is created (for example, a snapshot that is published only
/// after its entry id is assigned).
fn stable_location(location: &Path) -> PathBuf {
    let mut existing = location;
    let mut missing = Vec::new();
    loop {
        if let Ok(canonical) = existing.canonicalize() {
            return missing
                .iter()
                .rev()
                .fold(canonical, |path, component| path.join(component));
        }
        match (existing.parent(), existing.file_name()) {
            (Some(parent), Some(name)) => {
                missing.push(name);
                existing = parent;
            }
            _ => return location.to_path_buf(),
        }
    }
}

pub fn inventory_entry_id(
    source_kind: &str,
    source_location: &Path,
    content_id: &ContentId,
) -> InventoryEntryId {
    let canonical = stable_location(source_location);
    let mut digest = Sha256::new();
    digest.update(b"magnitude-model-id-v1\0");
    digest.update(source_kind.as_bytes());
    digest.update(b"\0");
    digest.update(canonical.to_string_lossy().as_bytes());
    digest.update(b"\0");
    digest.update(content_id.0.as_bytes());
    InventoryEntryId(format!("mdl_{:x}", digest.finalize()))
}

pub fn content_id(components: &[ModelComponent]) -> ContentId {
    let mut ordered = components.iter().collect::<Vec<_>>();
    ordered.sort_by(|left, right| left.path.cmp(&right.path));
    let mut digest = Sha256::new();
    digest.update(b"magnitude-content-id-v1\0");
    for component in ordered {
        digest.update(component.path.to_string_lossy().as_bytes());
        digest.update(b"\0");
        digest.update(format!("{:?}", component.role).as_bytes());
        digest.update(b"\0");
        digest.update(component.size_bytes.to_le_bytes());
        digest.update(b"\0");
        match &component.content {
            ContentIdentity::Sha256 { value } => {
                digest.update(b"sha256\0");
                digest.update(value.as_bytes());
            }
            ContentIdentity::GitOid { value } => {
                digest.update(b"git-oid\0");
                digest.update(value.as_bytes());
            }
            ContentIdentity::Xet { value } => {
                digest.update(b"xet\0");
                digest.update(value.as_bytes());
            }
            ContentIdentity::FileIdentity { value } => {
                digest.update(b"file-identity\0");
                digest.update(value.as_bytes());
            }
            ContentIdentity::Unknown => digest.update(b"unknown"),
        }
        digest.update(b"\0");
        digest.update(component.shard_index.unwrap_or(u32::MAX).to_le_bytes());
        digest.update(b"\0");
    }
    ContentId(format!("content_{:x}", digest.finalize()))
}

pub fn fingerprint(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use magnitude_service_contracts::{ComponentRole, ContentIdentity, ModelComponent};

    use super::*;

    fn component(path: &str, digest: &str) -> ModelComponent {
        ModelComponent {
            path: PathBuf::from(path),
            role: ComponentRole::Weights,
            size_bytes: 42,
            content: ContentIdentity::Sha256 {
                value: digest.to_owned(),
            },
            shard_index: None,
            relationship: None,
        }
    }

    #[test]
    fn content_identity_is_order_independent_but_content_sensitive() {
        let a = component("a.gguf", "a");
        let b = component("b.gguf", "b");
        assert_eq!(
            content_id(&[a.clone(), b.clone()]),
            content_id(&[b.clone(), a.clone()])
        );
        assert_ne!(
            content_id(&[a, b]),
            content_id(&[component("a.gguf", "different")])
        );
    }

    #[test]
    fn entry_id_is_the_same_before_and_after_its_location_exists() {
        let temporary = tempfile::tempdir().expect("temporary root");
        let snapshot = temporary
            .path()
            .join("hub/models--owner--model/snapshots/commit");
        let content = ContentId("content".to_owned());
        let before = inventory_entry_id("magnitude-cache", &snapshot, &content);
        std::fs::create_dir_all(&snapshot).expect("snapshot");
        assert_eq!(
            inventory_entry_id("magnitude-cache", &snapshot, &content),
            before
        );
    }

    #[cfg(unix)]
    #[test]
    fn entry_id_resolves_a_symlinked_root_before_its_location_exists() {
        let temporary = tempfile::tempdir().expect("temporary root");
        let real = temporary.path().join("real");
        std::fs::create_dir_all(&real).expect("real root");
        let linked = temporary.path().join("linked");
        std::os::unix::fs::symlink(&real, &linked).expect("linked root");
        let content = ContentId("content".to_owned());
        let through_link =
            inventory_entry_id("magnitude-cache", &linked.join("hub/snapshot"), &content);
        std::fs::create_dir_all(real.join("hub/snapshot")).expect("snapshot");
        assert_eq!(
            inventory_entry_id("magnitude-cache", &real.join("hub/snapshot"), &content),
            through_link
        );
    }
}
