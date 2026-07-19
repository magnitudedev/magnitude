use std::path::Path;

use icn_contracts::{ContentId, ContentIdentity, ModelComponent, ModelId};
use sha2::{Digest, Sha256};

pub fn model_id(source_kind: &str, source_location: &Path, content_id: &ContentId) -> ModelId {
    let canonical = source_location
        .canonicalize()
        .unwrap_or_else(|_| source_location.to_path_buf());
    let mut digest = Sha256::new();
    digest.update(b"magnitude-model-id-v1\0");
    digest.update(source_kind.as_bytes());
    digest.update(b"\0");
    digest.update(canonical.to_string_lossy().as_bytes());
    digest.update(b"\0");
    digest.update(content_id.0.as_bytes());
    ModelId(format!("mdl_{:x}", digest.finalize()))
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

    use icn_contracts::{ComponentRole, ContentIdentity, ModelComponent};

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
}
