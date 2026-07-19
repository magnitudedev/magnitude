use std::collections::BTreeSet;
use std::path::{Component, Path};

use icn_contracts::{
    ComponentRelationship, ComponentRole, DownloadModelRequest, HuggingFaceDownloadSource,
    InventoryError,
};

const MAX_COMPONENTS: usize = 128;
const MAX_PATH_BYTES: usize = 1_024;
const MAX_REPOSITORY_BYTES: usize = 256;
const MAX_REVISION_BYTES: usize = 256;

pub fn validate_download_request(request: &DownloadModelRequest) -> Result<(), InventoryError> {
    let HuggingFaceDownloadSource::HuggingFace {
        repository,
        revision,
    } = &request.source;
    validate_repository(repository)?;
    if revision.is_empty() || revision.len() > MAX_REVISION_BYTES || revision.contains('\0') {
        return Err(InventoryError::InvalidRequest(
            "revision must be non-empty, bounded, and contain no NUL byte".to_owned(),
        ));
    }
    if request.components.is_empty() || request.components.len() > MAX_COMPONENTS {
        return Err(InventoryError::InvalidRequest(format!(
            "components must contain between 1 and {MAX_COMPONENTS} entries"
        )));
    }
    let mut paths = BTreeSet::new();
    let mut shard_indices = BTreeSet::new();
    let mut has_weights = false;
    for component in &request.components {
        validate_relative_path(&component.path)?;
        if !paths.insert(component.path.clone()) {
            return Err(InventoryError::InvalidRequest(format!(
                "duplicate component path: {}",
                component.path.display()
            )));
        }
        match component.role {
            ComponentRole::Weights => {
                has_weights = true;
                if component.shard_index.is_some() {
                    return Err(InventoryError::InvalidRequest(
                        "shard_index is valid only for shard components".to_owned(),
                    ));
                }
            }
            ComponentRole::Shard => {
                has_weights = true;
                let index = component.shard_index.ok_or_else(|| {
                    InventoryError::InvalidRequest(
                        "shard components require shard_index".to_owned(),
                    )
                })?;
                if !shard_indices.insert(index) {
                    return Err(InventoryError::InvalidRequest(format!(
                        "duplicate shard_index: {index}"
                    )));
                }
            }
            ComponentRole::Projector
            | ComponentRole::Auxiliary
            | ComponentRole::Draft
            | ComponentRole::Mtp => {
                if component.shard_index.is_some() {
                    return Err(InventoryError::InvalidRequest(
                        "shard_index is valid only for shard components".to_owned(),
                    ));
                }
            }
        }
        if let Some(digest) = component.expected_sha256.as_deref()
            && (digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()))
        {
            return Err(InventoryError::InvalidRequest(format!(
                "invalid SHA-256 for {}",
                component.path.display()
            )));
        }
    }
    if !has_weights {
        return Err(InventoryError::InvalidRequest(
            "at least one weights or shard component is required".to_owned(),
        ));
    }
    for relationship in &request.relationships {
        let (component, target, expected_role) = match relationship {
            ComponentRelationship::ProjectorFor { projector, model } => {
                (projector, model, ComponentRole::Projector)
            }
            ComponentRelationship::DraftFor { draft, model } => {
                (draft, model, ComponentRole::Draft)
            }
            ComponentRelationship::MtpFor { mtp, model } => (mtp, model, ComponentRole::Mtp),
        };
        if !paths.contains(component) || !paths.contains(target) {
            return Err(InventoryError::InvalidRequest(
                "component relationship references an unselected path".to_owned(),
            ));
        }
        if !request
            .components
            .iter()
            .any(|candidate| candidate.path == *component && candidate.role == expected_role)
            || !request.components.iter().any(|candidate| {
                candidate.path == *target
                    && matches!(
                        candidate.role,
                        ComponentRole::Weights | ComponentRole::Shard
                    )
            })
        {
            return Err(InventoryError::InvalidRequest(
                "component relationship roles do not match their selected paths".to_owned(),
            ));
        }
    }
    Ok(())
}

pub fn validate_relative_path(path: &Path) -> Result<(), InventoryError> {
    let rendered = path.to_string_lossy();
    if rendered.is_empty() || rendered.len() > MAX_PATH_BYTES || rendered.contains('\0') {
        return Err(InventoryError::InvalidRequest(
            "component path must be non-empty, bounded, and contain no NUL byte".to_owned(),
        ));
    }
    if path.components().any(|component| {
        matches!(
            component,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        )
    }) {
        return Err(InventoryError::InvalidRequest(format!(
            "unsafe component path: {}",
            path.display()
        )));
    }
    Ok(())
}

pub(crate) fn validate_repository(repository: &str) -> Result<(), InventoryError> {
    if repository.is_empty()
        || repository.len() > MAX_REPOSITORY_BYTES
        || repository.contains(['\0', '\\'])
    {
        return Err(InventoryError::InvalidRequest(
            "invalid Hugging Face repository".to_owned(),
        ));
    }
    let mut parts = repository.split('/');
    let owner = parts.next().unwrap_or_default();
    let name = parts.next().unwrap_or_default();
    if owner.is_empty()
        || name.is_empty()
        || parts.next().is_some()
        || owner == "."
        || owner == ".."
        || name == "."
        || name == ".."
    {
        return Err(InventoryError::InvalidRequest(
            "repository must be exactly owner/name".to_owned(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use icn_contracts::{DownloadComponent, HuggingFaceDownloadSource};

    use super::*;

    fn request(path: &str) -> DownloadModelRequest {
        DownloadModelRequest {
            source: HuggingFaceDownloadSource::HuggingFace {
                repository: "owner/repo".to_owned(),
                revision: "main".to_owned(),
            },
            components: vec![DownloadComponent {
                path: PathBuf::from(path),
                role: ComponentRole::Weights,
                shard_index: None,
                expected_sha256: None,
            }],
            relationships: Vec::new(),
        }
    }

    #[test]
    fn rejects_traversal_and_absolute_component_paths() {
        assert!(validate_download_request(&request("../model.gguf")).is_err());
        assert!(validate_download_request(&request("/tmp/model.gguf")).is_err());
        assert!(validate_download_request(&request("models/model.gguf")).is_ok());
    }

    #[test]
    fn rejects_duplicate_paths_and_missing_weights() {
        let mut duplicate = request("model.gguf");
        duplicate.components.push(duplicate.components[0].clone());
        assert!(validate_download_request(&duplicate).is_err());
        let mut projector = request("projector.gguf");
        projector.components[0].role = ComponentRole::Projector;
        assert!(validate_download_request(&projector).is_err());
    }
}
