use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path};

use icn_contracts::models::{
    ModelFile, ModelFileRelationship, ModelFileRole, ModelPackage, ModelPackageSource,
};
use icn_contracts::{
    ComponentRelationship, ComponentRole, ContentIdentity, InventoryError, ModelComponent,
};

const MAX_COMPONENTS: usize = 128;
const MAX_PATH_BYTES: usize = 1_024;
const MAX_REPOSITORY_BYTES: usize = 256;
const MAX_REVISION_BYTES: usize = 256;

pub(crate) struct ValidatedDownloadPackage {
    repository: String,
    revision: String,
    components: Vec<ModelComponent>,
}

impl ValidatedDownloadPackage {
    pub(crate) fn new(package: ModelPackage) -> Result<Self, InventoryError> {
        validate_download_package(&package)?;
        let components = package
            .files
            .iter()
            .map(|file| {
                let shard_index = package.relationships.iter().find_map(|relationship| {
                    if let ModelFileRelationship::Shard { file_id, index, .. } = relationship
                        && file_id == &file.id
                    {
                        Some(*index)
                    } else {
                        None
                    }
                });
                ModelComponent {
                    path: file.path.clone(),
                    role: match (&file.role, shard_index) {
                        (ModelFileRole::Weights, Some(_)) => ComponentRole::Shard,
                        (ModelFileRole::Weights, None) => ComponentRole::Weights,
                        (ModelFileRole::Projector, _) => ComponentRole::Projector,
                        (ModelFileRole::Draft, _) => ComponentRole::Draft,
                        (ModelFileRole::Mtp, _) => ComponentRole::Mtp,
                        (ModelFileRole::Auxiliary, _) => ComponentRole::Auxiliary,
                    },
                    size_bytes: file.size_bytes,
                    content: ContentIdentity::Sha256 {
                        value: file.sha256.to_ascii_lowercase(),
                    },
                    shard_index,
                    relationship: component_relationship(&package, file),
                }
            })
            .collect();
        let ModelPackageSource::HuggingFace {
            repository,
            revision,
        } = package.source
        else {
            unreachable!("validated download packages have a Hugging Face source")
        };
        Ok(Self {
            repository,
            revision,
            components,
        })
    }

    pub(crate) fn repository_revision(&self) -> (&str, &str) {
        (&self.repository, &self.revision)
    }

    pub(crate) fn components(&self) -> &[ModelComponent] {
        &self.components
    }

    pub(crate) fn into_parts(self) -> (String, String, Vec<ModelComponent>) {
        (self.repository, self.revision, self.components)
    }
}

fn validate_download_package(package: &ModelPackage) -> Result<(), InventoryError> {
    let ModelPackageSource::HuggingFace {
        repository,
        revision,
    } = &package.source
    else {
        return Err(InventoryError::Unsupported(
            "only exact Hugging Face packages can be downloaded".to_owned(),
        ));
    };
    validate_repository(repository)?;
    if revision.is_empty() || revision.len() > MAX_REVISION_BYTES || revision.contains('\0') {
        return Err(InventoryError::InvalidRequest(
            "revision must be non-empty, bounded, and contain no NUL byte".to_owned(),
        ));
    }
    if package.files.is_empty() || package.files.len() > MAX_COMPONENTS {
        return Err(InventoryError::InvalidRequest(format!(
            "package files must contain between 1 and {MAX_COMPONENTS} entries"
        )));
    }

    let mut files_by_id = BTreeMap::new();
    let mut paths = BTreeSet::new();
    for file in &package.files {
        validate_relative_path(&file.path)?;
        if !paths.insert(file.path.clone()) {
            return Err(InventoryError::InvalidRequest(format!(
                "duplicate package file path: {}",
                file.path.display()
            )));
        }
        if files_by_id.insert(file.id.clone(), file).is_some() {
            return Err(InventoryError::InvalidRequest(format!(
                "duplicate package file id: {}",
                file.id.0
            )));
        }
        if file.sha256.len() != 64 || !file.sha256.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(InventoryError::InvalidRequest(format!(
                "invalid SHA-256 for {}",
                file.path.display()
            )));
        }
    }
    if !package
        .files
        .iter()
        .any(|file| file.role == ModelFileRole::Weights)
    {
        return Err(InventoryError::InvalidRequest(
            "package must contain at least one weights file".to_owned(),
        ));
    }

    let mut shard_indices = BTreeSet::new();
    for relationship in &package.relationships {
        match relationship {
            ModelFileRelationship::Shard {
                file_id,
                index,
                count: _,
            } => {
                require_role(&files_by_id, file_id, ModelFileRole::Weights, "shard")?;
                if !shard_indices.insert(*index) {
                    return Err(InventoryError::InvalidRequest(format!(
                        "duplicate shard index: {index}"
                    )));
                }
            }
            ModelFileRelationship::ProjectorFor {
                projector_file_id,
                weights_file_id,
            } => {
                require_role(
                    &files_by_id,
                    projector_file_id,
                    ModelFileRole::Projector,
                    "projector",
                )?;
                require_role(
                    &files_by_id,
                    weights_file_id,
                    ModelFileRole::Weights,
                    "projector target",
                )?;
            }
            ModelFileRelationship::MtpFor {
                mtp_file_id,
                weights_file_id,
            } => {
                require_role(&files_by_id, mtp_file_id, ModelFileRole::Mtp, "MTP")?;
                require_role(
                    &files_by_id,
                    weights_file_id,
                    ModelFileRole::Weights,
                    "MTP target",
                )?;
            }
            ModelFileRelationship::DraftFor {
                draft_file_id,
                weights_file_id,
                ..
            } => {
                require_role(&files_by_id, draft_file_id, ModelFileRole::Draft, "draft")?;
                require_role(
                    &files_by_id,
                    weights_file_id,
                    ModelFileRole::Weights,
                    "draft target",
                )?;
            }
        }
    }
    Ok(())
}

fn component_relationship(
    package: &ModelPackage,
    file: &ModelFile,
) -> Option<ComponentRelationship> {
    let path = |id: &icn_contracts::models::ModelFileId| {
        package
            .files
            .iter()
            .find(|candidate| &candidate.id == id)
            .expect("validated package relationships reference existing files")
            .path
            .clone()
    };
    package
        .relationships
        .iter()
        .find_map(|relationship| match relationship {
            ModelFileRelationship::ProjectorFor {
                projector_file_id,
                weights_file_id,
            } if projector_file_id == &file.id => Some(ComponentRelationship::ProjectorFor {
                projector: file.path.clone(),
                model: path(weights_file_id),
            }),
            ModelFileRelationship::MtpFor {
                mtp_file_id,
                weights_file_id,
            } if mtp_file_id == &file.id => Some(ComponentRelationship::MtpFor {
                mtp: file.path.clone(),
                model: path(weights_file_id),
            }),
            ModelFileRelationship::DraftFor {
                draft_file_id,
                weights_file_id,
                method,
            } if draft_file_id == &file.id => Some(ComponentRelationship::DraftFor {
                draft: file.path.clone(),
                model: path(weights_file_id),
                method: method.clone(),
            }),
            _ => None,
        })
}

fn require_role(
    files: &BTreeMap<icn_contracts::models::ModelFileId, &ModelFile>,
    id: &icn_contracts::models::ModelFileId,
    role: ModelFileRole,
    relationship: &str,
) -> Result<(), InventoryError> {
    let file = files.get(id).ok_or_else(|| {
        InventoryError::InvalidRequest(format!(
            "{relationship} relationship references an unknown file"
        ))
    })?;
    if file.role != role {
        return Err(InventoryError::InvalidRequest(format!(
            "{relationship} relationship references a file with the wrong role"
        )));
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

    use icn_contracts::models::{ModelFileId, ModelPackageId, ModelPackageProperties};

    use super::*;

    fn package(path: &str) -> ModelPackage {
        let sha256 = "a".repeat(64);
        ModelPackage {
            id: ModelPackageId("package_test".to_owned()),
            source: ModelPackageSource::HuggingFace {
                repository: "owner/repo".to_owned(),
                revision: "b".repeat(40),
            },
            files: vec![ModelFile {
                id: ModelFileId(format!("file_{sha256}")),
                path: PathBuf::from(path),
                role: ModelFileRole::Weights,
                size_bytes: 1,
                sha256,
            }],
            relationships: Vec::new(),
            properties: ModelPackageProperties {
                format: "gguf".to_owned(),
                quantization: "test".to_owned(),
                quantization_name: "test".to_owned(),
                architecture: "test".to_owned(),
                maximum_context_length: 1,
            },
        }
    }

    #[test]
    fn rejects_traversal_and_absolute_package_paths() {
        assert!(ValidatedDownloadPackage::new(package("../model.gguf")).is_err());
        assert!(ValidatedDownloadPackage::new(package("/tmp/model.gguf")).is_err());
        assert!(ValidatedDownloadPackage::new(package("models/model.gguf")).is_ok());
    }

    #[test]
    fn rejects_duplicate_paths_and_missing_weights() {
        let mut duplicate = package("model.gguf");
        let mut second = duplicate.files[0].clone();
        second.id = ModelFileId(format!("file_{}", "c".repeat(64)));
        second.sha256 = "c".repeat(64);
        duplicate.files.push(second);
        assert!(ValidatedDownloadPackage::new(duplicate).is_err());

        let mut projector = package("projector.gguf");
        projector.files[0].role = ModelFileRole::Projector;
        assert!(ValidatedDownloadPackage::new(projector).is_err());
    }
}
