use std::collections::BTreeMap;
use std::fs;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use futures_util::future::BoxFuture;
use icn_contracts::models::{
    InstalledModelPackage, InstalledModelPackages, InstalledModelPackagesResponse, ModelFailure,
    ModelFile, ModelFileId, ModelFileRelationship, ModelFileRole, ModelOfferingTarget,
    ModelOfferingTargetId, ModelPackage, ModelPackageId, ModelPackageInspection,
    ModelPackageOperand, ModelPackageProperties, ModelPackageSource, ModelTargetInput,
    OfferingAssessment, RemoveInstalledModelPackageResponse, ResolvedModelTarget,
    SpeculativeDecodingPairId,
};
use icn_contracts::{
    ComponentRelationship, ComponentRole, ContentIdentity, InventoryError, InventoryModel,
    InventoryProperties, ModelAvailability, ModelInventory, ModelLocation,
    ModelPreviewComponentSource, ModelPreviewSource, ModelSource, ResolvedModel,
};
use sha2::{Digest, Sha256};

use crate::PreparedPreview;
use crate::cache::ModelIndexKind;
use crate::capabilities::model_capabilities;
use crate::inventory::ModelManager;

struct ResolvedPackageOperand {
    package: ModelPackage,
    model: ResolvedModel,
    resolution_guard: Option<PreparedPreview>,
}

fn digest_file(path: &Path) -> Result<String, InventoryError> {
    let file = fs::File::open(path).map_err(|error| {
        InventoryError::Io(format!("failed to open {}: {error}", path.display()))
    })?;
    let mut reader = BufReader::with_capacity(1024 * 1024, file);
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = reader.read(&mut buffer).map_err(|error| {
            InventoryError::Io(format!("failed to read {}: {error}", path.display()))
        })?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn file_id(sha256: &str) -> ModelFileId {
    ModelFileId(format!("file_{sha256}"))
}

fn package_properties(properties: &InventoryProperties) -> ModelPackageProperties {
    match properties {
        InventoryProperties::Inspected {
            architecture,
            quantization,
            training_context_length,
            ..
        } => ModelPackageProperties {
            format: "gguf".to_owned(),
            quantization: quantization.clone().unwrap_or_else(|| "unknown".to_owned()),
            architecture: architecture.clone().unwrap_or_else(|| "unknown".to_owned()),
            maximum_context_length: training_context_length.unwrap_or(1).max(1),
        },
        InventoryProperties::Pending | InventoryProperties::Unavailable { .. } => {
            ModelPackageProperties {
                format: "gguf".to_owned(),
                quantization: "unknown".to_owned(),
                architecture: "unknown".to_owned(),
                maximum_context_length: 1,
            }
        }
    }
}

fn package_source(model: &InventoryModel, resolved: &ResolvedModel) -> ModelPackageSource {
    match &model.source {
        ModelSource::HuggingFace {
            repository, commit, ..
        } => ModelPackageSource::HuggingFace {
            repository: repository.clone(),
            revision: commit.clone(),
        },
        ModelSource::Local { .. } => {
            let root = match &model.location {
                ModelLocation::Directory { root, .. } => root.clone(),
                ModelLocation::File { path, .. } => path
                    .parent()
                    .map(Path::to_path_buf)
                    .unwrap_or_else(|| PathBuf::from(".")),
                ModelLocation::MagnitudeCache { .. } | ModelLocation::HuggingFaceCache { .. } => {
                    resolved
                        .components
                        .first()
                        .and_then(|component| component.path.parent())
                        .map(Path::to_path_buf)
                        .unwrap_or_else(|| PathBuf::from("."))
                }
            };
            ModelPackageSource::Local { path: root }
        }
    }
}

pub fn canonical_package_id(
    files: &[ModelFile],
    relationships: &[ModelFileRelationship],
) -> ModelPackageId {
    let mut digest = Sha256::new();
    digest.update(b"magnitude-model-package-v1\0");
    for file in files {
        digest.update(file.id.0.as_bytes());
        digest.update(b"\0");
        digest.update(format!("{:?}", file.role).as_bytes());
        digest.update(b"\0");
    }
    for relationship in relationships {
        digest.update(format!("{relationship:?}").as_bytes());
        digest.update(b"\0");
    }
    ModelPackageId(format!("package_{:x}", digest.finalize()))
}

fn package_from_resolved_with(
    resolved: &ResolvedModel,
    digest: impl Fn(&Path) -> Result<String, InventoryError>,
) -> Result<ModelPackage, InventoryError> {
    let model = &resolved.model;
    let source = package_source(model, resolved);
    let declared_components = model.location.components();
    if declared_components.len() != resolved.components.len() {
        return Err(InventoryError::Integrity(format!(
            "resolved model {} has {} declared components but {} resolved components",
            model.id.0,
            declared_components.len(),
            resolved.components.len(),
        )));
    }

    let mut files = Vec::with_capacity(declared_components.len());
    let mut ids_by_declared_path = BTreeMap::new();
    for (declared, resolved_component) in declared_components.iter().zip(&resolved.components) {
        let absolute = resolved_component.path.as_path();
        let sha256 = match &declared.content {
            ContentIdentity::Sha256 { value }
                if value.len() == 64
                    && value
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)) =>
            {
                value.clone()
            }
            _ => digest(absolute)?,
        };
        let id = file_id(&sha256);
        ids_by_declared_path.insert(declared.path.clone(), id.clone());
        files.push(ModelFile {
            id,
            path: declared.path.clone(),
            role: match declared.role {
                ComponentRole::Weights | ComponentRole::Shard => ModelFileRole::Weights,
                ComponentRole::Projector => ModelFileRole::Projector,
                ComponentRole::Mtp => ModelFileRole::Mtp,
                ComponentRole::Auxiliary | ComponentRole::Draft => ModelFileRole::Auxiliary,
            },
            size_bytes: declared.size_bytes,
            sha256,
        });
    }
    files.sort_by(|left, right| left.path.cmp(&right.path));

    let shard_count = shard_count(
        model
            .location
            .components()
            .iter()
            .map(|component| component.shard_index),
    );
    let mut relationships = Vec::new();
    for component in model.location.components() {
        let Some(file_id) = ids_by_declared_path.get(&component.path).cloned() else {
            continue;
        };
        if let Some(index) = component.shard_index {
            relationships.push(ModelFileRelationship::Shard {
                file_id: file_id.clone(),
                index,
                count: shard_count.max(1),
            });
        }
        match &component.relationship {
            Some(ComponentRelationship::ProjectorFor { projector, model }) => {
                if let (Some(projector_file_id), Some(weights_file_id)) = (
                    ids_by_declared_path.get(projector),
                    ids_by_declared_path.get(model),
                ) {
                    relationships.push(ModelFileRelationship::ProjectorFor {
                        projector_file_id: projector_file_id.clone(),
                        weights_file_id: weights_file_id.clone(),
                    });
                }
            }
            Some(ComponentRelationship::MtpFor { mtp, model })
            | Some(ComponentRelationship::DraftFor { draft: mtp, model }) => {
                if let (Some(mtp_file_id), Some(weights_file_id)) = (
                    ids_by_declared_path.get(mtp),
                    ids_by_declared_path.get(model),
                ) {
                    relationships.push(ModelFileRelationship::MtpFor {
                        mtp_file_id: mtp_file_id.clone(),
                        weights_file_id: weights_file_id.clone(),
                    });
                }
            }
            None => {}
        }
    }
    relationships.sort_by_key(|relationship| format!("{relationship:?}"));
    let properties = package_properties(&model.properties);
    let id = canonical_package_id(&files, &relationships);
    Ok(ModelPackage {
        id,
        source,
        files,
        relationships,
        properties,
    })
}

fn shard_count(indices: impl IntoIterator<Item = Option<u32>>) -> u32 {
    indices.into_iter().flatten().max().unwrap_or(0)
}

pub(crate) fn package_from_resolved(
    resolved: &ResolvedModel,
) -> Result<ModelPackage, InventoryError> {
    package_from_resolved_with(resolved, digest_file)
}

fn inspection_for(model: &InventoryModel) -> ModelPackageInspection {
    match &model.availability {
        ModelAvailability::InvalidArtifact { code, message, .. } => {
            ModelPackageInspection::Invalid {
                failure: ModelFailure {
                    code: code.clone(),
                    message: message.clone(),
                    retryable: false,
                },
            }
        }
        ModelAvailability::IncompatibleArtifact { code, message, .. } => {
            ModelPackageInspection::Incompatible {
                failure: ModelFailure {
                    code: code.clone(),
                    message: message.clone(),
                    retryable: false,
                },
            }
        }
        _ => match &model.properties {
            InventoryProperties::Pending => ModelPackageInspection::Pending,
            InventoryProperties::Unavailable { reason } => ModelPackageInspection::Invalid {
                failure: ModelFailure {
                    code: "inspection_unavailable".to_owned(),
                    message: reason.clone(),
                    retryable: true,
                },
            },
            InventoryProperties::Inspected { .. } => ModelPackageInspection::Inspected {
                capabilities: model_capabilities(&model.properties),
            },
        },
    }
}

fn installed_path(model: &InventoryModel, resolved: &ResolvedModel) -> PathBuf {
    match &model.location {
        ModelLocation::Directory { root, .. } => root.clone(),
        ModelLocation::File { path, .. } => path.clone(),
        ModelLocation::HuggingFaceCache { cache_root, .. } => cache_root.clone(),
        ModelLocation::MagnitudeCache { .. } => resolved
            .components
            .first()
            .and_then(|component| component.path.parent())
            .map(Path::to_path_buf)
            .unwrap_or_default(),
    }
}

pub fn offering_target_id(package_ids: &[&ModelPackageId]) -> ModelOfferingTargetId {
    let mut digest = Sha256::new();
    digest.update(b"magnitude-model-offering-target-v1\0");
    for package_id in package_ids {
        digest.update(package_id.0.as_bytes());
        digest.update(b"\0");
    }
    ModelOfferingTargetId(format!("target_{:x}", digest.finalize()))
}

fn speculative_pair_id(
    target: &ModelPackageId,
    draft: &ModelPackageId,
) -> SpeculativeDecodingPairId {
    let mut digest = Sha256::new();
    digest.update(b"magnitude-speculative-decoding-pair-v1\0");
    digest.update(target.0.as_bytes());
    digest.update(b"\0");
    digest.update(draft.0.as_bytes());
    SpeculativeDecodingPairId(format!("pair_{:x}", digest.finalize()))
}

impl ModelManager {
    fn package_from_resolved(
        &self,
        resolved: &ResolvedModel,
    ) -> Result<ModelPackage, InventoryError> {
        package_from_resolved_with(resolved, |path| {
            let metadata = fs::metadata(path).map_err(|error| {
                InventoryError::Io(format!("failed to inspect {}: {error}", path.display()))
            })?;
            let modified = metadata.modified().map_err(|error| {
                InventoryError::Io(format!(
                    "failed to inspect modification time for {}: {error}",
                    path.display()
                ))
            })?;
            if let Some((_, _, digest)) = self
                .package_digests
                .read()
                .map_err(|_| {
                    InventoryError::Internal("package digest cache lock poisoned".to_owned())
                })?
                .get(path)
                .filter(|(size, cached_modified, _)| {
                    *size == metadata.len() && *cached_modified == modified
                })
            {
                return Ok(digest.clone());
            }
            let digest = digest_file(path)?;
            self.package_digests
                .write()
                .map_err(|_| {
                    InventoryError::Internal("package digest cache lock poisoned".to_owned())
                })?
                .insert(
                    path.to_path_buf(),
                    (metadata.len(), modified, digest.clone()),
                );
            Ok(digest)
        })
    }

    #[must_use]
    pub fn read_offering_assessment(&self, evidence: &str) -> Option<OfferingAssessment> {
        self.cache
            .read_index(ModelIndexKind::OfferingAssessment, evidence)
    }

    pub fn write_offering_assessment(&self, evidence: &str, assessment: &OfferingAssessment) {
        self.cache
            .write_index(ModelIndexKind::OfferingAssessment, evidence, assessment);
    }

    async fn installed_package(
        &self,
        package_id: &ModelPackageId,
    ) -> Result<(ModelPackage, ResolvedModel), InventoryError> {
        let find = || {
            self.package_models
                .read()
                .map_err(|_| {
                    InventoryError::Internal("installed package index lock poisoned".to_owned())
                })?
                .get(package_id)
                .cloned()
                .ok_or_else(|| InventoryError::NotFound(package_id.0.clone()))
        };
        let model_id = match find() {
            Ok(model_id) => model_id,
            Err(InventoryError::NotFound(_)) => {
                <Self as InstalledModelPackages>::list_installed(self).await?;
                find()?
            }
            Err(error) => return Err(error),
        };
        let resolved = <Self as ModelInventory>::resolve_ready(self, &model_id).await?;
        let package = self.package_from_resolved(&resolved)?;
        if package.id != *package_id {
            self.package_models
                .write()
                .map_err(|_| {
                    InventoryError::Internal("installed package index lock poisoned".to_owned())
                })?
                .remove(package_id);
            return Err(InventoryError::NotFound(package_id.0.clone()));
        }
        Ok((package, resolved))
    }

    async fn resolve_package_operand(
        &self,
        operand: ModelPackageOperand,
    ) -> Result<ResolvedPackageOperand, InventoryError> {
        match operand {
            ModelPackageOperand::Installed { package_id } => {
                let (package, model) = self.installed_package(&package_id).await?;
                Ok(ResolvedPackageOperand {
                    package,
                    model,
                    resolution_guard: None,
                })
            }
            ModelPackageOperand::SourceBacked { package } => {
                match self.installed_package(&package.id).await {
                    Ok((package, model)) => {
                        return Ok(ResolvedPackageOperand {
                            package,
                            model,
                            resolution_guard: None,
                        });
                    }
                    Err(InventoryError::NotFound(_)) | Err(InventoryError::NotReady(_)) => {}
                    Err(error) => return Err(error),
                }
                let ModelPackageSource::HuggingFace {
                    repository,
                    revision,
                } = &package.source
                else {
                    return Err(InventoryError::NotReady(package.id.0));
                };
                let primary = package
                    .files
                    .iter()
                    .filter(|file| file.role == ModelFileRole::Weights)
                    .min_by(|left, right| left.path.cmp(&right.path))
                    .ok_or_else(|| {
                        InventoryError::InvalidRequest(format!(
                            "package {} has no weights",
                            package.id.0
                        ))
                    })?;
                let additional_components = package
                    .files
                    .iter()
                    .filter(|file| file.id != primary.id)
                    .filter_map(|file| {
                        let role = match file.role {
                            ModelFileRole::Projector => ComponentRole::Projector,
                            ModelFileRole::Mtp => ComponentRole::Mtp,
                            ModelFileRole::Auxiliary => ComponentRole::Auxiliary,
                            ModelFileRole::Weights => return None,
                        };
                        Some(ModelPreviewComponentSource {
                            path: file.path.clone(),
                            role,
                        })
                    })
                    .collect();
                let prepared = self
                    .prepare_preview(&ModelPreviewSource {
                        repository: repository.clone(),
                        revision: revision.clone(),
                        primary_gguf: primary.path.clone(),
                        additional_components,
                    })
                    .await?;
                let resolved_package = self.package_from_resolved(&prepared.model)?;
                if resolved_package.id != package.id {
                    return Err(InventoryError::Integrity(format!(
                        "source-backed package {} resolved as {}",
                        package.id.0, resolved_package.id.0
                    )));
                }
                let model = prepared.model.clone();
                Ok(ResolvedPackageOperand {
                    package: resolved_package,
                    model,
                    resolution_guard: Some(prepared),
                })
            }
        }
    }
}

impl InstalledModelPackages for ModelManager {
    fn list_installed(
        &self,
    ) -> BoxFuture<'_, Result<InstalledModelPackagesResponse, InventoryError>> {
        Box::pin(async move {
            self.ensure_installed_model_inventory().await?;
            let models = self
                .models
                .read()
                .map_err(|_| InventoryError::Internal("inventory lock poisoned".to_owned()))?
                .values()
                .cloned()
                .collect::<Vec<_>>();
            let mut packages = Vec::new();
            let mut package_models = BTreeMap::new();
            for model in models {
                if !matches!(
                    model.availability,
                    ModelAvailability::Available { .. }
                        | ModelAvailability::InvalidArtifact { .. }
                        | ModelAvailability::IncompatibleArtifact { .. }
                ) {
                    continue;
                }
                let resolved = ResolvedModel {
                    components: crate::service::resolve_components(&self.config.root, &model)?,
                    model,
                };
                let package = self.package_from_resolved(&resolved)?;
                package_models.insert(package.id.clone(), resolved.model.id.clone());
                packages.push(InstalledModelPackage {
                    target_id: offering_target_id(&[&package.id]),
                    path: installed_path(&resolved.model, &resolved),
                    inspection: inspection_for(&resolved.model),
                    package,
                });
            }
            packages.sort_by(|left, right| left.package.id.cmp(&right.package.id));
            packages.dedup_by(|left, right| left.package.id == right.package.id);
            *self.package_models.write().map_err(|_| {
                InventoryError::Internal("installed package index lock poisoned".to_owned())
            })? = package_models;
            Ok(InstalledModelPackagesResponse { packages })
        })
    }

    fn resolve_target(
        &self,
        target: ModelTargetInput,
    ) -> BoxFuture<'_, Result<ResolvedModelTarget, InventoryError>> {
        Box::pin(async move {
            match target {
                ModelTargetInput::Package { package } => {
                    let resolved = self.resolve_package_operand(package).await?;
                    let target_id = offering_target_id(&[&resolved.package.id]);
                    let target = ModelOfferingTarget::Package {
                        package: resolved.package,
                    };
                    let mut result =
                        ResolvedModelTarget::new(target_id, target, resolved.model, None);
                    if let Some(guard) = resolved.resolution_guard {
                        result = result.retain_resolution_guard(guard);
                    }
                    Ok(result)
                }
                ModelTargetInput::SpeculativeDecodingPair { target, draft } => {
                    let target = self.resolve_package_operand(target).await?;
                    let draft = self.resolve_package_operand(draft).await?;
                    let pair_id = speculative_pair_id(&target.package.id, &draft.package.id);
                    let target_id = offering_target_id(&[&target.package.id, &draft.package.id]);
                    let mut result = ResolvedModelTarget::new(
                        target_id,
                        ModelOfferingTarget::SpeculativeDecodingPair {
                            id: pair_id,
                            target: target.package,
                            draft: draft.package,
                        },
                        target.model,
                        Some(draft.model),
                    );
                    if let Some(guard) = target.resolution_guard {
                        result = result.retain_resolution_guard(guard);
                    }
                    if let Some(guard) = draft.resolution_guard {
                        result = result.retain_resolution_guard(guard);
                    }
                    Ok(result)
                }
            }
        })
    }

    fn remove_installed(
        &self,
        package_id: &ModelPackageId,
    ) -> BoxFuture<'_, Result<RemoveInstalledModelPackageResponse, InventoryError>> {
        let package_id = package_id.clone();
        Box::pin(async move {
            let (_, resolved) = self.installed_package(&package_id).await?;
            let deleted = <Self as ModelInventory>::delete(self, &resolved.model.id).await?;
            self.package_models
                .write()
                .map_err(|_| {
                    InventoryError::Internal("installed package index lock poisoned".to_owned())
                })?
                .remove(&package_id);
            Ok(RemoveInstalledModelPackageResponse {
                package_id,
                removed: deleted.deleted,
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::shard_count;

    #[test]
    fn shard_count_uses_one_based_component_indices() {
        assert_eq!(shard_count([Some(1), Some(2), Some(3)]), 3);
        assert_eq!(shard_count([None, None]), 0);
    }
}
