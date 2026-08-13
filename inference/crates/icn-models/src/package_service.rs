use std::collections::BTreeMap;
use std::fs;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

use futures_util::future::BoxFuture;
use icn_contracts::models::{
    InstalledModelPackage, InstalledModelPackages, InstalledModelPackagesResponse, ModelAssessment,
    ModelBundleInput, ModelFailure, ModelFile, ModelFileId, ModelFileRelationship, ModelFileRole,
    ModelPackage, ModelPackageId, ModelPackageInspection, ModelPackageInstallationOrigin,
    ModelPackageOperand, ModelPackageProperties, ModelPackageSource, ModelServingConfiguration,
    ModelServingConfigurationId, RemoveInstalledModelPackageResponse, ResolvedServableModelBundle,
    ServableModelBundle, ServableModelBundleKey, ServingProfile,
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
use crate::inventory::{InstalledPackageRecord, InstalledPackageSnapshot, ModelManager};

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
            quantization_name,
            training_context_length,
            ..
        } => ModelPackageProperties {
            format: "gguf".to_owned(),
            quantization: quantization.clone().unwrap_or_else(|| "unknown".to_owned()),
            quantization_name: quantization_name
                .clone()
                .unwrap_or_else(|| "unknown".to_owned()),
            architecture: architecture.clone().unwrap_or_else(|| "unknown".to_owned()),
            maximum_context_length: training_context_length.unwrap_or(1).max(1),
        },
        InventoryProperties::Pending | InventoryProperties::Unavailable { .. } => {
            ModelPackageProperties {
                format: "gguf".to_owned(),
                quantization: "unknown".to_owned(),
                quantization_name: "unknown".to_owned(),
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

pub fn serving_configuration_id(
    bundle_key: &ServableModelBundleKey,
    profile: &ServingProfile,
) -> ModelServingConfigurationId {
    let mut digest = Sha256::new();
    digest.update(bundle_key.0.as_bytes());
    digest.update(profile.context_length.to_le_bytes());
    ModelServingConfigurationId(format!("configuration_{:x}", digest.finalize()))
}

pub fn serving_configuration_identity_is_valid(configuration: &ModelServingConfiguration) -> bool {
    let bundle_key = servable_model_bundle_key_for_bundle(&configuration.bundle);
    configuration.id == serving_configuration_id(&bundle_key, &configuration.profile)
}

fn package_relationship(
    relationship: &ComponentRelationship,
    ids_by_declared_path: &BTreeMap<PathBuf, ModelFileId>,
) -> Option<ModelFileRelationship> {
    match relationship {
        ComponentRelationship::ProjectorFor { projector, model } => {
            Some(ModelFileRelationship::ProjectorFor {
                projector_file_id: ids_by_declared_path.get(projector)?.clone(),
                weights_file_id: ids_by_declared_path.get(model)?.clone(),
            })
        }
        ComponentRelationship::MtpFor { mtp, model } => Some(ModelFileRelationship::MtpFor {
            mtp_file_id: ids_by_declared_path.get(mtp)?.clone(),
            weights_file_id: ids_by_declared_path.get(model)?.clone(),
        }),
        ComponentRelationship::DraftFor {
            draft,
            model,
            method,
        } => Some(ModelFileRelationship::DraftFor {
            draft_file_id: ids_by_declared_path.get(draft)?.clone(),
            weights_file_id: ids_by_declared_path.get(model)?.clone(),
            method: method.clone(),
        }),
    }
}

fn package_from_resolved_with(
    resolved: &ResolvedModel,
    digest: impl Fn(&Path) -> Result<String, InventoryError>,
    tensor_storage: impl Fn(&Path, &ContentIdentity) -> Option<u64>,
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
                ComponentRole::Draft => ModelFileRole::Draft,
                ComponentRole::Mtp => ModelFileRole::Mtp,
                ComponentRole::Auxiliary => ModelFileRole::Auxiliary,
            },
            size_bytes: declared.size_bytes,
            tensor_storage_bytes: tensor_storage(absolute, &declared.content),
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
        if let Some(relationship) = component
            .relationship
            .as_ref()
            .and_then(|relationship| package_relationship(relationship, &ids_by_declared_path))
        {
            relationships.push(relationship);
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
    package_from_resolved_with(resolved, digest_file, |path, _| {
        crate::gguf::inspect(path)
            .ok()
            .map(|inspection| inspection.tensor_storage_bytes)
    })
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

pub fn servable_model_bundle_key(package_ids: &[&ModelPackageId]) -> ServableModelBundleKey {
    let mut digest = Sha256::new();
    digest.update(b"magnitude-servable-model-bundle-v1\0");
    for package_id in package_ids {
        digest.update(package_id.0.as_bytes());
        digest.update(b"\0");
    }
    ServableModelBundleKey(format!("bundle_{:x}", digest.finalize()))
}

pub fn speculative_servable_model_bundle_key(
    target: &ModelPackageId,
    draft: Option<&ModelPackageId>,
    method: &icn_contracts::models::SpeculativeMethod,
) -> ServableModelBundleKey {
    let mut digest = Sha256::new();
    digest.update(b"magnitude-servable-model-bundle-v2\0speculative\0");
    digest.update(target.0.as_bytes());
    digest.update(b"\0");
    digest.update(serde_json::to_vec(method).expect("speculative method is serializable"));
    digest.update(b"\0");
    match draft {
        None => digest.update(b"embedded\0"),
        Some(draft) => {
            digest.update(b"separate\0");
            digest.update(draft.0.as_bytes());
            digest.update(b"\0");
        }
    }
    ServableModelBundleKey(format!("bundle_{:x}", digest.finalize()))
}

pub fn servable_model_bundle_key_for_bundle(
    bundle: &ServableModelBundle,
) -> ServableModelBundleKey {
    match bundle {
        ServableModelBundle::Standalone { package } => servable_model_bundle_key(&[&package.id]),
        ServableModelBundle::SpeculativeDecoding {
            target,
            draft_source,
            method,
        } => speculative_servable_model_bundle_key(
            &target.id,
            match draft_source {
                icn_contracts::models::SpeculativeDraftSource::Embedded => None,
                icn_contracts::models::SpeculativeDraftSource::Separate { draft } => {
                    Some(&draft.id)
                }
            },
            method,
        ),
    }
}

impl ModelManager {
    fn seed_package_digests_from_snapshot(
        &self,
        resolved: &ResolvedModel,
    ) -> Result<(), InventoryError> {
        let cached = self
            .installed_packages
            .read()
            .map_err(|_| {
                InventoryError::Internal("installed package snapshot lock poisoned".to_owned())
            })?
            .records
            .values()
            .find(|record| {
                record.model.id == resolved.model.id
                    && record.model.content_id == resolved.model.content_id
            })
            .cloned();
        let Some(cached) = cached else {
            return Ok(());
        };
        let files = cached
            .installed
            .package
            .files
            .iter()
            .map(|file| (&file.path, file))
            .collect::<BTreeMap<_, _>>();
        let mut digests = self.package_digests.write().map_err(|_| {
            InventoryError::Internal("package digest cache lock poisoned".to_owned())
        })?;
        for (declared, component) in resolved
            .model
            .location
            .components()
            .iter()
            .zip(&resolved.components)
        {
            let Some(file) = files.get(&declared.path) else {
                continue;
            };
            if file.size_bytes != declared.size_bytes
                || file.id != file_id(&file.sha256)
                || file.sha256.len() != 64
                || !file
                    .sha256
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            {
                continue;
            }
            let Ok(metadata) = fs::metadata(&component.path) else {
                continue;
            };
            let Ok(modified) = metadata.modified() else {
                continue;
            };
            if metadata.len() == file.size_bytes {
                digests.insert(
                    component.path.clone(),
                    (metadata.len(), modified, file.sha256.clone()),
                );
            }
        }
        Ok(())
    }

    fn package_from_resolved(
        &self,
        resolved: &ResolvedModel,
    ) -> Result<ModelPackage, InventoryError> {
        package_from_resolved_with(
            resolved,
            |path| {
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
            },
            |path, content| self.inspect_tensor_storage_bytes(path, content),
        )
    }

    fn tensor_storage_cache_key(path: &Path, content: &ContentIdentity) -> Option<String> {
        let evidence = serde_json::to_vec(&(
            "gguf-tensor-storage-v1",
            content,
            path.metadata().ok()?.len(),
        ))
        .ok()?;
        Some(format!("tensor_storage_{:x}", Sha256::digest(evidence)))
    }

    fn inspect_tensor_storage_bytes(&self, path: &Path, content: &ContentIdentity) -> Option<u64> {
        let key = Self::tensor_storage_cache_key(path, content)?;
        if let Some(bytes) = self.cache.read_index(ModelIndexKind::TensorStorage, &key) {
            return Some(bytes);
        }
        let bytes = crate::gguf::inspect(path).ok()?.tensor_storage_bytes;
        self.cache
            .write_index(ModelIndexKind::TensorStorage, &key, &bytes);
        Some(bytes)
    }

    pub(crate) fn build_installed_package_snapshot(
        &self,
        models: &BTreeMap<icn_contracts::ModelId, InventoryModel>,
    ) -> Result<InstalledPackageSnapshot, InventoryError> {
        let mut records = BTreeMap::new();
        for model in models.values() {
            if !matches!(
                model.availability,
                ModelAvailability::Available { .. }
                    | ModelAvailability::InvalidArtifact { .. }
                    | ModelAvailability::IncompatibleArtifact { .. }
            ) {
                continue;
            }
            let resolved = ResolvedModel {
                components: crate::service::resolve_components(&self.config.root, model)?,
                model: model.clone(),
            };
            self.seed_package_digests_from_snapshot(&resolved)?;
            let package = self.package_from_resolved(&resolved)?;
            let installed = InstalledModelPackage {
                path: installed_path(model, &resolved),
                origin: match &model.location {
                    ModelLocation::MagnitudeCache { .. } => {
                        ModelPackageInstallationOrigin::Magnitude
                    }
                    ModelLocation::HuggingFaceCache { .. } => {
                        ModelPackageInstallationOrigin::HuggingFaceCache
                    }
                    ModelLocation::Directory { .. } | ModelLocation::File { .. } => {
                        ModelPackageInstallationOrigin::Magnitude
                    }
                },
                inspection: inspection_for(model),
                package,
            };
            records.insert(
                installed.package.id.clone(),
                InstalledPackageRecord {
                    installed,
                    model: model.clone(),
                },
            );
        }
        Ok(InstalledPackageSnapshot { records })
    }

    #[must_use]
    pub fn read_model_assessment(
        &self,
        evidence: &str,
        topology: &icn_contracts::MemoryTopology,
    ) -> Option<ModelAssessment> {
        if let Some(assessment) = self
            .model_assessments
            .read()
            .ok()?
            .get(evidence)
            .filter(|assessment| assessment.is_valid_for(topology))
            .cloned()
        {
            return Some(assessment);
        }
        let assessment = self.cache.read_model_assessment(evidence, topology)?;
        self.model_assessments
            .write()
            .ok()?
            .insert(evidence.to_owned(), assessment.clone());
        Some(assessment)
    }

    pub fn write_model_assessment(&self, evidence: &str, assessment: &ModelAssessment) {
        if let Ok(mut memory) = self.model_assessments.write() {
            memory.insert(evidence.to_owned(), assessment.clone());
        }
        self.cache.write_model_assessment(evidence, assessment);
    }

    pub(crate) async fn installed_package(
        &self,
        package_id: &ModelPackageId,
    ) -> Result<(ModelPackage, ResolvedModel), InventoryError> {
        let find = || {
            self.installed_packages
                .read()
                .map_err(|_| {
                    InventoryError::Internal("installed package snapshot lock poisoned".to_owned())
                })?
                .records
                .get(package_id)
                .cloned()
                .ok_or_else(|| InventoryError::NotFound(package_id.0.clone()))
        };
        let record = match find() {
            Ok(record) => record,
            Err(InventoryError::NotFound(_)) => {
                self.ensure_installed_model_inventory().await?;
                find()?
            }
            Err(error) => return Err(error),
        };
        let resolved = ResolvedModel {
            components: crate::service::resolve_components(&self.config.root, &record.model)?,
            model: record.model,
        };
        Ok((record.installed.package, resolved))
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
                            ModelFileRole::Projector => {
                                icn_contracts::ModelPreviewComponentRole::Projector
                            }
                            ModelFileRole::Draft => {
                                let method =
                                    package.relationships.iter().find_map(|relationship| {
                                        match relationship {
                                            ModelFileRelationship::DraftFor {
                                                draft_file_id,
                                                weights_file_id,
                                                method,
                                            } if draft_file_id == &file.id
                                                && weights_file_id == &primary.id =>
                                            {
                                                Some(method.clone())
                                            }
                                            _ => None,
                                        }
                                    });
                                let Some(method) = method else {
                                    return Some(Err(InventoryError::InvalidRequest(format!(
                                        "draft file {} has no typed relationship to target {}",
                                        file.id.0, primary.id.0
                                    ))));
                                };
                                return Some(Ok(ModelPreviewComponentSource {
                                    path: file.path.clone(),
                                    role: icn_contracts::ModelPreviewComponentRole::Draft {
                                        method,
                                    },
                                }));
                            }
                            ModelFileRole::Mtp => icn_contracts::ModelPreviewComponentRole::Mtp,
                            ModelFileRole::Auxiliary => return None,
                            ModelFileRole::Weights => return None,
                        };
                        Some(Ok(ModelPreviewComponentSource {
                            path: file.path.clone(),
                            role,
                        }))
                    })
                    .collect::<Result<Vec<_>, InventoryError>>()?;
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
        Box::pin(async move { self.installed_packages_response() })
    }

    fn resolve_bundle(
        &self,
        bundle: ModelBundleInput,
    ) -> BoxFuture<'_, Result<ResolvedServableModelBundle, InventoryError>> {
        Box::pin(async move {
            match bundle {
                ModelBundleInput::Standalone { package } => {
                    let resolved = self.resolve_package_operand(package).await?;
                    let bundle_key = servable_model_bundle_key(&[&resolved.package.id]);
                    let bundle = ServableModelBundle::Standalone {
                        package: resolved.package,
                    };
                    let mut result =
                        ResolvedServableModelBundle::new(bundle_key, bundle, resolved.model, None);
                    if let Some(guard) = resolved.resolution_guard {
                        result = result.retain_resolution_guard(guard);
                    }
                    Ok(result)
                }
                ModelBundleInput::SpeculativeDecoding {
                    target,
                    draft_source,
                    method,
                } => {
                    let target = self.resolve_package_operand(target).await?;
                    let (draft_source, draft_model, draft_guard) = match draft_source {
                        icn_contracts::models::SpeculativeDraftSourceInput::Embedded => (
                            icn_contracts::models::SpeculativeDraftSource::Embedded,
                            None,
                            None,
                        ),
                        icn_contracts::models::SpeculativeDraftSourceInput::Separate { draft } => {
                            let resolved = self.resolve_package_operand(draft).await?;
                            if resolved.package.id == target.package.id {
                                return Err(InventoryError::InvalidRequest(
                                    "a separate speculative draft must be distinct from its target"
                                        .to_owned(),
                                ));
                            }
                            (
                                icn_contracts::models::SpeculativeDraftSource::Separate {
                                    draft: resolved.package,
                                },
                                Some(resolved.model),
                                resolved.resolution_guard,
                            )
                        }
                    };
                    let bundle = ServableModelBundle::SpeculativeDecoding {
                        target: target.package,
                        draft_source,
                        method,
                    };
                    let bundle_key = servable_model_bundle_key_for_bundle(&bundle);
                    let mut result = ResolvedServableModelBundle::new(
                        bundle_key,
                        bundle,
                        target.model,
                        draft_model,
                    );
                    if let Some(guard) = target.resolution_guard {
                        result = result.retain_resolution_guard(guard);
                    }
                    if let Some(guard) = draft_guard {
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
            Ok(RemoveInstalledModelPackageResponse {
                package_id,
                removed: deleted.deleted,
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    use icn_contracts::ComponentRelationship;
    use icn_contracts::models::{
        ModelFileId, ModelFileRelationship, ModelPackageId, SpeculativeMethod,
    };

    use super::{package_relationship, shard_count, speculative_servable_model_bundle_key};

    #[test]
    fn shard_count_uses_one_based_component_indices() {
        assert_eq!(shard_count([Some(1), Some(2), Some(3)]), 3);
        assert_eq!(shard_count([None, None]), 0);
    }

    #[test]
    fn draft_relationship_preserves_method_and_does_not_collapse_to_mtp() {
        let target = PathBuf::from("target.gguf");
        let draft = PathBuf::from("dflash.gguf");
        let ids = BTreeMap::from([
            (target.clone(), ModelFileId("target".to_owned())),
            (draft.clone(), ModelFileId("draft".to_owned())),
        ]);

        let relationship = package_relationship(
            &ComponentRelationship::DraftFor {
                draft,
                model: target,
                method: SpeculativeMethod::DFlash,
            },
            &ids,
        );

        assert!(matches!(
            relationship,
            Some(ModelFileRelationship::DraftFor {
                method: SpeculativeMethod::DFlash,
                ..
            })
        ));
    }

    #[test]
    fn speculative_bundle_identity_includes_source_method_and_separate_draft() {
        let target = ModelPackageId("target".to_owned());
        let draft = ModelPackageId("draft".to_owned());
        let embedded =
            speculative_servable_model_bundle_key(&target, None, &SpeculativeMethod::Mtp);
        let dflash = speculative_servable_model_bundle_key(
            &target,
            Some(&draft),
            &SpeculativeMethod::DFlash,
        );
        let dspark = speculative_servable_model_bundle_key(
            &target,
            Some(&draft),
            &SpeculativeMethod::DSpark,
        );

        assert_ne!(embedded, dflash);
        assert_ne!(dflash, dspark);
    }
}
