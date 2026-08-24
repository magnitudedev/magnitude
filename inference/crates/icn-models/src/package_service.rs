use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;

use futures_util::future::BoxFuture;
use icn_contracts::models::{
    CatalogPackageAffiliation, CatalogPackageRole, InstalledCatalogAttribution,
    InstalledModelPackage, InstalledModelPackages, InstalledModelPackagesResponse, ModelAssessment,
    ModelBundleInput, ModelFailure, ModelFile, ModelFileId, ModelFileRelationship, ModelFileRole,
    ModelPackage, ModelPackageId, ModelPackageInspection, ModelPackageInstallationOrigin,
    ModelPackageOperand, ModelPackageProperties, ModelPackageSource,
    RemoveInstalledModelPackageResponse, ResolvedServableModelBundle, ServableModelBundle,
    ServableModelBundleKey, ServingProfile,
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
use crate::inventory::{
    InstalledPackageRecord, InstalledPackageSnapshot, ManagedModelStore, catalog_packages,
    catalog_target,
};

struct ResolvedPackageOperand {
    package: ModelPackage,
    model: ResolvedModel,
    resolution_guard: Option<PreparedPreview>,
}

#[derive(Debug)]
pub(crate) struct InspectedModelPackage {
    pub(crate) package: ModelPackage,
    pub(crate) inspection: ModelPackageInspection,
}

#[derive(Clone, Copy)]
struct ProjectorCapabilities {
    vision: bool,
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

fn intrinsic_target_identity(
    inspected: &[(ComponentRole, crate::gguf::GgufInspection)],
) -> (Option<String>, Option<String>) {
    let inspected = inspected
        .iter()
        .filter_map(|(role, inspection)| {
            matches!(role, ComponentRole::Weights | ComponentRole::Shard).then_some(inspection)
        })
        .collect::<Vec<_>>();
    let model_ids = inspected
        .iter()
        .filter_map(|inspection| inspection.name.clone())
        .collect::<BTreeSet<_>>();
    let quality_ids = inspected
        .iter()
        .filter_map(|inspection| inspection.quantization.clone())
        .collect::<BTreeSet<_>>();
    let complete = !inspected.is_empty() && model_ids.len() == 1 && quality_ids.len() == 1;
    if !complete {
        return (None, None);
    }
    (model_ids.into_iter().next(), quality_ids.into_iter().next())
}

fn package_properties(
    resolved: &ResolvedModel,
    inspections: &[(ComponentRole, crate::gguf::GgufInspection)],
) -> ModelPackageProperties {
    let properties = &resolved.model.properties;
    let (intrinsic_model_id, intrinsic_quality_id) = intrinsic_target_identity(inspections);
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
            maximum_context_length: *training_context_length,
            intrinsic_model_id,
            intrinsic_quality_id,
        },
        InventoryProperties::Pending | InventoryProperties::Unavailable { .. } => {
            ModelPackageProperties {
                format: "gguf".to_owned(),
                quantization: "unknown".to_owned(),
                quantization_name: "unknown".to_owned(),
                architecture: "unknown".to_owned(),
                maximum_context_length: None,
                intrinsic_model_id: None,
                intrinsic_quality_id: None,
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

/// Private implementation fingerprint for an exact derived serving configuration.
/// This is never a model identity and must not cross the ICN boundary.
pub fn serving_configuration_fingerprint(
    bundle_key: &ServableModelBundleKey,
    profile: &ServingProfile,
) -> String {
    let mut digest = Sha256::new();
    digest.update(bundle_key.0.as_bytes());
    digest.update(profile.context_length.to_le_bytes());
    format!("{:x}", digest.finalize())
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
    inspect: impl Fn(&Path, &ContentIdentity) -> Option<crate::gguf::GgufInspection>,
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
    let mut inspections = Vec::new();
    for (declared, resolved_component) in declared_components.iter().zip(&resolved.components) {
        let absolute = resolved_component.path.as_path();
        let inspection = matches!(model.properties, InventoryProperties::Inspected { .. })
            .then(|| inspect(absolute, &declared.content))
            .flatten();
        if let Some(inspection) = inspection.as_ref() {
            inspections.push((declared.role.clone(), inspection.clone()));
        }
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
            tensor_storage_bytes: inspection.map(|inspection| inspection.tensor_storage_bytes),
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
    let properties = package_properties(resolved, &inspections);
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

fn invalid_package_inspection(code: &str, message: impl Into<String>) -> ModelPackageInspection {
    ModelPackageInspection::Invalid {
        failure: ModelFailure {
            code: code.to_owned(),
            message: message.into(),
            retryable: false,
        },
    }
}

fn package_inspection_for(
    model: &InventoryModel,
    resolved: &ResolvedModel,
    package: &ModelPackage,
    inspect_projector: impl Fn(&Path) -> Result<ProjectorCapabilities, String>,
) -> ModelPackageInspection {
    match &model.availability {
        ModelAvailability::InvalidArtifact { code, message, .. } => {
            return ModelPackageInspection::Invalid {
                failure: ModelFailure {
                    code: code.clone(),
                    message: message.clone(),
                    retryable: false,
                },
            };
        }
        ModelAvailability::IncompatibleArtifact { code, message, .. } => {
            return ModelPackageInspection::Incompatible {
                failure: ModelFailure {
                    code: code.clone(),
                    message: message.clone(),
                    retryable: false,
                },
            };
        }
        ModelAvailability::Available { .. }
        | ModelAvailability::Downloading { .. }
        | ModelAvailability::Interrupted { .. } => {}
    }

    match &model.properties {
        InventoryProperties::Pending => ModelPackageInspection::Pending,
        InventoryProperties::Unavailable { reason } => ModelPackageInspection::Invalid {
            failure: ModelFailure {
                code: "inspection_unavailable".to_owned(),
                message: reason.clone(),
                retryable: true,
            },
        },
        InventoryProperties::Inspected { .. } => {
            let projectors = package
                .files
                .iter()
                .filter(|file| file.role == ModelFileRole::Projector)
                .collect::<Vec<_>>();
            let vision = match projectors.as_slice() {
                [] => false,
                [projector] => {
                    let related_to_weights = package.relationships.iter().any(|relationship| {
                        matches!(
                            relationship,
                            ModelFileRelationship::ProjectorFor {
                                projector_file_id,
                                weights_file_id,
                            } if projector_file_id == &projector.id
                                && package.files.iter().any(|file| {
                                    file.id == *weights_file_id && file.role == ModelFileRole::Weights
                                })
                        )
                    });
                    if !related_to_weights {
                        return invalid_package_inspection(
                            "invalid_projector_relationship",
                            "the multimodal projector is not related to package weights",
                        );
                    }
                    let Some(path) = model
                        .location
                        .components()
                        .iter()
                        .zip(&resolved.components)
                        .find_map(|(declared, resolved)| {
                            (declared.role == ComponentRole::Projector)
                                .then_some(resolved.path.as_path())
                        })
                    else {
                        return invalid_package_inspection(
                            "missing_projector_component",
                            "the package projector has no resolved component",
                        );
                    };
                    match inspect_projector(path) {
                        Ok(capabilities) if capabilities.vision => true,
                        Ok(_) => {
                            return ModelPackageInspection::Incompatible {
                                failure: ModelFailure {
                                    code: "projector_without_vision".to_owned(),
                                    message: "the configured multimodal projector does not support image input"
                                        .to_owned(),
                                    retryable: false,
                                },
                            };
                        }
                        Err(message) => {
                            return ModelPackageInspection::Incompatible {
                                failure: ModelFailure {
                                    code: "projector_inspection_failed".to_owned(),
                                    message,
                                    retryable: false,
                                },
                            };
                        }
                    }
                }
                _ => {
                    return invalid_package_inspection(
                        "ambiguous_projector_components",
                        "a model package may contain at most one multimodal projector",
                    );
                }
            };
            ModelPackageInspection::Inspected {
                capabilities: model_capabilities(&model.properties, vision),
            }
        }
    }
}

fn inspected_package_from_resolved_with(
    resolved: &ResolvedModel,
    digest: impl Fn(&Path) -> Result<String, InventoryError>,
    inspect_gguf: impl Fn(&Path, &ContentIdentity) -> Option<crate::gguf::GgufInspection>,
    inspect_projector: impl Fn(&Path) -> Result<ProjectorCapabilities, String>,
) -> Result<InspectedModelPackage, InventoryError> {
    let package = package_from_resolved_with(resolved, digest, inspect_gguf)?;
    let inspection = package_inspection_for(&resolved.model, resolved, &package, inspect_projector);
    Ok(InspectedModelPackage {
        package,
        inspection,
    })
}

pub(crate) fn inspected_package_from_resolved(
    resolved: &ResolvedModel,
) -> Result<InspectedModelPackage, InventoryError> {
    inspected_package_from_resolved_with(
        resolved,
        digest_file,
        |path, _| crate::gguf::inspect(path).ok(),
        |path| {
            llama_cpp_2::mtmd::mtmd_capabilities_from_file(path)
                .map(|capabilities| ProjectorCapabilities {
                    vision: capabilities.vision,
                })
                .map_err(|error| {
                    format!(
                        "failed to inspect multimodal projector {}: {error}",
                        path.display()
                    )
                })
        },
    )
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

impl ManagedModelStore {
    fn installed_catalog_attribution(
        &self,
        model: &InventoryModel,
        package: &ModelPackage,
    ) -> InstalledCatalogAttribution {
        if !matches!(model.location, ModelLocation::MagnitudeCache { .. }) {
            return InstalledCatalogAttribution::NotCatalogTarget;
        }

        let attributed = |model: &icn_contracts::models::RecommendableModel| {
            InstalledCatalogAttribution::Attributed {
                model_id: model.model_id.clone(),
                variant_id: model.variant_id.clone(),
            }
        };
        let failure = |code: &str, message: String| InstalledCatalogAttribution::Failed {
            failure: ModelFailure {
                code: code.to_owned(),
                message,
                retryable: false,
            },
        };
        let exact = self
            .config
            .catalog_models
            .iter()
            .filter(|catalog_model| catalog_target(catalog_model).id == package.id)
            .collect::<Vec<_>>();
        if exact.len() == 1 {
            self.remember_catalog_package(exact[0], package, CatalogPackageRole::Target);
            return attributed(exact[0]);
        }
        if exact.len() > 1 {
            return failure(
                "catalog_target_package_ambiguous",
                format!(
                    "package {} is the current target of multiple catalog variants",
                    package.id.0
                ),
            );
        }

        let ModelPackageSource::HuggingFace { repository, .. } = &package.source else {
            return failure(
                "catalog_target_source_missing",
                "managed catalog model has no Hugging Face target repository".to_owned(),
            );
        };
        let Some(quality) = package.properties.intrinsic_quality_id.as_deref() else {
            return failure(
                "catalog_target_identity_missing",
                "managed catalog model has no intrinsic GGUF quality identity".to_owned(),
            );
        };

        let affiliated_keys = self
            .catalog_affiliations
            .read()
            .ok()
            .map(|affiliations| {
                affiliations
                    .entries()
                    .filter(|affiliation| {
                        affiliation.role == CatalogPackageRole::Target
                            && affiliation.repository == *repository
                    })
                    .map(|affiliation| {
                        (affiliation.model_id.clone(), affiliation.variant_id.clone())
                    })
                    .collect::<BTreeSet<_>>()
            })
            .unwrap_or_default();
        let affiliated = self
            .config
            .catalog_models
            .iter()
            .filter(|catalog_model| {
                affiliated_keys.contains(&(
                    catalog_model.model_id.clone(),
                    catalog_model.variant_id.clone(),
                )) && catalog_target(catalog_model)
                    .properties
                    .intrinsic_quality_id
                    .as_deref()
                    == Some(quality)
            })
            .collect::<Vec<_>>();
        if affiliated.len() == 1 {
            return attributed(affiliated[0]);
        }
        if affiliated.len() > 1 {
            return failure(
                "catalog_repository_affiliation_ambiguous",
                format!(
                    "repository {repository} and GGUF quality {quality} identify multiple catalog variants"
                ),
            );
        }

        let Some(intrinsic_model_id) = package.properties.intrinsic_model_id.as_deref() else {
            return failure(
                "catalog_target_identity_missing",
                "managed catalog model has no intrinsic GGUF model identity".to_owned(),
            );
        };
        let intrinsic = self
            .config
            .catalog_models
            .iter()
            .filter(|catalog_model| {
                catalog_target(catalog_model)
                    .properties
                    .intrinsic_model_id
                    .as_deref()
                    == Some(intrinsic_model_id)
                    && catalog_target(catalog_model)
                        .properties
                        .intrinsic_quality_id
                        .as_deref()
                        == Some(quality)
            })
            .collect::<Vec<_>>();
        if intrinsic.len() == 1 {
            self.remember_catalog_package(intrinsic[0], package, CatalogPackageRole::Target);
            return attributed(intrinsic[0]);
        }
        if intrinsic.len() > 1 {
            failure(
                "catalog_intrinsic_identity_ambiguous",
                format!(
                    "intrinsic GGUF identity ({intrinsic_model_id}, {quality}) identifies multiple catalog variants"
                ),
            )
        } else {
            failure(
                "catalog_target_unrecognized",
                format!(
                    "intrinsic GGUF identity ({intrinsic_model_id}, {quality}) is not in the catalog"
                ),
            )
        }
    }

    fn remember_catalog_package(
        &self,
        catalog_model: &icn_contracts::models::RecommendableModel,
        package: &ModelPackage,
        role: CatalogPackageRole,
    ) {
        let ModelPackageSource::HuggingFace { repository, .. } = &package.source else {
            return;
        };
        let Ok(mut affiliations) = self.catalog_affiliations.write() else {
            tracing::warn!("catalog affiliation lock poisoned while recording installed package");
            return;
        };
        let added = affiliations.add(CatalogPackageAffiliation {
            model_id: catalog_model.model_id.clone(),
            variant_id: catalog_model.variant_id.clone(),
            package_id: package.id.clone(),
            repository: repository.clone(),
            role,
        });
        if added {
            self.catalog_affiliations_dirty
                .store(true, Ordering::Release);
        }
        if self.catalog_affiliations_dirty.load(Ordering::Acquire) {
            match affiliations.persist(&self.config.root) {
                Ok(()) => self
                    .catalog_affiliations_dirty
                    .store(false, Ordering::Release),
                Err(error) => tracing::warn!(
                    %error,
                    "failed to persist catalog package affiliations; retrying on the next inventory reconciliation"
                ),
            }
        }
    }

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

    fn inspect_package_from_resolved(
        &self,
        resolved: &ResolvedModel,
    ) -> Result<InspectedModelPackage, InventoryError> {
        inspected_package_from_resolved_with(
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
            |path, content| self.inspect_gguf(path, content),
            |path| {
                llama_cpp_2::mtmd::mtmd_capabilities_from_file(path)
                    .map(|capabilities| ProjectorCapabilities {
                        vision: capabilities.vision,
                    })
                    .map_err(|error| {
                        format!(
                            "failed to inspect multimodal projector {}: {error}",
                            path.display()
                        )
                    })
            },
        )
    }

    fn gguf_inspection_cache_key(path: &Path, content: &ContentIdentity) -> Option<String> {
        let evidence =
            serde_json::to_vec(&("gguf-inspection-v1", content, path.metadata().ok()?.len()))
                .ok()?;
        Some(format!("gguf_inspection_{:x}", Sha256::digest(evidence)))
    }

    fn inspect_gguf(
        &self,
        path: &Path,
        content: &ContentIdentity,
    ) -> Option<crate::gguf::GgufInspection> {
        let key = Self::gguf_inspection_cache_key(path, content)?;
        if let Some(inspection) = self.cache.read_index(ModelIndexKind::GgufInspection, &key) {
            return Some(inspection);
        }
        let inspection = crate::gguf::inspect(path).ok()?;
        self.cache
            .write_index(ModelIndexKind::GgufInspection, &key, &inspection);
        Some(inspection)
    }

    pub(crate) fn build_installed_package_snapshot(
        &self,
        models: &BTreeMap<icn_contracts::ModelId, InventoryModel>,
    ) -> Result<InstalledPackageSnapshot, InventoryError> {
        self.retry_catalog_affiliation_persistence();
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
            let InspectedModelPackage {
                package,
                inspection,
            } = self.inspect_package_from_resolved(&resolved)?;
            for catalog_model in &self.config.catalog_models {
                for (catalog_package, role) in catalog_packages(catalog_model) {
                    if catalog_package.id == package.id {
                        self.remember_catalog_package(catalog_model, &package, role);
                    }
                }
            }
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
                inspection,
                catalog_attribution: self.installed_catalog_attribution(model, &package),
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

    fn retry_catalog_affiliation_persistence(&self) {
        if !self.catalog_affiliations_dirty.load(Ordering::Acquire) {
            return;
        }
        let Ok(affiliations) = self.catalog_affiliations.read() else {
            tracing::warn!("catalog affiliation lock poisoned while retrying persistence");
            return;
        };
        match affiliations.persist(&self.config.root) {
            Ok(()) => self
                .catalog_affiliations_dirty
                .store(false, Ordering::Release),
            Err(error) => tracing::warn!(
                %error,
                "failed to persist catalog package affiliations; retrying on the next inventory reconciliation"
            ),
        }
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
                let resolved_package = self.inspect_package_from_resolved(&prepared.model)?.package;
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

impl InstalledModelPackages for ManagedModelStore {
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

    use icn_contracts::models::{
        ModelFile, ModelFileId, ModelFileRelationship, ModelFileRole, ModelPackage, ModelPackageId,
        ModelPackageInspection, ModelPackageProperties, ModelPackageSource, SpeculativeMethod,
    };
    use icn_contracts::{
        CapabilityEvidence, CapabilitySupport, ComponentRelationship, ComponentRole, ContentId,
        ContentIdentity, Integrity, InventoryModel, InventoryProperties, LocalDeclaration,
        ModelAvailability, ModelComponent, ModelId, ModelLocation, ModelSource,
        ReasoningCapability, ResolvedComponent, ResolvedModel,
    };

    use super::{
        ProjectorCapabilities, package_inspection_for, package_relationship, shard_count,
        speculative_servable_model_bundle_key,
    };

    fn inspected_resolved_model(components: Vec<ModelComponent>) -> ResolvedModel {
        let resolved = components
            .iter()
            .map(|component| ResolvedComponent {
                path: PathBuf::from("/models").join(&component.path),
                role: component.role.clone(),
                shard_index: component.shard_index,
                relationship: component.relationship.clone(),
            })
            .collect();
        ResolvedModel {
            model: InventoryModel {
                id: ModelId("model".to_owned()),
                content_id: ContentId("content".to_owned()),
                created: 0,
                name: "model".to_owned(),
                supported_parameters: Vec::new(),
                availability: ModelAvailability::Available { ready_at: 0 },
                source: ModelSource::Local {
                    declared_by: LocalDeclaration::Discovery,
                },
                location: ModelLocation::Directory {
                    source_id: "source".to_owned(),
                    root: PathBuf::from("/models"),
                    components,
                    total_bytes: 2,
                    integrity: Integrity::Unverified {
                        reason: "test".to_owned(),
                    },
                },
                properties: InventoryProperties::Inspected {
                    architecture: Some("test".to_owned()),
                    quantization: None,
                    quantization_name: None,
                    parameter_count: None,
                    active_parameter_count: None,
                    training_context_length: Some(32_768),
                    nextn_predict_layers: None,
                    tokenizer: None,
                    modalities: vec!["text".to_owned(), "image".to_owned()],
                    base_models: Vec::new(),
                    tools: CapabilitySupport::Unsupported,
                    structured_output: CapabilitySupport::Unsupported,
                    reasoning: ReasoningCapability::Unsupported {
                        evidence: CapabilityEvidence::DeclaredMetadata {
                            source: "test".to_owned(),
                        },
                    },
                    evidence_fingerprint: "test".to_owned(),
                },
                operations: Vec::new(),
                updated_at: 0,
            },
            components: resolved,
        }
    }

    fn package(with_projector: bool) -> ModelPackage {
        let weights_id = ModelFileId("weights".to_owned());
        let projector_id = ModelFileId("projector".to_owned());
        ModelPackage {
            id: ModelPackageId("package".to_owned()),
            source: ModelPackageSource::Local {
                path: PathBuf::from("/models"),
            },
            files: std::iter::once(ModelFile {
                id: weights_id.clone(),
                path: PathBuf::from("model.gguf"),
                role: ModelFileRole::Weights,
                size_bytes: 1,
                tensor_storage_bytes: None,
                sha256: "a".repeat(64),
            })
            .chain(with_projector.then_some(ModelFile {
                id: projector_id.clone(),
                path: PathBuf::from("mmproj.gguf"),
                role: ModelFileRole::Projector,
                size_bytes: 1,
                tensor_storage_bytes: None,
                sha256: "b".repeat(64),
            }))
            .collect(),
            relationships: with_projector
                .then_some(ModelFileRelationship::ProjectorFor {
                    projector_file_id: projector_id,
                    weights_file_id: weights_id,
                })
                .into_iter()
                .collect(),
            properties: ModelPackageProperties {
                format: "gguf".to_owned(),
                quantization: "unknown".to_owned(),
                quantization_name: "unknown".to_owned(),
                architecture: "test".to_owned(),
                maximum_context_length: Some(32_768),
                intrinsic_model_id: None,
                intrinsic_quality_id: None,
            },
        }
    }

    fn components(with_projector: bool) -> Vec<ModelComponent> {
        let weights = PathBuf::from("model.gguf");
        let mut components = vec![ModelComponent {
            path: weights.clone(),
            role: ComponentRole::Weights,
            size_bytes: 1,
            content: ContentIdentity::Unknown,
            shard_index: None,
            relationship: None,
        }];
        if with_projector {
            let projector = PathBuf::from("mmproj.gguf");
            components.push(ModelComponent {
                path: projector.clone(),
                role: ComponentRole::Projector,
                size_bytes: 1,
                content: ContentIdentity::Unknown,
                shard_index: None,
                relationship: Some(ComponentRelationship::ProjectorFor {
                    projector,
                    model: weights,
                }),
            });
        }
        components
    }

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

    #[test]
    fn target_metadata_does_not_advertise_vision_without_a_projector() {
        let resolved = inspected_resolved_model(components(false));
        let inspection =
            package_inspection_for(&resolved.model, &resolved, &package(false), |_| {
                panic!("a package without a projector must not inspect one")
            });

        assert!(matches!(
            inspection,
            ModelPackageInspection::Inspected { capabilities } if !capabilities.vision
        ));
    }

    #[test]
    fn exact_related_native_projector_establishes_vision() {
        let resolved = inspected_resolved_model(components(true));
        let inspection = package_inspection_for(&resolved.model, &resolved, &package(true), |_| {
            Ok(ProjectorCapabilities { vision: true })
        });

        assert!(matches!(
            inspection,
            ModelPackageInspection::Inspected { capabilities } if capabilities.vision
        ));
    }

    #[test]
    fn non_vision_projector_makes_the_package_incompatible() {
        let resolved = inspected_resolved_model(components(true));
        let inspection = package_inspection_for(&resolved.model, &resolved, &package(true), |_| {
            Ok(ProjectorCapabilities { vision: false })
        });

        assert!(matches!(
            inspection,
            ModelPackageInspection::Incompatible { failure }
                if failure.code == "projector_without_vision"
        ));
    }

    #[test]
    fn unbound_projector_makes_the_package_invalid() {
        let resolved = inspected_resolved_model(components(true));
        let mut model_package = package(true);
        model_package.relationships.clear();
        let inspection = package_inspection_for(&resolved.model, &resolved, &model_package, |_| {
            panic!("an unbound projector must not reach native inspection")
        });

        assert!(matches!(
            inspection,
            ModelPackageInspection::Invalid { failure }
                if failure.code == "invalid_projector_relationship"
        ));
    }

    #[test]
    fn unreadable_projector_makes_the_package_incompatible() {
        let resolved = inspected_resolved_model(components(true));
        let inspection = package_inspection_for(&resolved.model, &resolved, &package(true), |_| {
            Err("native parse failed".to_owned())
        });

        assert!(matches!(
            inspection,
            ModelPackageInspection::Incompatible { failure }
                if failure.code == "projector_inspection_failed"
                    && failure.message == "native parse failed"
        ));
    }
}
