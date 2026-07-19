use std::collections::BTreeSet;
use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};

use fs2::FileExt;
use futures_util::future::BoxFuture;
use icn_contracts::{
    DeletePlan, DeletedModel, DownloadEventStream, DownloadModelRequest, InventoryError,
    InventoryModel, ModelId, ModelInventory, ModelLocation, ModelSource, ModelStatus,
    ResolvedComponent, ResolvedModel,
};

use crate::download::blob_key;
use crate::inventory::{ModelManager, hf_repo_dir, now};
use crate::manifest::{MANIFEST_VERSION, ManagedManifest, OperationManifest};

impl ModelInventory for ModelManager {
    fn list(&self) -> BoxFuture<'_, Result<Vec<InventoryModel>, InventoryError>> {
        Box::pin(async move {
            self.ensure_model_inventory().await?;
            let mut models = self
                .models
                .read()
                .map_err(|_| InventoryError::Internal("inventory lock poisoned".to_owned()))?
                .values()
                .cloned()
                .collect::<Vec<_>>();
            models.sort_by(|left, right| {
                status_rank(&left.status)
                    .cmp(&status_rank(&right.status))
                    .then_with(|| left.name.to_lowercase().cmp(&right.name.to_lowercase()))
                    .then_with(|| left.id.cmp(&right.id))
            });
            Ok(models)
        })
    }

    fn get(&self, id: &ModelId) -> BoxFuture<'_, Result<InventoryModel, InventoryError>> {
        let id = id.clone();
        Box::pin(async move {
            self.models
                .read()
                .map_err(|_| InventoryError::Internal("inventory lock poisoned".to_owned()))?
                .get(&id)
                .cloned()
                .ok_or_else(|| InventoryError::NotFound(id.0.clone()))
        })
    }

    fn download(
        &self,
        request: DownloadModelRequest,
    ) -> BoxFuture<'_, Result<DownloadEventStream, InventoryError>> {
        Box::pin(async move { self.start_download(request).await })
    }

    fn plan_delete(&self, id: &ModelId) -> BoxFuture<'_, Result<DeletePlan, InventoryError>> {
        let id = id.clone();
        Box::pin(async move {
            let model = self
                .models
                .read()
                .map_err(|_| InventoryError::Internal("inventory lock poisoned".to_owned()))?
                .get(&id)
                .cloned()
                .ok_or_else(|| InventoryError::NotFound(id.0.clone()))?;
            ensure_deletable_status(&model)?;
            if matches!(model.status, ModelStatus::Interrupted { .. }) {
                return plan_interrupted_delete(&self.config.root, &model);
            }
            match &model.location {
                ModelLocation::MagnitudeCache { components, .. } => {
                    plan_managed_delete(&self.config.root, &model, components)
                }
                ModelLocation::HuggingFaceCache { cache_root, .. } => {
                    plan_hf_cache_delete(&model, cache_root)
                }
                ModelLocation::Directory { .. } | ModelLocation::File { .. } => Ok(DeletePlan {
                    model_id: id,
                    supported: false,
                    reason: Some(
                        "configured directories and ad-hoc files are read-only".to_owned(),
                    ),
                    reclaimable_bytes: 0,
                    retained_shared_bytes: 0,
                    paths: Vec::new(),
                }),
            }
        })
    }

    fn delete(&self, id: &ModelId) -> BoxFuture<'_, Result<DeletedModel, InventoryError>> {
        let id = id.clone();
        Box::pin(async move {
            let lock_path = self
                .config
                .root
                .join("locks")
                .join(format!("{}.lock", id.0));
            let lock = acquire_delete_lock(&lock_path)?;
            let model = self
                .models
                .read()
                .map_err(|_| InventoryError::Internal("inventory lock poisoned".to_owned()))?
                .get(&id)
                .cloned()
                .ok_or_else(|| InventoryError::NotFound(id.0.clone()))?;
            ensure_deletable_status(&model)?;
            if matches!(model.status, ModelStatus::Interrupted { .. }) {
                let plan = plan_interrupted_delete(&self.config.root, &model)?;
                let freed_bytes = delete_interrupted(&self.config.root, &model)?;
                drop(lock);
                self.remove_published_model(&id).await?;
                return Ok(DeletedModel {
                    id: id.clone(),
                    deleted: true,
                    freed_bytes,
                    retained_shared_bytes: plan.retained_shared_bytes,
                    plan,
                });
            }
            let plan = match &model.location {
                ModelLocation::MagnitudeCache { components, .. } => {
                    plan_managed_delete(&self.config.root, &model, components)?
                }
                ModelLocation::HuggingFaceCache { cache_root, .. } => {
                    plan_hf_cache_delete(&model, cache_root)?
                }
                ModelLocation::Directory { .. } | ModelLocation::File { .. } => {
                    return Err(InventoryError::Unsupported(
                        "configured directories and ad-hoc files are read-only".to_owned(),
                    ));
                }
            };
            if !plan.supported {
                return Err(InventoryError::Unsupported(
                    plan.reason
                        .clone()
                        .unwrap_or_else(|| "deletion unsupported".to_owned()),
                ));
            }
            let freed_bytes = match &model.location {
                ModelLocation::MagnitudeCache { components, .. } => {
                    delete_managed(&self.config.root, &model, components)?
                }
                ModelLocation::HuggingFaceCache { cache_root, .. } => {
                    delete_hf_cache(&model, cache_root)?
                }
                ModelLocation::Directory { .. } | ModelLocation::File { .. } => unreachable!(),
            };
            drop(lock);
            self.remove_published_model(&id).await?;
            Ok(DeletedModel {
                id: id.clone(),
                deleted: true,
                freed_bytes,
                retained_shared_bytes: plan.retained_shared_bytes,
                plan,
            })
        })
    }

    fn resolve_ready(&self, id: &ModelId) -> BoxFuture<'_, Result<ResolvedModel, InventoryError>> {
        let id = id.clone();
        Box::pin(async move {
            let model = self
                .models
                .read()
                .map_err(|_| InventoryError::Internal("inventory lock poisoned".to_owned()))?
                .get(&id)
                .cloned()
                .ok_or_else(|| InventoryError::NotFound(id.0.clone()))?;
            if !matches!(
                model.status,
                ModelStatus::Available { .. } | ModelStatus::Loaded { .. }
            ) {
                return Err(InventoryError::NotReady(id.0.clone()));
            }
            let components = resolve_components(&self.config.root, &model)?;
            Ok(ResolvedModel { model, components })
        })
    }

    fn update_status(
        &self,
        id: &ModelId,
        status: ModelStatus,
    ) -> BoxFuture<'_, Result<(), InventoryError>> {
        let id = id.clone();
        Box::pin(async move {
            let mut models = self
                .models
                .write()
                .map_err(|_| InventoryError::Internal("inventory lock poisoned".to_owned()))?;
            let model = models
                .get_mut(&id)
                .ok_or_else(|| InventoryError::NotFound(id.0.clone()))?;
            model.status = status;
            model.operations = operations_for(model);
            model.updated_at = now();
            Ok(())
        })
    }
}

fn operations_for(model: &InventoryModel) -> Vec<icn_contracts::ModelOperation> {
    use icn_contracts::ModelOperation;
    match model.status {
        ModelStatus::Available { .. } | ModelStatus::LoadFailed { .. } => {
            let mut operations = vec![ModelOperation::Load];
            if matches!(
                model.location,
                ModelLocation::MagnitudeCache { .. } | ModelLocation::HuggingFaceCache { .. }
            ) {
                operations.push(ModelOperation::Delete);
            }
            operations
        }
        ModelStatus::Loaded { .. } => vec![ModelOperation::Unload],
        ModelStatus::Interrupted { .. } => vec![ModelOperation::Delete],
        ModelStatus::InvalidArtifact { .. } | ModelStatus::IncompatibleArtifact { .. } => {
            if matches!(
                model.location,
                ModelLocation::MagnitudeCache { .. } | ModelLocation::HuggingFaceCache { .. }
            ) {
                vec![ModelOperation::Delete]
            } else {
                Vec::new()
            }
        }
        ModelStatus::Downloading { .. }
        | ModelStatus::Loading { .. }
        | ModelStatus::Unloading { .. } => Vec::new(),
    }
}

fn plan_interrupted_delete(
    root: &Path,
    model: &InventoryModel,
) -> Result<DeletePlan, InventoryError> {
    let operation = root.join("operations").join(format!("{}.json", model.id.0));
    if !operation.is_file() {
        return Err(InventoryError::DeletionUnsafe(
            "interrupted model is missing its operation manifest".to_owned(),
        ));
    }
    let mut reclaimable_bytes = 0_u64;
    let mut paths = vec![operation];
    if let ModelSource::HuggingFace { repository, .. } = &model.source {
        let blobs = root.join("hub").join(hf_repo_dir(repository)).join("blobs");
        for component in model.location.components() {
            let partial = blobs.join(format!("{}.incomplete", blob_key(&component.content)));
            if let Ok(metadata) = partial.symlink_metadata()
                && metadata.is_file()
            {
                reclaimable_bytes = reclaimable_bytes.saturating_add(metadata.len());
                paths.push(partial);
            }
        }
    }
    Ok(DeletePlan {
        model_id: model.id.clone(),
        supported: true,
        reason: None,
        reclaimable_bytes,
        retained_shared_bytes: 0,
        paths,
    })
}

fn delete_interrupted(root: &Path, model: &InventoryModel) -> Result<u64, InventoryError> {
    let plan = plan_interrupted_delete(root, model)?;
    let operation = root.join("operations").join(format!("{}.json", model.id.0));
    let tombstone = root
        .join("trash")
        .join(format!("operation-{}.{}.json", model.id.0, now()));
    fs::rename(&operation, &tombstone).map_err(io_error)?;
    let mut freed = 0_u64;
    for path in plan.paths.iter().skip(1) {
        if let Ok(metadata) = path.symlink_metadata() {
            if metadata.file_type().is_symlink() || !metadata.is_file() {
                return Err(InventoryError::DeletionUnsafe(format!(
                    "interrupted partial is not a regular file: {}",
                    path.display()
                )));
            }
            fs::remove_file(path).map_err(io_error)?;
            freed = freed.saturating_add(metadata.len());
        }
    }
    fs::remove_file(tombstone).map_err(io_error)?;
    Ok(freed)
}

pub(crate) fn reconcile_tombstones(root: &Path) -> Result<(), InventoryError> {
    let trash = root.join("trash");
    if !trash.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(&trash).map_err(io_error)? {
        let path = entry.map_err(io_error)?.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let Ok(bytes) = fs::read(&path) else {
            continue;
        };
        if path
            .file_name()
            .and_then(|value| value.to_str())
            .is_some_and(|value| value.starts_with("operation-"))
        {
            let Ok(manifest) = serde_json::from_slice::<OperationManifest>(&bytes) else {
                continue;
            };
            finish_operation_tombstone(root, &manifest, &path)?;
            continue;
        }
        let Ok(manifest) = serde_json::from_slice::<ManagedManifest>(&bytes) else {
            continue;
        };
        if manifest.validate().is_err() {
            continue;
        }
        let model = InventoryModel {
            id: manifest.model_id.clone(),
            content_id: manifest.content_id.clone(),
            created: manifest.created_at,
            name: manifest.repository.clone(),
            supported_parameters: Vec::new(),
            status: ModelStatus::Available {
                ready_at: manifest.ready_at,
            },
            source: ModelSource::HuggingFace {
                repository: manifest.repository.clone(),
                requested_revision: manifest.requested_revision.clone(),
                commit: manifest.commit.clone(),
                metadata: None,
            },
            location: ModelLocation::MagnitudeCache {
                total_bytes: manifest.components.iter().map(|item| item.size_bytes).sum(),
                components: manifest.components.clone(),
                integrity: icn_contracts::Integrity::Verified {
                    method: "manifest".to_owned(),
                },
            },
            properties: icn_contracts::InventoryProperties::Pending,
            hardware: icn_contracts::HardwareAssessment::NotAssessed {
                reason: "deleted".to_owned(),
            },
            operations: Vec::new(),
            updated_at: manifest.ready_at,
        };
        finish_managed_tombstone(root, &model, &manifest.components, &path)?;
    }
    Ok(())
}

fn finish_operation_tombstone(
    root: &Path,
    manifest: &OperationManifest,
    tombstone: &Path,
) -> Result<(), InventoryError> {
    if manifest.version != MANIFEST_VERSION
        || manifest.components.is_empty()
        || crate::validation::validate_repository(&manifest.repository).is_err()
        || manifest.components.iter().any(|component| {
            component.expected_size == 0
                || component.content_key != blob_key(&component.content)
                || crate::validation::validate_relative_path(&component.path).is_err()
        })
    {
        return Err(InventoryError::DeletionUnsafe(format!(
            "invalid interrupted-download tombstone: {}",
            tombstone.display()
        )));
    }
    let blobs = root
        .join("hub")
        .join(hf_repo_dir(&manifest.repository))
        .join("blobs");
    for component in &manifest.components {
        let partial = blobs.join(format!("{}.incomplete", component.content_key));
        let Ok(metadata) = partial.symlink_metadata() else {
            continue;
        };
        if metadata.is_file() {
            fs::remove_file(&partial).map_err(io_error)?;
        } else {
            let quarantine = root.join("quarantine").join(format!(
                "{}-{}-{}",
                manifest.model_id.0,
                component.content_key,
                now()
            ));
            fs::rename(&partial, quarantine).map_err(io_error)?;
        }
    }
    fs::remove_file(tombstone).map_err(io_error)
}

fn finish_managed_tombstone(
    root: &Path,
    model: &InventoryModel,
    components: &[icn_contracts::ModelComponent],
    tombstone: &Path,
) -> Result<(), InventoryError> {
    let referenced = other_managed_blob_keys(root, &model.id)?;
    let ModelSource::HuggingFace {
        repository, commit, ..
    } = &model.source
    else {
        return Err(InventoryError::Internal(
            "managed tombstone is missing Hugging Face identity".to_owned(),
        ));
    };
    let repo_root = root.join("hub").join(hf_repo_dir(repository));
    let snapshot = repo_root.join("snapshots").join(commit);
    for component in components {
        let link = snapshot.join(&component.path);
        if link.symlink_metadata().is_ok() {
            fs::remove_file(&link).map_err(io_error)?;
        }
        let key = blob_key(&component.content);
        if !referenced.contains(&key) {
            let blob = repo_root.join("blobs").join(key);
            if blob.is_file() {
                fs::remove_file(blob).map_err(io_error)?;
            }
        }
    }
    remove_empty_parents(&snapshot, &repo_root.join("snapshots"));
    fs::remove_file(tombstone).map_err(io_error)
}

fn status_rank(status: &ModelStatus) -> u8 {
    match status {
        ModelStatus::Loaded { .. } => 0,
        ModelStatus::Loading { .. } | ModelStatus::Unloading { .. } => 1,
        ModelStatus::Downloading { .. } => 2,
        ModelStatus::Interrupted { .. } => 3,
        ModelStatus::Available { .. } => 4,
        ModelStatus::InvalidArtifact { .. } | ModelStatus::IncompatibleArtifact { .. } => 5,
        ModelStatus::LoadFailed { .. } => 6,
    }
}

fn ensure_deletable_status(model: &InventoryModel) -> Result<(), InventoryError> {
    match model.status {
        ModelStatus::Downloading { .. } => Err(InventoryError::Busy(model.id.0.clone())),
        ModelStatus::Loading { .. }
        | ModelStatus::Loaded { .. }
        | ModelStatus::Unloading { .. } => Err(InventoryError::Loaded(model.id.0.clone())),
        ModelStatus::Interrupted { .. }
        | ModelStatus::Available { .. }
        | ModelStatus::InvalidArtifact { .. }
        | ModelStatus::IncompatibleArtifact { .. }
        | ModelStatus::LoadFailed { .. } => Ok(()),
    }
}

pub(crate) fn resolve_components(
    root: &Path,
    model: &InventoryModel,
) -> Result<Vec<ResolvedComponent>, InventoryError> {
    let (base, containment) = match (&model.location, &model.source) {
        (
            ModelLocation::MagnitudeCache { .. },
            ModelSource::HuggingFace {
                repository, commit, ..
            },
        ) => {
            let repository_root = root.join("hub").join(hf_repo_dir(repository));
            (
                repository_root.join("snapshots").join(commit),
                repository_root,
            )
        }
        (ModelLocation::HuggingFaceCache { cache_root, .. }, _) => {
            (cache_root.clone(), hf_repo_root(cache_root)?)
        }
        (ModelLocation::Directory { root, .. }, _) => (root.clone(), root.clone()),
        (ModelLocation::File { path, .. }, _) => {
            let parent = path
                .parent()
                .ok_or_else(|| InventoryError::Internal("ad-hoc model has no parent".to_owned()))?
                .to_path_buf();
            (parent.clone(), parent)
        }
        _ => {
            return Err(InventoryError::Internal(
                "model source and location are inconsistent".to_owned(),
            ));
        }
    };
    let canonical_containment = containment.canonicalize().map_err(io_error)?;
    model
        .location
        .components()
        .iter()
        .map(|component| {
            let path = match &model.location {
                ModelLocation::File { path, .. } => path.clone(),
                _ => base.join(&component.path),
            };
            let canonical = path.canonicalize().map_err(io_error)?;
            if !canonical.starts_with(&canonical_containment) {
                return Err(InventoryError::DeletionUnsafe(format!(
                    "model component escaped its source root: {}",
                    path.display()
                )));
            }
            Ok(ResolvedComponent {
                path: canonical,
                role: component.role.clone(),
                shard_index: component.shard_index,
                relationship: component.relationship.clone(),
            })
        })
        .collect()
}

fn plan_managed_delete(
    root: &Path,
    model: &InventoryModel,
    components: &[icn_contracts::ModelComponent],
) -> Result<DeletePlan, InventoryError> {
    let referenced = other_managed_blob_keys(root, &model.id)?;
    let mut reclaimable = 0_u64;
    let mut retained = 0_u64;
    let mut paths = vec![
        root.join("installations")
            .join(format!("{}.json", model.id.0)),
    ];
    let ModelSource::HuggingFace {
        repository, commit, ..
    } = &model.source
    else {
        return Err(InventoryError::Internal(
            "managed location is missing Hugging Face identity".to_owned(),
        ));
    };
    let repo_root = root.join("hub").join(hf_repo_dir(repository));
    let snapshot = repo_root.join("snapshots").join(commit);
    for component in components {
        paths.push(snapshot.join(&component.path));
        if referenced.contains(&blob_key(&component.content)) {
            retained = retained.saturating_add(component.size_bytes);
        } else {
            reclaimable = reclaimable.saturating_add(component.size_bytes);
            paths.push(repo_root.join("blobs").join(blob_key(&component.content)));
        }
    }
    Ok(DeletePlan {
        model_id: model.id.clone(),
        supported: true,
        reason: None,
        reclaimable_bytes: reclaimable,
        retained_shared_bytes: retained,
        paths,
    })
}

fn delete_managed(
    root: &Path,
    model: &InventoryModel,
    components: &[icn_contracts::ModelComponent],
) -> Result<u64, InventoryError> {
    let manifest = root
        .join("installations")
        .join(format!("{}.json", model.id.0));
    let tombstone = root
        .join("trash")
        .join(format!("{}.{}.json", model.id.0, now()));
    fs::rename(&manifest, &tombstone).map_err(io_error)?;
    let referenced = other_managed_blob_keys(root, &model.id)?;
    let ModelSource::HuggingFace {
        repository, commit, ..
    } = &model.source
    else {
        return Err(InventoryError::Internal(
            "managed location is missing Hugging Face identity".to_owned(),
        ));
    };
    let repo_root = root.join("hub").join(hf_repo_dir(repository));
    let snapshot = repo_root.join("snapshots").join(commit);
    let mut freed = 0_u64;
    for component in components {
        let link = snapshot.join(&component.path);
        if link.symlink_metadata().is_ok() {
            fs::remove_file(&link).map_err(io_error)?;
        }
        let key = blob_key(&component.content);
        if !referenced.contains(&key) {
            let blob = repo_root.join("blobs").join(key);
            if let Ok(metadata) = blob.metadata() {
                fs::remove_file(&blob).map_err(io_error)?;
                freed = freed.saturating_add(metadata.len());
            }
        }
    }
    remove_empty_parents(&snapshot, &repo_root.join("snapshots"));
    fs::remove_file(tombstone).map_err(io_error)?;
    Ok(freed)
}

fn other_managed_blob_keys(
    root: &Path,
    excluded: &ModelId,
) -> Result<BTreeSet<String>, InventoryError> {
    let mut keys = BTreeSet::new();
    let installations = root.join("installations");
    if !installations.is_dir() {
        return Ok(keys);
    }
    for entry in fs::read_dir(installations).map_err(io_error)? {
        let path = entry.map_err(io_error)?.path();
        let manifest = fs::read(&path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<ManagedManifest>(&bytes).ok());
        if let Some(manifest) = manifest
            && manifest.model_id != *excluded
        {
            keys.extend(
                manifest
                    .components
                    .iter()
                    .map(|component| blob_key(&component.content)),
            );
        }
    }
    Ok(keys)
}

fn plan_hf_cache_delete(
    model: &InventoryModel,
    snapshot: &Path,
) -> Result<DeletePlan, InventoryError> {
    let repo_root = hf_repo_root(snapshot)?;
    let (target_blobs, remaining_blobs) = hf_blob_reference_sets(&repo_root, snapshot)?;
    let mut reclaimable = 0_u64;
    let mut retained = 0_u64;
    let mut paths = vec![snapshot.to_path_buf()];
    for blob in target_blobs {
        let size = blob.metadata().map(|metadata| metadata.len()).unwrap_or(0);
        if remaining_blobs.contains(&blob) {
            retained = retained.saturating_add(size);
        } else {
            reclaimable = reclaimable.saturating_add(size);
            paths.push(blob);
        }
    }
    Ok(DeletePlan {
        model_id: model.id.clone(),
        supported: true,
        reason: None,
        reclaimable_bytes: reclaimable,
        retained_shared_bytes: retained,
        paths,
    })
}

fn delete_hf_cache(model: &InventoryModel, snapshot: &Path) -> Result<u64, InventoryError> {
    let repo_root = hf_repo_root(snapshot)?;
    let plan = plan_hf_cache_delete(model, snapshot)?;
    let commit = snapshot
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| InventoryError::DeletionUnsafe("snapshot has no commit name".to_owned()))?;
    remove_refs_to_commit(&repo_root.join("refs"), commit)?;
    let (target_blobs, remaining_blobs) = hf_blob_reference_sets(&repo_root, snapshot)?;
    fs::remove_dir_all(snapshot).map_err(io_error)?;
    let mut freed = 0_u64;
    for blob in target_blobs.difference(&remaining_blobs) {
        if let Ok(metadata) = blob.metadata() {
            fs::remove_file(blob).map_err(io_error)?;
            freed = freed.saturating_add(metadata.len());
        }
    }
    let snapshots = repo_root.join("snapshots");
    if fs::read_dir(&snapshots).map_err(io_error)?.next().is_none() {
        let _ = fs::remove_dir_all(&repo_root);
    }
    let _ = plan;
    Ok(freed)
}

fn hf_repo_root(snapshot: &Path) -> Result<PathBuf, InventoryError> {
    let canonical = snapshot.canonicalize().map_err(io_error)?;
    let snapshots = canonical.parent().ok_or_else(|| {
        InventoryError::DeletionUnsafe("snapshot has no snapshots root".to_owned())
    })?;
    if snapshots.file_name().and_then(|value| value.to_str()) != Some("snapshots") {
        return Err(InventoryError::DeletionUnsafe(
            "recognized Hugging Face snapshot is not under snapshots/".to_owned(),
        ));
    }
    snapshots
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| InventoryError::DeletionUnsafe("snapshot has no repository root".to_owned()))
}

fn hf_blob_reference_sets(
    repo_root: &Path,
    target_snapshot: &Path,
) -> Result<(BTreeSet<PathBuf>, BTreeSet<PathBuf>), InventoryError> {
    let mut target = BTreeSet::new();
    let mut remaining = BTreeSet::new();
    let snapshots = repo_root.join("snapshots");
    for entry in fs::read_dir(&snapshots).map_err(io_error)? {
        let snapshot = entry.map_err(io_error)?.path();
        let destination = if snapshot == target_snapshot {
            &mut target
        } else {
            &mut remaining
        };
        collect_snapshot_blobs(&snapshot, repo_root, destination)?;
    }
    Ok((target, remaining))
}

fn collect_snapshot_blobs(
    path: &Path,
    repo_root: &Path,
    output: &mut BTreeSet<PathBuf>,
) -> Result<(), InventoryError> {
    for entry in fs::read_dir(path).map_err(io_error)? {
        let entry = entry.map_err(io_error)?;
        let path = entry.path();
        let kind = entry.file_type().map_err(io_error)?;
        if kind.is_dir() {
            collect_snapshot_blobs(&path, repo_root, output)?;
        } else if kind.is_symlink() || kind.is_file() {
            let canonical = path.canonicalize().map_err(io_error)?;
            if !canonical.starts_with(repo_root.join("blobs")) {
                return Err(InventoryError::DeletionUnsafe(format!(
                    "snapshot entry does not resolve to repository blobs: {}",
                    path.display()
                )));
            }
            output.insert(canonical);
        }
    }
    Ok(())
}

fn remove_refs_to_commit(refs: &Path, commit: &str) -> Result<(), InventoryError> {
    if !refs.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(refs).map_err(io_error)? {
        let entry = entry.map_err(io_error)?;
        let path = entry.path();
        if entry.file_type().map_err(io_error)?.is_dir() {
            remove_refs_to_commit(&path, commit)?;
        } else if fs::read_to_string(&path).is_ok_and(|value| value.trim() == commit) {
            fs::remove_file(path).map_err(io_error)?;
        }
    }
    Ok(())
}

fn remove_empty_parents(path: &Path, stop: &Path) {
    let mut current = path.to_path_buf();
    while current.starts_with(stop) && current != stop {
        if fs::remove_dir(&current).is_err() {
            break;
        }
        let Some(parent) = current.parent() else {
            break;
        };
        current = parent.to_path_buf();
    }
}

fn acquire_delete_lock(path: &Path) -> Result<File, InventoryError> {
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(path)
        .map_err(io_error)?;
    FileExt::lock_exclusive(&file).map_err(io_error)?;
    Ok(file)
}

fn io_error(error: impl std::fmt::Display) -> InventoryError {
    InventoryError::Io(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{OperationComponent, OperationManifest};
    use icn_contracts::{ComponentRole, ContentId, ContentIdentity, ModelId};

    #[test]
    fn startup_finishes_interrupted_download_tombstone() {
        let root = tempfile::tempdir().expect("temp store");
        for directory in ["trash", "quarantine"] {
            fs::create_dir_all(root.path().join(directory)).expect("layout");
        }
        let repository = "owner/repo";
        let content = ContentIdentity::Sha256 {
            value: "a".repeat(64),
        };
        let key = blob_key(&content);
        let blobs = root
            .path()
            .join("hub")
            .join(hf_repo_dir(repository))
            .join("blobs");
        fs::create_dir_all(&blobs).expect("blobs");
        let partial = blobs.join(format!("{key}.incomplete"));
        fs::write(&partial, b"partial").expect("partial");
        let manifest = OperationManifest {
            version: MANIFEST_VERSION,
            operation_id: "download_test".to_owned(),
            model_id: ModelId("mdl_test".to_owned()),
            content_id: ContentId("content_test".to_owned()),
            repository: repository.to_owned(),
            requested_revision: "main".to_owned(),
            commit: "commit".to_owned(),
            components: vec![OperationComponent {
                path: PathBuf::from("model.gguf"),
                role: ComponentRole::Weights,
                content,
                shard_index: None,
                relationship: None,
                expected_size: 100,
                content_key: key,
                completed_bytes: 7,
            }],
            stage: "downloading".to_owned(),
            started_at: 1,
            updated_at: 1,
            last_error: None,
        };
        let tombstone = root.path().join("trash/operation-mdl_test.1.json");
        fs::write(
            &tombstone,
            serde_json::to_vec(&manifest).expect("manifest json"),
        )
        .expect("tombstone");

        reconcile_tombstones(root.path()).expect("reconcile");

        assert!(!partial.exists());
        assert!(!tombstone.exists());
    }
}
