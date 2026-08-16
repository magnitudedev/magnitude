use std::collections::BTreeSet;
use std::fs::{self, File};
use std::path::{Path, PathBuf};

use futures_util::future::BoxFuture;
use icn_contracts::{
    DeletePlan, DeletedModel, InventoryError, InventoryModel, ModelAvailability, ModelId,
    ModelInventory, ModelLocation, ModelSource, ResolvedComponent, ResolvedModel,
};

use crate::inventory::{ManagedModelStore, hf_repo_dir, repository_lock_path};
use crate::store_fs::acquire_exclusive_lock;

impl ModelInventory for ManagedModelStore {
    fn list(&self) -> BoxFuture<'_, Result<Vec<InventoryModel>, InventoryError>> {
        Box::pin(async move {
            let mut models = self
                .models
                .read()
                .map_err(|_| InventoryError::Internal("inventory lock poisoned".to_owned()))?
                .values()
                .cloned()
                .collect::<Vec<_>>();
            models.sort_by(|left, right| {
                status_rank(left)
                    .cmp(&status_rank(right))
                    .then_with(|| left.name.to_lowercase().cmp(&right.name.to_lowercase()))
                    .then_with(|| left.id.cmp(&right.id))
            });
            Ok(models)
        })
    }

    fn get(&self, id: &ModelId) -> BoxFuture<'_, Result<InventoryModel, InventoryError>> {
        let id = id.clone();
        Box::pin(async move {
            let model = self
                .models
                .read()
                .map_err(|_| InventoryError::Internal("inventory lock poisoned".to_owned()))?
                .get(&id)
                .cloned()
                .ok_or_else(|| InventoryError::NotFound(id.0.clone()))?;
            Ok(model)
        })
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
            self.ensure_installed_model_inventory().await?;
            let observed = self
                .models
                .read()
                .map_err(|_| InventoryError::Internal("inventory lock poisoned".to_owned()))?
                .get(&id)
                .cloned()
                .ok_or_else(|| InventoryError::NotFound(id.0.clone()))?;
            let lock_path = match (&observed.location, &observed.source) {
                (
                    ModelLocation::MagnitudeCache { .. },
                    ModelSource::HuggingFace { repository, .. },
                ) => repository_lock_path(&self.config.root, repository),
                _ => self
                    .config
                    .root
                    .join("locks")
                    .join(format!("{}.lock", id.0)),
            };
            let lock = acquire_delete_lock(&lock_path)?;
            let model = self
                .models
                .read()
                .map_err(|_| InventoryError::Internal("inventory lock poisoned".to_owned()))?
                .get(&id)
                .cloned()
                .ok_or_else(|| InventoryError::NotFound(id.0.clone()))?;
            ensure_deletable_status(&model)?;
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
            if !matches!(model.availability, ModelAvailability::Available { .. }) {
                return Err(InventoryError::NotReady(id.0.clone()));
            }
            let components = resolve_components(&self.config.root, &model)?;
            Ok(ResolvedModel { model, components })
        })
    }
}

fn status_rank(model: &InventoryModel) -> u8 {
    match &model.availability {
        ModelAvailability::Available { .. } => 0,
        ModelAvailability::Downloading { .. } => 1,
        ModelAvailability::Interrupted { .. } => 2,
        ModelAvailability::InvalidArtifact { .. }
        | ModelAvailability::IncompatibleArtifact { .. } => 3,
    }
}

fn ensure_deletable_status(model: &InventoryModel) -> Result<(), InventoryError> {
    match model.availability {
        ModelAvailability::Downloading { .. } => Err(InventoryError::Busy(model.id.0.clone())),
        _ => Ok(()),
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
    let resolved = model
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
                path,
                role: component.role.clone(),
                shard_index: component.shard_index,
                relationship: component.relationship.clone(),
            })
        })
        .collect::<Result<Vec<_>, InventoryError>>()?;
    validate_shard_layout(&resolved)?;
    Ok(resolved)
}

fn validate_shard_layout(components: &[ResolvedComponent]) -> Result<(), InventoryError> {
    let mut groups = Vec::<(icn_contracts::ComponentRole, PathBuf)>::new();
    for component in components
        .iter()
        .filter(|component| component.shard_index.is_some())
    {
        let directory = component.path.parent().unwrap_or_else(|| Path::new(""));
        if !groups
            .iter()
            .any(|(role, parent)| role == &component.role && parent == directory)
        {
            groups.push((component.role.clone(), directory.to_path_buf()));
        }
    }
    for (role, directory) in groups {
        let shards = components
            .iter()
            .filter(|component| {
                component.role == role
                    && component.shard_index.is_some()
                    && component.path.parent() == Some(directory.as_path())
            })
            .collect::<Vec<_>>();
        let count = u32::try_from(shards.len()).map_err(|_| invalid_split_layout(&role))?;
        let mut indices = BTreeSet::new();
        for component in shards {
            let index = component
                .shard_index
                .expect("sharded components were filtered");
            let name = component
                .path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or_default();
            let suffix = format!("-{index:05}-of-{count:05}.gguf");
            if !indices.insert(index) || !name.ends_with(&suffix) {
                return Err(invalid_split_layout(&role));
            }
        }
        if indices != (1..=count).collect() {
            return Err(invalid_split_layout(&role));
        }
    }
    Ok(())
}

fn invalid_split_layout(role: &icn_contracts::ComponentRole) -> InventoryError {
    InventoryError::ModelOperation {
        code: "invalid_split_layout".to_owned(),
        message: format!("the {role:?} GGUF shard layout is invalid").to_ascii_lowercase(),
        retryable: false,
    }
}

fn plan_managed_delete(
    root: &Path,
    model: &InventoryModel,
    components: &[icn_contracts::ModelComponent],
) -> Result<DeletePlan, InventoryError> {
    let mut reclaimable = 0_u64;
    let mut retained = 0_u64;
    let mut paths = Vec::new();
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
    let links = components
        .iter()
        .map(|component| snapshot.join(&component.path))
        .collect::<BTreeSet<_>>();
    let referenced = other_snapshot_blob_references(&repo_root, &links)?;
    for component in components {
        let link = snapshot.join(&component.path);
        let blob = link.canonicalize().map_err(io_error)?;
        paths.push(link);
        if referenced.contains(&blob) {
            retained = retained.saturating_add(component.size_bytes);
        } else {
            reclaimable = reclaimable.saturating_add(component.size_bytes);
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

fn delete_managed(
    root: &Path,
    model: &InventoryModel,
    components: &[icn_contracts::ModelComponent],
) -> Result<u64, InventoryError> {
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
    let links = components
        .iter()
        .map(|component| snapshot.join(&component.path))
        .collect::<BTreeSet<_>>();
    let referenced = other_snapshot_blob_references(&repo_root, &links)?;
    let mut freed = 0_u64;
    for component in components {
        let link = snapshot.join(&component.path);
        let blob = link.canonicalize().map_err(io_error)?;
        if link.symlink_metadata().is_ok() {
            fs::remove_file(&link).map_err(io_error)?;
        }
        if !referenced.contains(&blob) {
            if let Ok(metadata) = blob.metadata() {
                fs::remove_file(blob).map_err(io_error)?;
                freed = freed.saturating_add(metadata.len());
            }
        }
    }
    remove_empty_parents(&snapshot, &repo_root.join("snapshots"));
    Ok(freed)
}

fn other_snapshot_blob_references(
    repo_root: &Path,
    excluded_links: &BTreeSet<PathBuf>,
) -> Result<BTreeSet<PathBuf>, InventoryError> {
    let mut references = BTreeSet::new();
    collect_other_snapshot_blobs(
        &repo_root.join("snapshots"),
        repo_root,
        excluded_links,
        &mut references,
    )?;
    Ok(references)
}

fn collect_other_snapshot_blobs(
    path: &Path,
    repo_root: &Path,
    excluded_links: &BTreeSet<PathBuf>,
    output: &mut BTreeSet<PathBuf>,
) -> Result<(), InventoryError> {
    if !path.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(path).map_err(io_error)? {
        let entry = entry.map_err(io_error)?;
        let path = entry.path();
        let kind = entry.file_type().map_err(io_error)?;
        if kind.is_dir() {
            collect_other_snapshot_blobs(&path, repo_root, excluded_links, output)?;
        } else if !excluded_links.contains(&path) && (kind.is_symlink() || kind.is_file()) {
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
    acquire_exclusive_lock(path)
}

fn io_error(error: impl std::fmt::Display) -> InventoryError {
    InventoryError::Io(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use icn_contracts::{
        ComponentRole, ContentId, ContentIdentity, Integrity, InventoryProperties, ModelComponent,
        ModelId,
    };

    #[cfg(unix)]
    #[test]
    fn managed_shards_retain_snapshot_paths_after_containment_validation() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().expect("temp store");
        let repository = "owner/repo";
        let commit = "commit";
        let repository_root = root.path().join("hub").join(hf_repo_dir(repository));
        let snapshot = repository_root
            .join("snapshots")
            .join(commit)
            .join("UD-Q6_K_XL");
        let blobs = repository_root.join("blobs");
        fs::create_dir_all(&snapshot).expect("snapshot");
        fs::create_dir_all(&blobs).expect("blobs");

        let components = (1..=4)
            .map(|index| {
                let name = format!("model-{index:05}-of-00004.gguf");
                let blob_name = format!("sha256-{index}");
                fs::write(blobs.join(&blob_name), format!("shard {index}")).expect("write blob");
                symlink(
                    Path::new("../../../blobs").join(&blob_name),
                    snapshot.join(&name),
                )
                .expect("snapshot symlink");
                ModelComponent {
                    path: PathBuf::from("UD-Q6_K_XL").join(name),
                    role: ComponentRole::Shard,
                    size_bytes: 7,
                    content: ContentIdentity::Sha256 {
                        value: format!("{index:064x}"),
                    },
                    shard_index: Some(index),
                    relationship: None,
                }
            })
            .collect::<Vec<_>>();
        let model = InventoryModel {
            id: ModelId("mdl_split".to_owned()),
            content_id: ContentId("content_split".to_owned()),
            created: 1,
            name: "split".to_owned(),
            supported_parameters: Vec::new(),
            availability: ModelAvailability::Available { ready_at: 1 },
            source: ModelSource::HuggingFace {
                repository: repository.to_owned(),
                requested_revision: commit.to_owned(),
                commit: commit.to_owned(),
                metadata: None,
            },
            location: ModelLocation::MagnitudeCache {
                components,
                total_bytes: 28,
                integrity: Integrity::Verified {
                    method: "test".to_owned(),
                },
            },
            properties: InventoryProperties::Pending,
            operations: Vec::new(),
            updated_at: 1,
        };

        let resolved = resolve_components(root.path(), &model).expect("resolve shards");

        assert_eq!(resolved.len(), 4);
        assert_eq!(resolved[0].path, snapshot.join("model-00001-of-00004.gguf"));
        assert_ne!(
            resolved[0].path,
            resolved[0].path.canonicalize().expect("canonical blob")
        );
    }

    #[test]
    fn invalid_split_filename_is_rejected_before_native_planning() {
        let directory = tempfile::tempdir().expect("shards");
        let components = (1..=2)
            .map(|index| {
                let path = directory.path().join(format!("blob-{index}"));
                fs::write(&path, b"fixture").expect("fixture");
                ResolvedComponent {
                    path,
                    role: ComponentRole::Shard,
                    shard_index: Some(index),
                    relationship: None,
                }
            })
            .collect::<Vec<_>>();

        let error = validate_shard_layout(&components).expect_err("invalid split layout");

        assert!(matches!(
            error,
            InventoryError::ModelOperation { code, retryable: false, .. }
                if code == "invalid_split_layout"
        ));
    }
}
