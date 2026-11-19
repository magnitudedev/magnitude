use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

use futures_util::future::BoxFuture;
use magnitude_service_contracts::{
    DeletePlan, ExternalOrigin, ModelComponent, InventoryEntryId, InventoryError, InventoryModel, ModelAvailability,
    ModelInventory, ModelLocation, ModelSource, ResolvedComponent, ResolvedModel,
};

use crate::inventory::{LinkedFiles, ManagedModelStore};
use crate::snapshot_blobs::{BlobStorage, RepositoryBlobs};
use crate::store_fs::{Garbage, RepositoryGarbage, StoredFile, finish, model_store, repository_record, snapshot_entry};

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

    fn get(&self, id: &InventoryEntryId) -> BoxFuture<'_, Result<InventoryModel, InventoryError>> {
        let id = id.clone();
        Box::pin(async move { self.published_model(&id) })
    }

    fn plan_delete(
        &self,
        id: &InventoryEntryId,
    ) -> BoxFuture<'_, Result<DeletePlan, InventoryError>> {
        let id = id.clone();
        Box::pin(async move {
            let model = self.published_model(&id)?;
            ensure_deletable_status(&model)?;
            match &model.location {
                ModelLocation::MagnitudeCache { components, .. } => {
                    plan_managed_delete(&self.config.root, &model, components)
                }
                ModelLocation::HuggingFaceCache { .. }
                | ModelLocation::External { .. }
                | ModelLocation::Directory { .. }
                | ModelLocation::File { .. } => Ok(DeletePlan {
                    model_id: id,
                    supported: false,
                    reason: Some(EXTERNAL_READ_ONLY.to_owned()),
                    reclaimable_bytes: 0,
                    retained_shared_bytes: 0,
                    paths: Vec::new(),
                }),
            }
        })
    }

    fn resolve_ready(
        &self,
        id: &InventoryEntryId,
    ) -> BoxFuture<'_, Result<ResolvedModel, InventoryError>> {
        let id = id.clone();
        Box::pin(async move {
            let model = self.published_model(&id)?;
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

/// The files a model is served from, each contained in the root its location scans. A file found
/// in a user folder (or listed by its own path, which its folder contains) or in LM Studio's folder
/// lies there itself or as a link to anywhere. Ollama's files are the blobs its manifests name,
/// read through whatever links lead to them, as people move Ollama's storage to another drive by
/// linking a blob, its blobs folder or the store.
pub(crate) fn resolve_components(
    root: &Path,
    model: &InventoryModel,
) -> Result<Vec<ResolvedComponent>, InventoryError> {
    let (base, containment, linked_files) = match (&model.location, &model.source) {
        (
            ModelLocation::MagnitudeCache { .. },
            ModelSource::HuggingFace {
                repository, commit, ..
            },
        ) => {
            let repository_root = root.join(repository_record(repository));
            (
                repository_root.join("snapshots").join(commit),
                Some(repository_root),
                LinkedFiles::WithinRoot,
            )
        }
        (ModelLocation::HuggingFaceCache { cache_root, .. }, _) => (
            cache_root.clone(),
            Some(hf_hub_root(cache_root)?),
            LinkedFiles::WithinRoot,
        ),
        (
            ModelLocation::External {
                origin: ExternalOrigin::Folder | ExternalOrigin::LmStudio { .. },
                root,
                ..
            },
            _,
        ) => (root.clone(), Some(root.clone()), LinkedFiles::Anywhere),
        (
            ModelLocation::External {
                origin: ExternalOrigin::Ollama { .. },
                root,
                ..
            },
            _,
        ) => (root.clone(), None, LinkedFiles::Anywhere),
        (ModelLocation::Directory { root, .. } | ModelLocation::External { root, .. }, _) => {
            (root.clone(), Some(root.clone()), LinkedFiles::WithinRoot)
        }
        (ModelLocation::File { path, .. }, _) => {
            let parent = path
                .parent()
                .ok_or_else(|| InventoryError::Internal("ad-hoc model has no parent".to_owned()))?
                .to_path_buf();
            (parent.clone(), Some(parent), LinkedFiles::WithinRoot)
        }
        _ => {
            return Err(InventoryError::Internal(
                "model source and location are inconsistent".to_owned(),
            ));
        }
    };
    let canonical_containment = containment
        .map(|containment| containment.canonicalize().map_err(io_error))
        .transpose()?;
    let resolved = model
        .location
        .components()
        .iter()
        .map(|component| {
            let path = match &model.location {
                ModelLocation::File { path, .. } => path.clone(),
                _ => base.join(&component.path),
            };
            if let Some(containment) = &canonical_containment {
                let contained = match (linked_files, path.parent(), path.file_name()) {
                    (LinkedFiles::Anywhere, Some(parent), Some(name)) => {
                        parent.canonicalize().map_err(io_error)?.join(name)
                    }
                    _ => path.canonicalize().map_err(io_error)?,
                };
                if !contained.starts_with(containment) {
                    return Err(InventoryError::DeletionUnsafe(format!(
                        "model component escaped its source root: {}",
                        path.display()
                    )));
                }
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
    let mut groups = Vec::<(magnitude_service_contracts::ComponentRole, PathBuf)>::new();
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

fn invalid_split_layout(role: &magnitude_service_contracts::ComponentRole) -> InventoryError {
    InventoryError::ModelOperation {
        code: "invalid_split_layout".to_owned(),
        message: format!("the {role:?} GGUF shard layout is invalid").to_ascii_lowercase(),
        retryable: false,
    }
}

/// A file of a Magnitude snapshot: its repository, commit and path within the snapshot.
type SnapshotFile = (String, String, PathBuf);

impl ManagedModelStore {
    /// Deletes Magnitude's copies of `ids` as one removal: the bytes freed. A snapshot file stays
    /// while a model outside `ids` is served from it (a variant's target holding the projector
    /// its sibling variant shares); a file several of `ids` hold goes with the first of them.
    ///
    /// The removal is its files' transitions in their owner ([`crate::store_fs::Readers`]): it
    /// takes them from every acquirer
    /// at once (T3), refused while a download, another process or another removal holds one, and
    /// finds what it deletes; `release_resident` lets go of the model loading or serving them
    /// (T4); it waits for the holds left here, which are finite (T6); commits in one durable
    /// write, its garbage record, which removes the models (T7); and deletes the garbage (T8).
    /// Every refusal or failure before the commit drops the removal, which leaves everything
    /// installed (T5).
    pub(crate) async fn remove_entries(
        &self,
        ids: &[InventoryEntryId],
        release_resident: BoxFuture<'_, Result<(), InventoryError>>,
    ) -> Result<u64, InventoryError> {
        let root = &self.config.root;
        let removing = ids.iter().collect::<BTreeSet<_>>();
        let in_use = self
            .models
            .read()
            .map_err(|_| InventoryError::Internal("inventory lock poisoned".to_owned()))?
            .values()
            .filter(|model| !removing.contains(&model.id))
            .flat_map(|model| snapshot_files(model).into_iter().map(|(file, _)| file))
            .collect::<BTreeSet<_>>();
        let (mut removals, mut deleted_files, mut repositories, mut files) =
            (Vec::new(), BTreeSet::new(), BTreeSet::new(), BTreeSet::new());
        for id in ids {
            let model = self.published_model(id)?;
            ensure_deletable_status(&model)?;
            // Magnitude deletes only from its own store. Every other location belongs to the
            // user or another tool, whatever the request.
            let (ModelLocation::MagnitudeCache { .. }, ModelSource::HuggingFace { repository, .. }) =
                (&model.location, &model.source)
            else {
                return Err(InventoryError::Unsupported(EXTERNAL_READ_ONLY.to_owned()));
            };
            repositories.insert(repository_record(repository));
            let mut owned = Vec::new();
            for (file, component) in snapshot_files(&model) {
                if !in_use.contains(&file) && deleted_files.insert(file.clone()) {
                    owned.push(component.clone());
                    let (repository, commit, path) = &file;
                    files.insert(StoredFile::new(repository, commit, path));
                }
            }
            removals.push((model, owned));
        }
        let removal = self.readers.begin_removal(root, &repositories, &files).await?;
        let garbage = magnitude_service_utils::blocking::run({
            let root = root.clone();
            move || garbage(&root, &removals)
        })
        .await?;
        release_resident.await?;
        removal.drained().await;
        let store = model_store(root);
        let garbage = magnitude_service_utils::blocking::run(move || garbage.commit(&store).map(|()| garbage)).await?;
        // The commit removed the models: the inventory here takes it in at once, as every scan does.
        self.remove_published_models(ids).await?;
        let store = model_store(root);
        let freed_bytes = magnitude_service_utils::blocking::run(move || {
            garbage.0.keys().fold(0_u64, |freed, repository| match finish(&store, repository) {
                Ok(bytes) => freed.saturating_add(bytes),
                Err(error) => {
                    tracing::warn!(repository = %repository.display(), %error, "removed model garbage is left for the repository's next holder");
                    freed
                }
            })
        })
        .await;
        drop(removal);
        Ok(freed_bytes)
    }

    fn published_model(&self, id: &InventoryEntryId) -> Result<InventoryModel, InventoryError> {
        self.models
            .read()
            .map_err(|_| InventoryError::Internal("inventory lock poisoned".to_owned()))?
            .get(id)
            .cloned()
            .ok_or_else(|| InventoryError::NotFound(id.0.clone()))
    }
}

/// The snapshot files a Magnitude-stored model is served from, with their components.
fn snapshot_files(model: &InventoryModel) -> Vec<(SnapshotFile, &magnitude_service_contracts::ModelComponent)> {
    match (&model.location, &model.source) {
        (
            ModelLocation::MagnitudeCache { components, .. },
            ModelSource::HuggingFace {
                repository, commit, ..
            },
        ) => components
            .iter()
            .map(|component| ((repository.clone(), commit.clone(), component.path.clone()), component))
            .collect(),
        _ => Vec::new(),
    }
}

fn plan_managed_delete(
    root: &Path,
    model: &InventoryModel,
    components: &[magnitude_service_contracts::ModelComponent],
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
    let entries = managed_snapshot_entries(root, repository, commit, components)?;
    let repository_root = root.join(repository_record(repository));
    for (component, ManagedSnapshotEntry { link, blob }) in components.iter().zip(entries.entries) {
        paths.push(link);
        if entries.referenced.contains(&blob) {
            retained = retained.saturating_add(component.size_bytes);
        } else {
            reclaimable = reclaimable.saturating_add(component.size_bytes);
            paths.push(repository_root.join(blob));
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

/// The garbage of removing `removals`' components ([`Garbage`]): every entry must resolve to a
/// blob of its repository, and a blob another entry still publishes stays. Its caller holds the
/// repositories' records. Each path is named under the store's root by the store's layout, as
/// given, whatever links the root or the repository is reached through.
fn garbage(root: &Path, removals: &[(InventoryModel, Vec<ModelComponent>)]) -> Result<Garbage, InventoryError> {
    // By repository record: its blobs, the entries removed, and their blobs within it.
    let mut opened = BTreeMap::<PathBuf, (RepositoryBlobs, BTreeSet<PathBuf>, BTreeSet<PathBuf>)>::new();
    for (model, owned) in removals {
        let ModelSource::HuggingFace { repository, commit, .. } = &model.source else {
            return Err(InventoryError::Internal("managed location is missing Hugging Face identity".to_owned()));
        };
        let record = repository_record(repository);
        let (blobs, entries, removed) = match opened.entry(record.clone()) {
            std::collections::btree_map::Entry::Occupied(found) => found.into_mut(),
            std::collections::btree_map::Entry::Vacant(vacant) => {
                let blobs = RepositoryBlobs::open(&root.join(&record)).map_err(io_error)?;
                vacant.insert((blobs, BTreeSet::new(), BTreeSet::new()))
            }
        };
        for component in owned {
            let entry = snapshot_entry(repository, commit, &component.path);
            removed.insert(repository_blob(blobs, &root.join(&entry))?);
            entries.insert(entry);
        }
    }
    let mut garbage = Garbage::default();
    for (record, (blobs, entries, removed)) in opened {
        let links = entries.iter().map(|entry| root.join(entry)).collect::<BTreeSet<_>>();
        let mut referenced = BTreeSet::new();
        collect_other_snapshot_blobs(&root.join(&record).join("snapshots"), &blobs, &links, &mut referenced)?;
        let freed = removed.difference(&referenced).map(|blob| record.join(blob)).collect();
        garbage.0.insert(record, RepositoryGarbage { entries, blobs: freed });
    }
    Ok(garbage)
}

struct ManagedSnapshotEntry {
    link: PathBuf,
    blob: PathBuf,
}

struct ManagedSnapshotEntries {
    entries: Vec<ManagedSnapshotEntry>,
    referenced: BTreeSet<PathBuf>,
}

/// Resolve a managed package's snapshot entries to their blobs, plus the blobs other snapshot
/// entries of the repository still publish. Every entry must resolve to a repository blob.
fn managed_snapshot_entries(
    root: &Path,
    repository: &str,
    commit: &str,
    components: &[magnitude_service_contracts::ModelComponent],
) -> Result<ManagedSnapshotEntries, InventoryError> {
    let repo_root = root.join(repository_record(repository));
    let snapshot = repo_root.join("snapshots").join(commit);
    let blobs = RepositoryBlobs::open(&repo_root).map_err(io_error)?;
    let entries = components
        .iter()
        .map(|component| {
            let link = snapshot.join(&component.path);
            let blob = repository_blob(&blobs, &link)?;
            Ok(ManagedSnapshotEntry { link, blob })
        })
        .collect::<Result<Vec<_>, InventoryError>>()?;
    let links = entries
        .iter()
        .map(|entry| entry.link.clone())
        .collect::<BTreeSet<_>>();
    let mut referenced = BTreeSet::new();
    collect_other_snapshot_blobs(
        &repo_root.join("snapshots"),
        &blobs,
        &links,
        &mut referenced,
    )?;
    Ok(ManagedSnapshotEntries {
        entries,
        referenced,
    })
}

/// The blob a managed snapshot entry owns, by its path within its repository. Magnitude deletes
/// only bytes its repository keeps itself, never a blob store shared with other repositories.
fn repository_blob(blobs: &RepositoryBlobs, link: &Path) -> Result<PathBuf, InventoryError> {
    let blob = blobs.resolve(link).map_err(io_error)?.ok_or_else(|| {
        InventoryError::DeletionUnsafe(format!(
            "snapshot entry does not resolve to repository blobs: {}",
            link.display()
        ))
    })?;
    match blob.storage {
        BlobStorage::Repository => Ok(blob.within_repository()),
        BlobStorage::Shared => Err(InventoryError::DeletionUnsafe(format!(
            "snapshot entry links into a shared blob store: {}",
            link.display()
        ))),
    }
}

fn collect_other_snapshot_blobs(
    path: &Path,
    blobs: &RepositoryBlobs,
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
            collect_other_snapshot_blobs(&path, blobs, excluded_links, output)?;
        } else if !excluded_links.contains(&path) && (kind.is_symlink() || kind.is_file()) {
            output.insert(repository_blob(blobs, &path)?);
        }
    }
    Ok(())
}

const EXTERNAL_READ_ONLY: &str =
    "files outside Magnitude's model store are read-only and are never deleted";

/// The hub a recognized Hugging Face snapshot belongs to: its files may resolve into the blob
/// store the hub shares between repositories.
fn hf_hub_root(snapshot: &Path) -> Result<PathBuf, InventoryError> {
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
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .ok_or_else(|| InventoryError::DeletionUnsafe("snapshot has no hub root".to_owned()))
}

pub(crate) fn remove_empty_parents(path: &Path, stop: &Path) {
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

fn io_error(error: impl std::fmt::Display) -> InventoryError {
    InventoryError::Io(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inventory::hf_repo_dir;
    use magnitude_service_contracts::{
        ComponentRole, ContentId, ContentIdentity, Integrity, InventoryEntryId,
        InventoryProperties, ModelComponent,
    };

    #[cfg(unix)]
    #[test]
    fn managed_deletion_refuses_entries_that_link_into_a_shared_blob_store() {
        use std::os::unix::fs::symlink;

        let root = tempfile::tempdir().expect("temp store");
        let repository = "owner/repo";
        let hub = root.path().join("hub");
        let repository_root = hub.join(hf_repo_dir(repository));
        let snapshot = repository_root.join("snapshots/commit");
        fs::create_dir_all(repository_root.join("blobs")).expect("blobs");
        fs::create_dir_all(hub.join("blobs/ab")).expect("shared store");
        fs::create_dir_all(&snapshot).expect("snapshot");
        fs::write(hub.join("blobs/ab/abcd"), b"weights").expect("shared blob");
        symlink("../../blobs/ab/abcd", repository_root.join("blobs/hash")).expect("blob link");
        symlink("../../blobs/hash", snapshot.join("model.gguf")).expect("snapshot link");
        let component = ModelComponent {
            path: PathBuf::from("model.gguf"),
            role: ComponentRole::Weights,
            size_bytes: 7,
            content: ContentIdentity::Sha256 {
                value: "0".repeat(64),
            },
            shard_index: None,
            relationship: None,
        };

        let refused = managed_snapshot_entries(root.path(), repository, "commit", &[component]);

        assert!(matches!(refused, Err(InventoryError::DeletionUnsafe(_))));
        assert!(hub.join("blobs/ab/abcd").is_file());
    }

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
        let model = managed_model(repository, commit, components);

        let resolved = resolve_components(root.path(), &model).expect("resolve shards");

        assert_eq!(resolved.len(), 4);
        assert_eq!(resolved[0].path, snapshot.join("model-00001-of-00004.gguf"));
        assert_ne!(
            resolved[0].path,
            resolved[0].path.canonicalize().expect("canonical blob")
        );
    }

    /// A removal names the files it deletes under the store's root, as the store lays them out,
    /// whatever form their blobs resolve in: through a link to the models folder, and on Windows
    /// in the verbatim form (`\\?\C:\…`) every resolved path takes there and a root as given
    /// never does. The snapshot entries are hard links, as Windows makes them.
    #[test]
    fn garbage_names_the_store_files_whatever_form_the_root_resolves_in() {
        let temporary = tempfile::tempdir().expect("temporary root");
        let store = temporary.path().join("store");
        #[cfg(unix)]
        let root = {
            let root = temporary.path().join("models");
            std::os::unix::fs::symlink(&store, &root).expect("models folder link");
            root
        };
        #[cfg(not(unix))]
        let root = store.clone();
        let repository = "owner/repo";
        let record = repository_record(repository);
        let blobs = store.join(&record).join("blobs");
        fs::create_dir_all(&blobs).expect("blobs");
        fs::write(blobs.join("weights"), b"weights").expect("weights blob");
        fs::write(blobs.join("projector"), b"projector").expect("projector blob");
        // The older snapshot still publishes the projector, so its blob stays.
        for (commit, file, blob) in [
            ("commit", "model.gguf", "weights"),
            ("commit", "mmproj.gguf", "projector"),
            ("older", "mmproj.gguf", "projector"),
        ] {
            let entry = store.join(snapshot_entry(repository, commit, Path::new(file)));
            fs::create_dir_all(entry.parent().expect("snapshot")).expect("snapshot");
            fs::hard_link(blobs.join(blob), entry).expect("snapshot entry");
        }
        let components = [("model.gguf", ComponentRole::Weights), ("mmproj.gguf", ComponentRole::Projector)]
            .map(|(file, role)| ModelComponent {
                path: PathBuf::from(file),
                role,
                size_bytes: 7,
                content: ContentIdentity::Sha256 {
                    value: "0".repeat(64),
                },
                shard_index: None,
                relationship: None,
            })
            .to_vec();
        let model = managed_model(repository, "commit", components.clone());

        let garbage = garbage(&root, &[(model, components)]).expect("garbage");

        let removed = &garbage.0[&record];
        assert_eq!(
            removed.entries,
            BTreeSet::from(["model.gguf", "mmproj.gguf"].map(|file| snapshot_entry(repository, "commit", Path::new(file))))
        );
        assert_eq!(removed.blobs, BTreeSet::from([record.join("blobs").join("weights")]));
    }

    fn managed_model(repository: &str, commit: &str, components: Vec<ModelComponent>) -> InventoryModel {
        InventoryModel {
            id: InventoryEntryId("mdl_managed".to_owned()),
            content_id: ContentId("content_managed".to_owned()),
            created: 1,
            name: "managed".to_owned(),
            supported_parameters: Vec::new(),
            availability: ModelAvailability::Available { ready_at: 1 },
            source: ModelSource::HuggingFace {
                repository: repository.to_owned(),
                requested_revision: commit.to_owned(),
                commit: commit.to_owned(),
                metadata: None,
            },
            location: ModelLocation::MagnitudeCache {
                total_bytes: components.iter().map(|component| component.size_bytes).sum(),
                components,
                integrity: Integrity::Verified {
                    method: "test".to_owned(),
                },
            },
            properties: InventoryProperties::Pending,
            operations: Vec::new(),
            updated_at: 1,
        }
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
