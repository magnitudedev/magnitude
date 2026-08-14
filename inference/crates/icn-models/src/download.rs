use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;
use std::time::Instant;

use futures_util::{StreamExt, stream};
use getrandom::fill;
use hf_hub::HFError;
use icn_contracts::models::ModelPackage;
use icn_contracts::{
    ContentIdentity, DownloadEventStream, DownloadFailure, DownloadFileProgress, DownloadStage,
    Integrity, InventoryError, InventoryModel, InventoryProperties, ModelAvailability,
    ModelComponent, ModelDownloadEvent, ModelLocation, ModelSource,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sha2_state::digest::common::hazmat::{SerializableState, SerializedState};
use sha2_state::{Digest as StateDigest, Sha256 as StatefulSha256};
use tokio::io::{AsyncSeekExt, AsyncWriteExt};
use tokio::sync::watch;

use crate::hugging_face::{require_requested_revision, revision_metadata_url};
use crate::identity::{content_id, model_id};
use crate::inventory::{ManagedModelStore, build_model, hf_repo_dir, now, repository_lock_path};
use crate::store_fs::{
    acquire_exclusive_lock, ensure_owned_directory as ensure_store_directory,
    quarantine_owned_path_sync,
};
use crate::validation::ValidatedDownloadPackage;

const MAX_ATTEMPTS: usize = 5;
const INTEGRITY_CHECKPOINT_INTERVAL: u64 = 256 * 1024 * 1024;
const MAX_INTEGRITY_RECORD_BYTES: u64 = 4 * 1024;

pub(crate) struct DownloadOperation {
    sender: watch::Sender<ModelDownloadEvent>,
    cancelled: AtomicBool,
}

impl DownloadOperation {
    fn subscribe(&self) -> DownloadEventStream {
        watch_stream(self.sender.subscribe())
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    fn ensure_active(&self) -> Result<(), DownloadError> {
        if self.cancelled.load(Ordering::Acquire) {
            Err(DownloadError {
                kind: DownloadErrorKind::Cancelled,
                message: "download was cancelled".to_owned(),
                retryable: true,
                resumable: true,
            })
        } else {
            Ok(())
        }
    }
}

#[derive(Debug, serde::Deserialize)]
struct HubApiModel {
    sha: Option<String>,
    #[serde(default)]
    siblings: Vec<HubApiSibling>,
}

#[derive(Debug, serde::Deserialize)]
struct HubApiSibling {
    rfilename: String,
    size: Option<u64>,
    lfs: Option<HubApiLfs>,
}

#[derive(Debug, serde::Deserialize)]
struct HubApiLfs {
    sha256: String,
    size: u64,
}

#[derive(Debug)]
struct ResolvedRemoteMetadata {
    size: u64,
    sha256: Option<String>,
}

struct DownloadComponentPaths {
    partial: PathBuf,
    blob: PathBuf,
    checkpoint: PathBuf,
}

#[derive(Clone)]
struct DownloadIntegrity {
    digest: Option<StatefulSha256>,
    bytes: u64,
    checkpointed_bytes: u64,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DownloadIntegrityRecord {
    content: ContentIdentity,
    expected_size: u64,
    completed_bytes: u64,
    sha256_state: Option<Vec<u8>>,
}

impl DownloadComponentPaths {
    fn new(blobs: &Path, content_key: &str) -> Self {
        Self {
            blob: blobs.join(content_key),
            partial: blobs.join(format!("{content_key}.incomplete")),
            checkpoint: blobs.join(format!("{content_key}.integrity")),
        }
    }
}

impl DownloadIntegrityRecord {
    fn restore(self, content: &ContentIdentity, expected_size: u64) -> Option<DownloadIntegrity> {
        if &self.content != content
            || self.expected_size != expected_size
            || self.completed_bytes > expected_size
        {
            return None;
        }
        let digest = match (content, self.sha256_state) {
            (ContentIdentity::Sha256 { .. }, Some(bytes)) => {
                let serialized =
                    SerializedState::<StatefulSha256>::try_from(bytes.as_slice()).ok()?;
                Some(StatefulSha256::deserialize(&serialized).ok()?)
            }
            (ContentIdentity::Sha256 { .. }, None) | (_, Some(_)) => return None,
            (_, None) => None,
        };
        Some(DownloadIntegrity {
            digest,
            bytes: self.completed_bytes,
            checkpointed_bytes: self.completed_bytes,
        })
    }
}

impl DownloadIntegrity {
    fn empty(component: &ModelComponent) -> Self {
        Self {
            digest: matches!(&component.content, ContentIdentity::Sha256 { .. })
                .then(StatefulSha256::new),
            bytes: 0,
            checkpointed_bytes: 0,
        }
    }

    fn restore(
        component: &ModelComponent,
        record: DownloadIntegrityRecord,
    ) -> Result<Self, DownloadError> {
        record
            .restore(&component.content, component.size_bytes)
            .ok_or_else(|| invalid_checkpoint(component))
    }

    fn update(&mut self, bytes: &[u8]) -> Result<(), DownloadError> {
        let count = u64::try_from(bytes.len()).map_err(|_| DownloadError {
            kind: DownloadErrorKind::Integrity,
            message: "download chunk size overflows u64".to_owned(),
            retryable: false,
            resumable: false,
        })?;
        self.bytes = self.bytes.checked_add(count).ok_or_else(|| DownloadError {
            kind: DownloadErrorKind::Integrity,
            message: "download byte count overflows u64".to_owned(),
            retryable: false,
            resumable: false,
        })?;
        if let Some(digest) = self.digest.as_mut() {
            digest.update(bytes);
        }
        Ok(())
    }

    fn record(&self, component: &ModelComponent) -> DownloadIntegrityRecord {
        DownloadIntegrityRecord {
            content: component.content.clone(),
            expected_size: component.size_bytes,
            completed_bytes: self.bytes,
            sha256_state: self
                .digest
                .as_ref()
                .map(|digest| digest.serialize().as_slice().to_vec()),
        }
    }

    fn needs_checkpoint(&self) -> bool {
        self.bytes.saturating_sub(self.checkpointed_bytes) >= INTEGRITY_CHECKPOINT_INTERVAL
    }

    fn mark_checkpointed(&mut self) {
        self.checkpointed_bytes = self.bytes;
    }

    fn verify(&self, component: &ModelComponent) -> Result<(), DownloadError> {
        if self.bytes != component.size_bytes {
            return Err(DownloadError {
                kind: DownloadErrorKind::Integrity,
                message: format!("unexpected size for {}", component.path.display()),
                retryable: true,
                resumable: true,
            });
        }
        let ContentIdentity::Sha256 { value: expected } = &component.content else {
            return Ok(());
        };
        let actual = self
            .digest
            .clone()
            .expect("SHA-256 content has an integrity digest")
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        if &actual != expected {
            return Err(DownloadError {
                kind: DownloadErrorKind::Integrity,
                message: format!("SHA-256 mismatch for {}", component.path.display()),
                retryable: false,
                resumable: false,
            });
        }
        Ok(())
    }
}

#[derive(Debug)]
enum DownloadErrorKind {
    Cancelled,
    InsufficientDiskSpace {
        required_bytes: u64,
        available_bytes: u64,
    },
    SourceUnavailable,
    SourceAccessDenied,
    MissingSource,
    Network,
    Integrity,
    FileSystem,
    InvalidRequest,
    Internal,
}

#[derive(Debug, thiserror::Error)]
#[error("{message}")]
struct DownloadError {
    kind: DownloadErrorKind,
    message: String,
    retryable: bool,
    resumable: bool,
}

impl DownloadError {
    fn resumable(&self) -> bool {
        self.resumable
    }

    fn retryable(&self) -> bool {
        self.retryable
    }

    fn to_failure(&self) -> Option<DownloadFailure> {
        Some(match &self.kind {
            DownloadErrorKind::Cancelled => return None,
            DownloadErrorKind::InsufficientDiskSpace {
                required_bytes,
                available_bytes,
            } => DownloadFailure::InsufficientDiskSpace {
                required_bytes: *required_bytes,
                available_bytes: *available_bytes,
            },
            DownloadErrorKind::SourceUnavailable
            | DownloadErrorKind::SourceAccessDenied
            | DownloadErrorKind::MissingSource => DownloadFailure::SourceUnavailable,
            DownloadErrorKind::Network => DownloadFailure::NetworkUnavailable,
            DownloadErrorKind::Integrity => DownloadFailure::CorruptDownload,
            DownloadErrorKind::FileSystem => DownloadFailure::LocalStorageFailure,
            DownloadErrorKind::InvalidRequest => DownloadFailure::Internal {
                message: self.message.clone(),
            },
            DownloadErrorKind::Internal => DownloadFailure::Internal {
                message: self.message.clone(),
            },
        })
    }
}

impl ManagedModelStore {
    pub(crate) async fn start_target_downloads(
        &self,
        packages: Vec<ModelPackage>,
    ) -> Result<Vec<DownloadEventStream>, InventoryError> {
        // Admission is a mutation boundary. Re-observe the store before deciding that any exact
        // package is already present; the query snapshot is deliberately not mutation authority.
        self.ensure_installed_model_inventory().await?;
        let packages = packages
            .into_iter()
            .map(|package| {
                let package_id = package.id.clone();
                let key = package_download_key(&package);
                ValidatedDownloadPackage::new(package).map(|package| (package_id, key, package))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut resolved_packages = Vec::with_capacity(packages.len());
        for (package_id, key, package) in packages {
            let installed = match self.installed_package(&package_id).await {
                Ok((_package, resolved)) => Some(resolved.model),
                Err(InventoryError::NotFound(_)) | Err(InventoryError::NotReady(_)) => None,
                Err(error) => return Err(error),
            };
            resolved_packages.push((key, package, installed));
        }
        let mut operations = self.operations.lock().await;
        let mut streams = Vec::with_capacity(resolved_packages.len());
        let mut admitted = Vec::new();
        for (key, package, installed) in resolved_packages {
            if let Some(operation) = operations.get(&key) {
                streams.push(operation.subscribe());
                continue;
            }
            let operation_id = random_id("download")?;
            if let Some(model) = installed {
                let (_sender, receiver) = watch::channel(ModelDownloadEvent::Ready {
                    operation_id,
                    model: Box::new(model),
                });
                streams.push(watch_stream(receiver));
                continue;
            }
            let (repository, revision) = package.repository_revision();
            let initial = ModelDownloadEvent::Resolving {
                operation_id: operation_id.clone(),
                repository: repository.to_owned(),
                revision: revision.to_owned(),
            };
            let (sender, _receiver) = watch::channel(initial);
            let operation = Arc::new(DownloadOperation {
                sender,
                cancelled: AtomicBool::new(false),
            });
            streams.push(operation.subscribe());
            admitted.push((key.clone(), operation_id, package, Arc::clone(&operation)));
            operations.insert(key, operation);
        }
        drop(operations);
        for (key, operation_id, package, operation) in admitted {
            let manager = self.clone();
            tokio::spawn(async move {
                manager
                    .run_download(key, operation_id, package, operation)
                    .await;
            });
        }
        Ok(streams)
    }

    pub(crate) async fn cancel_package_download(
        &self,
        package: &ModelPackage,
    ) -> Result<(), InventoryError> {
        ValidatedDownloadPackage::new(package.clone())?;
        let key = package_download_key(package);
        let operation = self
            .operations
            .lock()
            .await
            .get(&key)
            .cloned()
            .ok_or_else(|| InventoryError::NotFound("active download".to_owned()))?;
        operation.cancel();
        Ok(())
    }

    async fn run_download(
        &self,
        operation_key: String,
        operation_id: String,
        package: ValidatedDownloadPackage,
        operation: Arc<DownloadOperation>,
    ) {
        let result = self
            .run_download_inner(&operation_id, package, &operation)
            .await;
        if let Err(failure) = result {
            let model_id = current_model_id(&operation.sender.borrow());
            let (completed_bytes, total_bytes) = progress_totals(&operation.sender.borrow());
            let Some(download_failure) = failure.to_failure() else {
                if let Some(model_id) = model_id.as_ref() {
                    if let Ok(mut models) = self.models.write() {
                        models.remove(model_id);
                    }
                }
                operation
                    .sender
                    .send_replace(ModelDownloadEvent::Cancelled {
                        operation_id,
                        model_id,
                        completed_bytes,
                        total_bytes,
                    });
                self.operations.lock().await.remove(&operation_key);
                return;
            };
            let resumable = failure.resumable();
            if let Some(model_id) = model_id.as_ref()
                && let Ok(mut models) = self.models.write()
                && let Some(model) = models.get_mut(model_id)
            {
                let (completed_bytes, total_bytes) = progress_totals(&operation.sender.borrow());
                model.availability = ModelAvailability::Interrupted {
                    completed_bytes,
                    total_bytes,
                    resumable,
                    failure: download_failure.clone(),
                    updated_at: now(),
                };
                model.updated_at = now();
            }
            operation.sender.send_replace(ModelDownloadEvent::Failed {
                operation_id,
                model_id,
                error: download_failure,
                completed_bytes,
                total_bytes,
                resumable,
            });
        }
        self.operations.lock().await.remove(&operation_key);
    }

    async fn run_download_inner(
        &self,
        operation_id: &str,
        package: ValidatedDownloadPackage,
        operation: &DownloadOperation,
    ) -> Result<(), DownloadError> {
        operation.ensure_active()?;
        let (repository, revision) = package.repository_revision();
        let (owner, name) = repository.split_once('/').ok_or_else(|| DownloadError {
            kind: DownloadErrorKind::InvalidRequest,
            message: "repository must be owner/name".to_owned(),
            retryable: false,
            resumable: false,
        })?;
        let repo = self.client.model(owner.to_owned(), name.to_owned());
        let repository_lock =
            acquire_lock(repository_lock_path(&self.config.root, repository)).await?;

        let pinned = resolve_download_revision(
            &self.client,
            &repo,
            repository,
            revision,
            package.components(),
            None,
        )
        .await;
        let resolved = match pinned {
            Ok(resolved) => resolved,
            Err(error) if missing_upstream_content(&error) => {
                let resolved = resolve_download_revision(
                    &self.client,
                    &repo,
                    repository,
                    "main",
                    package.components(),
                    Some(revision),
                )
                .await?;
                tracing::info!(
                    repository,
                    pinned_revision = revision,
                    resolved_revision = resolved,
                    "using content-equivalent current revision for model acquisition"
                );
                resolved
            }
            Err(error) => return Err(error),
        };
        let commit = resolved;
        let (repository, revision, components) = package.into_parts();
        let content_id = content_id(&components);
        let repo_root = self.config.root.join("hub").join(hf_repo_dir(&repository));
        let snapshot = repo_root.join("snapshots").join(&commit);
        let model_id = model_id("magnitude-cache", &snapshot, &content_id);
        let total_bytes: u64 = components
            .iter()
            .map(|component| component.size_bytes)
            .sum();
        let existing = self.models.read().ok().and_then(|models| {
            models
                .get(&model_id)
                .filter(|model| matches!(model.availability, ModelAvailability::Available { .. }))
                .cloned()
        });
        if let Some(model) = existing {
            operation.sender.send_replace(ModelDownloadEvent::Ready {
                operation_id: operation_id.to_owned(),
                model: Box::new(model),
            });
            return Ok(());
        }

        let completed_bytes = resumable_bytes(&repo_root, &components).await;
        let missing_bytes = total_bytes.saturating_sub(completed_bytes);
        let available_bytes =
            fs2::available_space(&self.config.root).map_err(|error| DownloadError {
                kind: DownloadErrorKind::FileSystem,
                message: error.to_string(),
                retryable: true,
                resumable: true,
            })?;
        operation
            .sender
            .send_replace(ModelDownloadEvent::CheckingSpace {
                operation_id: operation_id.to_owned(),
                model_id: model_id.clone(),
                required_bytes: missing_bytes.saturating_add(self.config.disk_reserve_bytes),
                available_bytes,
                completed_bytes,
                total_bytes,
            });
        let required_bytes = missing_bytes.saturating_add(self.config.disk_reserve_bytes);
        if required_bytes > available_bytes {
            return Err(DownloadError {
                kind: DownloadErrorKind::InsufficientDiskSpace {
                    required_bytes,
                    available_bytes,
                },
                message: "insufficient disk space".to_owned(),
                retryable: false,
                resumable: true,
            });
        }

        let started_at = now();
        let planned = InventoryModel {
            id: model_id.clone(),
            content_id: content_id.clone(),
            created: started_at,
            name: repository.clone(),
            supported_parameters: Vec::new(),
            availability: ModelAvailability::Downloading {
                operation_id: operation_id.to_owned(),
                stage: DownloadStage::Queued,
                completed_bytes,
                total_bytes,
                current_component: None,
                started_at,
                updated_at: started_at,
            },
            source: ModelSource::HuggingFace {
                repository: repository.clone(),
                requested_revision: revision.clone(),
                commit: commit.clone(),
                metadata: None,
            },
            location: ModelLocation::MagnitudeCache {
                components: components.clone(),
                total_bytes,
                integrity: Integrity::Unverified {
                    reason: "download_in_progress".to_owned(),
                },
            },
            properties: InventoryProperties::Pending,
            operations: Vec::new(),
            updated_at: started_at,
        };
        self.models
            .write()
            .map_err(|_| DownloadError {
                kind: DownloadErrorKind::Internal,
                message: "inventory lock poisoned".to_owned(),
                retryable: false,
                resumable: true,
            })?
            .insert(model_id.clone(), planned);

        if let Some(first) = components.first() {
            operation.sender.send_replace(ModelDownloadEvent::Progress {
                operation_id: operation_id.to_owned(),
                model_id: model_id.clone(),
                stage: DownloadStage::Queued,
                completed_bytes,
                total_bytes,
                file: DownloadFileProgress {
                    path: first.path.clone(),
                    completed_bytes: component_partial_len(&repo_root, first).await,
                    total_bytes: first.size_bytes,
                },
                bytes_per_second: None,
                resumed_from_bytes: completed_bytes,
            });
        }

        let _slot = self
            .download_slots
            .acquire()
            .await
            .map_err(|error| DownloadError {
                kind: DownloadErrorKind::Internal,
                message: error.to_string(),
                retryable: false,
                resumable: true,
            })?;
        ensure_owned_directory(&repo_root).await?;
        ensure_owned_directory(&repo_root.join("blobs")).await?;
        let started = Instant::now();
        let mut resumed_by_component = Vec::with_capacity(components.len());
        for component in &components {
            resumed_by_component.push(component_partial_len(&repo_root, component).await);
        }
        for (index, component) in components.iter().enumerate() {
            operation.ensure_active()?;
            let resumed_from = resumed_by_component[index];
            let mut last_progress_emit = Instant::now()
                .checked_sub(Duration::from_millis(100))
                .unwrap_or_else(Instant::now);
            download_component_with_retry(
                &repo,
                &self.config.root,
                &commit,
                component,
                |file_completed, stage| {
                    let timestamp = Instant::now();
                    if file_completed != component.size_bytes
                        && timestamp.duration_since(last_progress_emit) < Duration::from_millis(100)
                    {
                        return;
                    }
                    last_progress_emit = timestamp;
                    let previous_files = components[..index]
                        .iter()
                        .map(|item| item.size_bytes)
                        .sum::<u64>();
                    let future_resumed = resumed_by_component[index + 1..].iter().sum::<u64>();
                    let completed = previous_files
                        .saturating_add(file_completed)
                        .saturating_add(future_resumed);
                    let elapsed = started.elapsed().as_secs_f64();
                    let previous_transferred = components[..index]
                        .iter()
                        .zip(&resumed_by_component[..index])
                        .map(|(item, resumed)| item.size_bytes.saturating_sub(*resumed))
                        .sum::<u64>();
                    let transferred = previous_transferred
                        .saturating_add(file_completed.saturating_sub(resumed_from));
                    let rate =
                        (elapsed > 0.0 && transferred > 0).then(|| transferred as f64 / elapsed);
                    operation.sender.send_replace(ModelDownloadEvent::Progress {
                        operation_id: operation_id.to_owned(),
                        model_id: model_id.clone(),
                        stage,
                        completed_bytes: completed,
                        total_bytes,
                        file: DownloadFileProgress {
                            path: component.path.clone(),
                            completed_bytes: file_completed,
                            total_bytes: component.size_bytes,
                        },
                        bytes_per_second: rate,
                        resumed_from_bytes: resumed_from,
                    });
                    if let Ok(mut models) = self.models.write()
                        && let Some(model) = models.get_mut(&model_id)
                    {
                        let updated_at = now();
                        model.availability = ModelAvailability::Downloading {
                            operation_id: operation_id.to_owned(),
                            stage,
                            completed_bytes: completed,
                            total_bytes,
                            current_component: Some(component.path.clone()),
                            started_at,
                            updated_at,
                        };
                        model.updated_at = updated_at;
                    }
                },
                &operation.cancelled,
            )
            .await?;
        }

        if let Some(last) = components.last() {
            operation.sender.send_replace(ModelDownloadEvent::Progress {
                operation_id: operation_id.to_owned(),
                model_id: model_id.clone(),
                stage: DownloadStage::Verifying,
                completed_bytes: total_bytes,
                total_bytes,
                file: DownloadFileProgress {
                    path: last.path.clone(),
                    completed_bytes: last.size_bytes,
                    total_bytes: last.size_bytes,
                },
                bytes_per_second: None,
                resumed_from_bytes: 0,
            });
        }

        operation.ensure_active()?;
        publish_package_snapshot(&repo_root, &snapshot, &commit, &components).await?;
        let ready_at = now();
        let primary = components
            .iter()
            .filter(|component| {
                matches!(
                    component.role,
                    icn_contracts::ComponentRole::Weights | icn_contracts::ComponentRole::Shard
                )
            })
            .min_by_key(|component| component.shard_index.unwrap_or(0))
            .map(|component| snapshot.join(&component.path))
            .ok_or_else(|| DownloadError {
                kind: DownloadErrorKind::Internal,
                message: "published model has no runnable weight component".to_owned(),
                retryable: false,
                resumable: false,
            })?;
        let model = build_model(
            model_id.clone(),
            content_id,
            started_at,
            ready_at,
            ModelSource::HuggingFace {
                repository: repository.clone(),
                requested_revision: revision.clone(),
                commit,
                metadata: None,
            },
            ModelLocation::MagnitudeCache {
                total_bytes,
                components,
                integrity: Integrity::Verified {
                    method: "content_identity".to_owned(),
                },
            },
            &primary,
            true,
            &self.cache,
            self.template_assessor.as_deref(),
        )
        .map_err(|error| DownloadError {
            kind: DownloadErrorKind::Internal,
            message: error.to_string(),
            retryable: true,
            resumable: true,
        })?;
        let ready = self
            .complete_and_publish_model(model)
            .await
            .map_err(|error| DownloadError {
                kind: DownloadErrorKind::Internal,
                message: error.to_string(),
                retryable: true,
                resumable: true,
            })?;
        operation.sender.send_replace(ModelDownloadEvent::Ready {
            operation_id: operation_id.to_owned(),
            model: Box::new(ready),
        });
        drop(repository_lock);
        Ok(())
    }
}

async fn publish_package_snapshot(
    repo_root: &Path,
    snapshot: &Path,
    commit: &str,
    components: &[ModelComponent],
) -> Result<(), DownloadError> {
    let incomplete = repo_root.join(".incomplete");
    let snapshots = repo_root.join("snapshots");
    ensure_owned_directory(&incomplete).await?;
    ensure_owned_directory(&snapshots).await?;

    match tokio::fs::symlink_metadata(snapshot).await {
        Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => {
            for component in components {
                publish_snapshot_link(repo_root, snapshot, component).await?;
            }
            return sync_directory(snapshot).await;
        }
        Ok(_) => quarantine(snapshot).await?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(download_io(error)),
    }

    let staged_snapshot = incomplete.join(format!("snapshot-{commit}"));
    quarantine(&staged_snapshot).await?;
    ensure_owned_directory(&staged_snapshot).await?;
    for component in components {
        publish_snapshot_link(repo_root, &staged_snapshot, component).await?;
    }
    tokio::fs::rename(staged_snapshot, snapshot)
        .await
        .map_err(download_io)?;
    sync_parent(snapshot).await
}

async fn ensure_owned_directory(path: &Path) -> Result<(), DownloadError> {
    ensure_store_directory(path)
        .await
        .map_err(inventory_download_error)
}

async fn download_component_with_retry(
    repo: &hf_hub::HFRepository<hf_hub::RepoTypeModel>,
    root: &Path,
    commit: &str,
    component: &ModelComponent,
    mut progress: impl FnMut(u64, DownloadStage),
    cancelled: &AtomicBool,
) -> Result<(), DownloadError> {
    let blobs = root
        .join("hub")
        .join(hf_repo_dir(&repo.repo_path()))
        .join("blobs");
    let paths = DownloadComponentPaths::new(&blobs, &blob_key(&component.content));
    if recover_completed_blob(&paths, component).await? {
        progress(component.size_bytes, DownloadStage::Verifying);
        return Ok(());
    }
    let mut integrity = recover_partial(&paths, component).await?;
    progress(integrity.bytes, DownloadStage::Downloading);

    for attempt in 0..MAX_ATTEMPTS {
        if cancelled.load(Ordering::Acquire) {
            persist_integrity_checkpoint(&paths, component, &mut integrity, None).await?;
            return Err(cancelled_error());
        }
        match download_component_once(
            repo,
            commit,
            component,
            &paths,
            &mut integrity,
            &mut progress,
            cancelled,
        )
        .await
        {
            Ok(()) => return Ok(()),
            Err(error) if error.retryable() && attempt + 1 < MAX_ATTEMPTS => {
                tokio::time::sleep(std::time::Duration::from_secs(1_u64 << attempt.min(4))).await;
            }
            Err(error) => return Err(error),
        }
    }
    unreachable!("bounded retry loop returns on its final attempt")
}

async fn download_component_once(
    repo: &hf_hub::HFRepository<hf_hub::RepoTypeModel>,
    commit: &str,
    component: &ModelComponent,
    paths: &DownloadComponentPaths,
    integrity: &mut DownloadIntegrity,
    progress: &mut impl FnMut(u64, DownloadStage),
    cancelled: &AtomicBool,
) -> Result<(), DownloadError> {
    let mut offset = integrity.bytes;
    let partial_len = tokio::fs::metadata(&paths.partial)
        .await
        .map(|metadata| metadata.len())
        .unwrap_or(0);
    if partial_len > offset {
        tokio::fs::OpenOptions::new()
            .write(true)
            .open(&paths.partial)
            .await
            .map_err(download_io)?
            .set_len(offset)
            .await
            .map_err(download_io)?;
    } else if partial_len < offset {
        return Err(DownloadError {
            kind: DownloadErrorKind::Integrity,
            message: format!(
                "partial download is shorter than verified progress for {}",
                component.path.display()
            ),
            retryable: true,
            resumable: false,
        });
    }
    if offset == component.size_bytes {
        progress(component.size_bytes, DownloadStage::Verifying);
        if let Err(error) = integrity.verify(component) {
            discard_component_files(paths).await?;
            return Err(error);
        }
        publish_verified_blob(paths).await?;
        return Ok(());
    }

    let (_reported_length, mut stream) = repo
        .download_file_stream()
        .filename(component.path.to_string_lossy().into_owned())
        .revision(commit.to_owned())
        .range(offset..component.size_bytes)
        .send()
        .await
        .map_err(map_hf_error)?;
    let std_file = open_partial(&paths.partial)?;
    let mut file = tokio::fs::File::from_std(std_file);
    file.seek(std::io::SeekFrom::Start(offset))
        .await
        .map_err(download_io)?;
    while let Some(chunk) = stream.next().await {
        if cancelled.load(Ordering::Acquire) {
            persist_integrity_checkpoint(paths, component, integrity, Some(&mut file)).await?;
            return Err(cancelled_error());
        }
        let chunk = match chunk {
            Ok(chunk) => chunk,
            Err(error) => {
                persist_integrity_checkpoint(paths, component, integrity, Some(&mut file)).await?;
                return Err(map_hf_error(error));
            }
        };
        let chunk_len = u64::try_from(chunk.len()).map_err(|_| DownloadError {
            kind: DownloadErrorKind::Integrity,
            message: "download chunk size overflows u64".to_owned(),
            retryable: false,
            resumable: false,
        })?;
        if offset
            .checked_add(chunk_len)
            .is_none_or(|next| next > component.size_bytes)
        {
            return Err(DownloadError {
                kind: DownloadErrorKind::Integrity,
                message: format!(
                    "download exceeded expected size for {}",
                    component.path.display()
                ),
                retryable: true,
                resumable: true,
            });
        }
        file.write_all(&chunk).await.map_err(download_io)?;
        integrity.update(&chunk)?;
        offset += chunk_len;
        progress(offset, DownloadStage::Downloading);
        if integrity.needs_checkpoint() {
            persist_integrity_checkpoint(paths, component, integrity, Some(&mut file)).await?;
        }
    }
    if offset != component.size_bytes {
        persist_integrity_checkpoint(paths, component, integrity, Some(&mut file)).await?;
        return Err(DownloadError {
            kind: DownloadErrorKind::Integrity,
            message: format!(
                "download ended at {offset} bytes; expected {} for {}",
                component.size_bytes,
                component.path.display()
            ),
            retryable: true,
            resumable: true,
        });
    }
    persist_integrity_checkpoint(paths, component, integrity, Some(&mut file)).await?;
    drop(file);
    progress(component.size_bytes, DownloadStage::Verifying);
    if let Err(error) = integrity.verify(component) {
        discard_component_files(paths).await?;
        return Err(error);
    }
    publish_verified_blob(paths).await?;
    Ok(())
}

async fn recover_partial(
    paths: &DownloadComponentPaths,
    component: &ModelComponent,
) -> Result<DownloadIntegrity, DownloadError> {
    let partial_len = regular_file_len(&paths.partial).await;
    let record = read_integrity_record(&paths.checkpoint).await;
    let Some((partial_len, record)) = partial_len.zip(record) else {
        if partial_len.is_some() || paths.checkpoint.exists() {
            discard_component_files(paths).await?;
        }
        return Ok(DownloadIntegrity::empty(component));
    };
    let integrity = match DownloadIntegrity::restore(component, record) {
        Ok(integrity) if partial_len >= integrity.bytes => integrity,
        _ => {
            discard_component_files(paths).await?;
            return Ok(DownloadIntegrity::empty(component));
        }
    };
    if partial_len > integrity.bytes {
        tokio::fs::OpenOptions::new()
            .write(true)
            .open(&paths.partial)
            .await
            .map_err(download_io)?
            .set_len(integrity.bytes)
            .await
            .map_err(download_io)?;
    }
    Ok(integrity)
}

async fn recover_completed_blob(
    paths: &DownloadComponentPaths,
    component: &ModelComponent,
) -> Result<bool, DownloadError> {
    let Some(blob_len) = regular_file_len(&paths.blob).await else {
        return Ok(false);
    };
    let checkpoint = read_integrity_record(&paths.checkpoint)
        .await
        .and_then(|record| DownloadIntegrity::restore(component, record).ok());
    if blob_len == component.size_bytes
        && let Some(integrity) = checkpoint
        && integrity.bytes == component.size_bytes
        && integrity.verify(component).is_ok()
    {
        return Ok(true);
    }

    discard_component_files(paths).await?;
    Ok(false)
}

async fn persist_integrity_checkpoint(
    paths: &DownloadComponentPaths,
    component: &ModelComponent,
    integrity: &mut DownloadIntegrity,
    file: Option<&mut tokio::fs::File>,
) -> Result<(), DownloadError> {
    if integrity.bytes == integrity.checkpointed_bytes {
        return Ok(());
    }
    if let Some(file) = file {
        file.flush().await.map_err(download_io)?;
        file.sync_data().await.map_err(download_io)?;
    } else {
        tokio::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&paths.partial)
            .await
            .map_err(download_io)?
            .sync_data()
            .await
            .map_err(download_io)?;
    }
    atomic_json(&paths.checkpoint, &integrity.record(component)).await?;
    integrity.mark_checkpointed();
    Ok(())
}

async fn publish_verified_blob(paths: &DownloadComponentPaths) -> Result<(), DownloadError> {
    tokio::fs::rename(&paths.partial, &paths.blob)
        .await
        .map_err(download_io)?;
    sync_parent(&paths.blob).await
}

async fn read_integrity_record(path: &Path) -> Option<DownloadIntegrityRecord> {
    read_bounded_json(path).await
}

async fn read_bounded_json<T: for<'de> Deserialize<'de>>(path: &Path) -> Option<T> {
    let metadata = tokio::fs::symlink_metadata(path).await.ok()?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return None;
    }
    if metadata.len() > MAX_INTEGRITY_RECORD_BYTES {
        return None;
    }
    let bytes = tokio::fs::read(path).await.ok()?;
    serde_json::from_slice(&bytes).ok()
}

async fn regular_file_len(path: &Path) -> Option<u64> {
    let metadata = tokio::fs::symlink_metadata(path).await.ok()?;
    (metadata.is_file() && !metadata.file_type().is_symlink()).then_some(metadata.len())
}

async fn discard_component_files(paths: &DownloadComponentPaths) -> Result<(), DownloadError> {
    for path in [&paths.partial, &paths.blob, &paths.checkpoint] {
        match tokio::fs::symlink_metadata(path).await {
            Ok(metadata) if metadata.is_file() && !metadata.file_type().is_symlink() => {
                tokio::fs::remove_file(path).await.map_err(download_io)?;
            }
            // The store only ever writes regular files at these paths; anything
            // else is foreign and is preserved by rename instead of deleted.
            Ok(_) => quarantine(path).await?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(download_io(error)),
        }
    }
    Ok(())
}

fn invalid_checkpoint(component: &ModelComponent) -> DownloadError {
    DownloadError {
        kind: DownloadErrorKind::Integrity,
        message: format!(
            "download integrity checkpoint does not match {}",
            component.path.display()
        ),
        retryable: true,
        resumable: false,
    }
}

fn cancelled_error() -> DownloadError {
    DownloadError {
        kind: DownloadErrorKind::Cancelled,
        message: "download was cancelled".to_owned(),
        retryable: false,
        resumable: true,
    }
}

async fn publish_snapshot_link(
    repo_root: &Path,
    snapshot: &Path,
    component: &ModelComponent,
) -> Result<(), DownloadError> {
    let destination = snapshot.join(&component.path);
    if let Some(parent) = destination.parent() {
        let relative = parent.strip_prefix(snapshot).map_err(download_io)?;
        let mut current = snapshot.to_path_buf();
        for component in relative.components() {
            current.push(component.as_os_str());
            ensure_owned_directory(&current).await?;
        }
    }
    let blob = repo_root.join("blobs").join(blob_key(&component.content));
    let blob_metadata = tokio::fs::symlink_metadata(&blob)
        .await
        .map_err(download_io)?;
    if !blob_metadata.is_file() || blob_metadata.file_type().is_symlink() {
        return Err(DownloadError {
            kind: DownloadErrorKind::FileSystem,
            message: format!("verified blob is not a regular file: {}", blob.display()),
            retryable: true,
            resumable: true,
        });
    }
    let destination_clone = destination.clone();
    tokio::task::spawn_blocking(move || -> Result<(), DownloadError> {
        match destination_clone.symlink_metadata() {
            Ok(_) => {
                let matching = destination_clone
                    .canonicalize()
                    .ok()
                    .zip(blob.canonicalize().ok())
                    .is_some_and(|(existing, expected)| existing == expected);
                if matching {
                    return Ok(());
                }
                quarantine_owned_path_sync(&destination_clone).map_err(inventory_download_error)?;
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(download_io(error)),
        }
        #[cfg(unix)]
        {
            let relative = pathdiff(&blob, destination_clone.parent().unwrap_or(Path::new(".")));
            std::os::unix::fs::symlink(relative, &destination_clone).map_err(download_io)?;
        }
        #[cfg(not(unix))]
        {
            fs::hard_link(&blob, &destination_clone).map_err(download_io)?;
        }
        Ok(())
    })
    .await
    .map_err(|error| DownloadError {
        kind: DownloadErrorKind::FileSystem,
        message: error.to_string(),
        retryable: true,
        resumable: true,
    })??;
    Ok(())
}

fn pathdiff(path: &Path, base: &Path) -> PathBuf {
    let path_components = path.components().collect::<Vec<_>>();
    let base_components = base.components().collect::<Vec<_>>();
    let shared = path_components
        .iter()
        .zip(&base_components)
        .take_while(|(left, right)| left == right)
        .count();
    let mut result = PathBuf::new();
    for _ in shared..base_components.len() {
        result.push("..");
    }
    for component in &path_components[shared..] {
        result.push(component.as_os_str());
    }
    result
}

async fn atomic_json(path: &Path, value: &impl serde::Serialize) -> Result<(), DownloadError> {
    let bytes = serde_json::to_vec_pretty(value).map_err(|error| DownloadError {
        kind: DownloadErrorKind::Internal,
        message: error.to_string(),
        retryable: false,
        resumable: true,
    })?;
    let temporary = path.with_extension(format!(
        "tmp-{}",
        random_id("write").map_err(inventory_download_error)?
    ));
    let mut file = tokio::fs::File::create(&temporary)
        .await
        .map_err(download_io)?;
    file.write_all(&bytes).await.map_err(download_io)?;
    file.flush().await.map_err(download_io)?;
    file.sync_all().await.map_err(download_io)?;
    drop(file);
    tokio::fs::rename(&temporary, path)
        .await
        .map_err(download_io)?;
    // Persist the directory entry as well as the manifest contents. Without
    // this fsync a power loss can lose the rename even though the file itself
    // was synced successfully.
    sync_parent(path).await
}

async fn sync_parent(path: &Path) -> Result<(), DownloadError> {
    let Some(parent) = path.parent() else {
        return Ok(());
    };
    let directory = tokio::fs::File::open(parent).await.map_err(download_io)?;
    directory.sync_all().await.map_err(download_io)
}

async fn sync_directory(path: &Path) -> Result<(), DownloadError> {
    let directory = tokio::fs::File::open(path).await.map_err(download_io)?;
    directory.sync_all().await.map_err(download_io)
}

async fn resumable_bytes(repo_root: &Path, components: &[ModelComponent]) -> u64 {
    let mut total = 0_u64;
    for component in components {
        total = total.saturating_add(component_partial_len(repo_root, component).await);
    }
    total
}

async fn component_partial_len(repo_root: &Path, component: &ModelComponent) -> u64 {
    let paths =
        DownloadComponentPaths::new(&repo_root.join("blobs"), &blob_key(&component.content));
    recoverable_download_bytes(&paths, &component.content, component.size_bytes).await
}

async fn recoverable_download_bytes(
    paths: &DownloadComponentPaths,
    content: &ContentIdentity,
    expected_size: u64,
) -> u64 {
    if regular_file_len(&paths.blob).await == Some(expected_size)
        && read_integrity_record(&paths.checkpoint)
            .await
            .and_then(|record| record.restore(content, expected_size))
            .is_some_and(|integrity| integrity.bytes == expected_size)
    {
        return expected_size;
    }
    let Some(partial_len) = regular_file_len(&paths.partial).await else {
        return 0;
    };
    read_integrity_record(&paths.checkpoint)
        .await
        .and_then(|record| record.restore(content, expected_size))
        .map(|integrity| integrity.bytes)
        .filter(|completed| *completed <= partial_len)
        .unwrap_or(0)
}

async fn quarantine(path: &Path) -> Result<(), DownloadError> {
    match tokio::fs::symlink_metadata(path).await {
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(download_io(error)),
    }
    let destination = path.with_extension(format!(
        "invalid-{}",
        random_id("partial").map_err(inventory_download_error)?
    ));
    tokio::fs::rename(path, destination)
        .await
        .map_err(download_io)
}

async fn acquire_lock(path: PathBuf) -> Result<File, DownloadError> {
    tokio::task::spawn_blocking(move || {
        acquire_exclusive_lock(&path).map_err(inventory_download_error)
    })
    .await
    .map_err(|error| DownloadError {
        kind: DownloadErrorKind::FileSystem,
        message: error.to_string(),
        retryable: true,
        resumable: true,
    })?
}

fn open_partial(path: &Path) -> Result<File, DownloadError> {
    if path
        .symlink_metadata()
        .is_ok_and(|metadata| metadata.file_type().is_symlink())
    {
        return Err(DownloadError {
            kind: DownloadErrorKind::FileSystem,
            message: format!("partial path is a symlink: {}", path.display()),
            retryable: false,
            resumable: false,
        });
    }
    let mut options = OpenOptions::new();
    options.create(true).read(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW);
        options.mode(0o600);
    }
    options.open(path).map_err(download_io)
}

fn package_download_key(package: &ModelPackage) -> String {
    let bytes = serde_json::to_vec(package).expect("validated model packages serialize");
    format!("{:x}", Sha256::digest(bytes))
}

pub(crate) fn blob_key(content: &ContentIdentity) -> String {
    match content {
        ContentIdentity::Sha256 { value } => format!("lfs-sha256-{value}"),
        ContentIdentity::Xet { value } => format!("xet-{value}"),
        ContentIdentity::GitOid { value } => format!("git-oid-{value}"),
        ContentIdentity::FileIdentity { value } => format!("file-{value}"),
        ContentIdentity::Unknown => "unknown".to_owned(),
    }
}

fn random_id(prefix: &str) -> Result<String, InventoryError> {
    let mut bytes = [0_u8; 16];
    fill(&mut bytes).map_err(|error| InventoryError::Internal(error.to_string()))?;
    Ok(format!(
        "{prefix}_{}",
        bytes
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    ))
}

fn inventory_download_error(error: InventoryError) -> DownloadError {
    DownloadError {
        kind: DownloadErrorKind::Internal,
        message: error.to_string(),
        retryable: false,
        resumable: true,
    }
}

fn missing_upstream_content(error: &DownloadError) -> bool {
    matches!(error.kind, DownloadErrorKind::MissingSource)
}

async fn resolve_download_revision(
    client: &hf_hub::HFClient,
    repo: &hf_hub::HFRepository<hf_hub::RepoTypeModel>,
    repository: &str,
    revision: &str,
    components: &[ModelComponent],
    equivalent_to_revision: Option<&str>,
) -> Result<String, DownloadError> {
    let api = match hub_api_metadata(client, repository, revision).await {
        Ok(api) => api,
        Err(error) if equivalent_to_revision.is_some() && missing_upstream_content(&error) => {
            let pinned = equivalent_to_revision.expect("checked equivalent package");
            return Err(package_unavailable(
                repository,
                pinned,
                None,
                None,
                "current main is unavailable",
            ));
        }
        Err(error) => return Err(error),
    };
    let commit = api.sha.clone().ok_or_else(|| DownloadError {
        kind: DownloadErrorKind::Network,
        message: "Hugging Face repository response did not include a commit".to_owned(),
        retryable: true,
        resumable: false,
    })?;
    if revision == "main" {
        if !is_immutable_commit(&commit) {
            return Err(DownloadError {
                kind: DownloadErrorKind::Network,
                message: "Hugging Face main did not resolve to an immutable commit".to_owned(),
                retryable: true,
                resumable: false,
            });
        }
    } else {
        require_requested_revision(revision, Some(&commit)).map_err(|message| DownloadError {
            kind: DownloadErrorKind::SourceUnavailable,
            message,
            retryable: false,
            resumable: false,
        })?;
    }

    for component in components {
        let metadata = match resolve_remote_metadata(repo, &api, &commit, &component.path).await {
            Ok(metadata) => metadata,
            Err(error) if equivalent_to_revision.is_some() && missing_upstream_content(&error) => {
                let pinned = equivalent_to_revision.expect("checked equivalent package");
                return Err(package_unavailable(
                    repository,
                    pinned,
                    Some(&commit),
                    Some(&component.path),
                    "required file is absent from current main",
                ));
            }
            Err(error) => return Err(error),
        };
        if metadata.size == 0 {
            return Err(DownloadError {
                kind: DownloadErrorKind::SourceUnavailable,
                message: format!(
                    "Hugging Face did not report a non-zero size for {}",
                    component.path.display()
                ),
                retryable: false,
                resumable: false,
            });
        }
        if let Some(pinned) = equivalent_to_revision {
            validate_equivalent_file(repository, pinned, &commit, component, &metadata)?;
        }
    }
    Ok(commit)
}

fn is_immutable_commit(value: &str) -> bool {
    (40..=64).contains(&value.len())
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn validate_equivalent_file(
    repository: &str,
    pinned: &str,
    observed: &str,
    expected: &ModelComponent,
    metadata: &ResolvedRemoteMetadata,
) -> Result<(), DownloadError> {
    if metadata.size != expected.size_bytes {
        return Err(package_unavailable(
            repository,
            pinned,
            Some(observed),
            Some(&expected.path),
            "current main reports a different file size",
        ));
    }
    if metadata
        .sha256
        .as_deref()
        .is_none_or(|digest| match &expected.content {
            ContentIdentity::Sha256 { value } => !digest.eq_ignore_ascii_case(value),
            _ => true,
        })
    {
        return Err(package_unavailable(
            repository,
            pinned,
            Some(observed),
            Some(&expected.path),
            "current main reports different file content",
        ));
    }
    Ok(())
}

fn package_unavailable(
    repository: &str,
    pinned: &str,
    observed: Option<&str>,
    path: Option<&Path>,
    reason: &'static str,
) -> DownloadError {
    tracing::warn!(
        repository,
        pinned_revision = pinned,
        observed_revision = observed.unwrap_or("unresolved"),
        path = %path.map_or_else(|| "unknown".to_owned(), |path| path.display().to_string()),
        reason,
        "pinned model package is unavailable and current main is not equivalent"
    );
    DownloadError {
        kind: DownloadErrorKind::SourceUnavailable,
        message: match path {
            Some(path) => format!(
                "the publisher no longer provides the catalog package at {}: {reason}",
                path.display()
            ),
            None => format!("the publisher no longer provides the catalog package: {reason}"),
        },
        retryable: true,
        resumable: false,
    }
}

async fn hub_api_metadata(
    client: &hf_hub::HFClient,
    repository: &str,
    revision: &str,
) -> Result<HubApiModel, DownloadError> {
    let url =
        revision_metadata_url(client.endpoint(), repository, revision).map_err(|message| {
            DownloadError {
                kind: DownloadErrorKind::InvalidRequest,
                message,
                retryable: false,
                resumable: false,
            }
        })?;
    let http = reqwest::Client::builder().build().map_err(download_io)?;
    let mut request = http.get(url).query(&[("blobs", "true")]);
    if let Some(token) = std::env::var_os("HF_TOKEN").and_then(|value| value.into_string().ok()) {
        request = request.bearer_auth(token);
    }
    let response = request.send().await.map_err(reqwest_download_error)?;
    let status = response.status();
    if !status.is_success() {
        let retryable = status.as_u16() == 429 || status.is_server_error();
        return Err(DownloadError {
            kind: match status.as_u16() {
                401 | 403 => DownloadErrorKind::SourceAccessDenied,
                404 => DownloadErrorKind::MissingSource,
                _ if retryable => DownloadErrorKind::Network,
                _ => DownloadErrorKind::InvalidRequest,
            },
            message: format!("Hugging Face repository metadata returned HTTP {status}"),
            retryable,
            resumable: false,
        });
    }
    response.json().await.map_err(reqwest_download_error)
}

async fn resolve_remote_metadata(
    repo: &hf_hub::HFRepository<hf_hub::RepoTypeModel>,
    api: &HubApiModel,
    commit: &str,
    path: &Path,
) -> Result<ResolvedRemoteMetadata, DownloadError> {
    let filename = path.to_string_lossy().into_owned();
    match repo
        .get_file_metadata()
        .filepath(filename.clone())
        .revision(commit)
        .send()
        .await
    {
        Ok(metadata) => {
            if metadata.commit_hash != commit {
                return Err(DownloadError {
                    kind: DownloadErrorKind::SourceUnavailable,
                    message: format!("{} resolved outside pinned commit", path.display()),
                    retryable: true,
                    resumable: false,
                });
            }
            let sha256 = (metadata.etag.len() == 64
                && metadata.etag.bytes().all(|byte| byte.is_ascii_hexdigit()))
            .then(|| metadata.etag.to_ascii_lowercase())
            .or_else(|| {
                api.siblings
                    .iter()
                    .find(|sibling| sibling.rfilename == filename)
                    .and_then(|sibling| sibling.lfs.as_ref())
                    .map(|lfs| lfs.sha256.to_ascii_lowercase())
            });
            Ok(ResolvedRemoteMetadata {
                size: metadata.file_size,
                sha256,
            })
        }
        // hf-hub 1.0 follows the HEAD redirect before reading resolver headers. The repository
        // API with `blobs=true` is the narrow fallback when those headers are lost at the CDN.
        Err(HFError::MalformedResponse { .. }) => {
            let sibling = api
                .siblings
                .iter()
                .find(|candidate| candidate.rfilename == filename)
                .ok_or_else(|| DownloadError {
                    kind: DownloadErrorKind::MissingSource,
                    message: format!("Hugging Face repository has no {}", path.display()),
                    retryable: false,
                    resumable: false,
                })?;
            let (size, sha256) = match sibling.lfs.as_ref() {
                Some(lfs) => (lfs.size, Some(lfs.sha256.to_ascii_lowercase())),
                None => (sibling.size.unwrap_or(0), None),
            };
            Ok(ResolvedRemoteMetadata { size, sha256 })
        }
        Err(error) => Err(map_hf_error(error)),
    }
}

fn reqwest_download_error(error: reqwest::Error) -> DownloadError {
    DownloadError {
        kind: DownloadErrorKind::Network,
        message: error.to_string(),
        retryable: error.is_timeout() || error.is_connect() || error.is_request(),
        resumable: false,
    }
}

fn map_hf_error(error: HFError) -> DownloadError {
    let (kind, retryable) = match &error {
        HFError::AuthRequired { .. } | HFError::Forbidden { .. } => {
            (DownloadErrorKind::SourceAccessDenied, false)
        }
        HFError::RepoNotFound { .. }
        | HFError::RevisionNotFound { .. }
        | HFError::EntryNotFound { .. } => (DownloadErrorKind::MissingSource, false),
        HFError::RateLimited { .. } | HFError::Request { .. } | HFError::Xet { .. } => {
            (DownloadErrorKind::Network, true)
        }
        HFError::Http { context } => {
            let retryable = context.status.as_u16() == 408 || context.status.is_server_error();
            (
                if retryable {
                    DownloadErrorKind::Network
                } else {
                    DownloadErrorKind::SourceUnavailable
                },
                retryable,
            )
        }
        HFError::Io(_) => (DownloadErrorKind::FileSystem, true),
        HFError::MalformedResponse { .. } => (DownloadErrorKind::Network, true),
        _ => (DownloadErrorKind::SourceUnavailable, false),
    };
    DownloadError {
        kind,
        message: error.to_string(),
        retryable,
        resumable: retryable,
    }
}

fn download_io(error: impl std::fmt::Display) -> DownloadError {
    DownloadError {
        kind: DownloadErrorKind::FileSystem,
        message: error.to_string(),
        retryable: true,
        resumable: true,
    }
}

fn watch_stream(receiver: watch::Receiver<ModelDownloadEvent>) -> DownloadEventStream {
    stream::unfold(
        (receiver, false, false),
        |(mut receiver, started, terminal)| async move {
            if terminal {
                return None;
            }
            if started && receiver.changed().await.is_err() {
                return None;
            }
            let event = receiver.borrow_and_update().clone();
            let terminal = matches!(
                event,
                ModelDownloadEvent::Ready { .. }
                    | ModelDownloadEvent::Cancelled { .. }
                    | ModelDownloadEvent::Failed { .. }
            );
            Some((event, (receiver, true, terminal)))
        },
    )
    .boxed()
}

fn current_model_id(event: &ModelDownloadEvent) -> Option<icn_contracts::ModelId> {
    match event {
        ModelDownloadEvent::CheckingSpace { model_id, .. }
        | ModelDownloadEvent::Progress { model_id, .. } => Some(model_id.clone()),
        ModelDownloadEvent::Ready { model, .. } => Some(model.id.clone()),
        ModelDownloadEvent::Cancelled { model_id, .. } => model_id.clone(),
        ModelDownloadEvent::Failed { model_id, .. } => model_id.clone(),
        ModelDownloadEvent::Resolving { .. } => None,
    }
}

fn progress_totals(event: &ModelDownloadEvent) -> (u64, u64) {
    match event {
        ModelDownloadEvent::CheckingSpace {
            completed_bytes,
            total_bytes,
            ..
        }
        | ModelDownloadEvent::Progress {
            completed_bytes,
            total_bytes,
            ..
        }
        | ModelDownloadEvent::Failed {
            completed_bytes,
            total_bytes,
            ..
        }
        | ModelDownloadEvent::Cancelled {
            completed_bytes,
            total_bytes,
            ..
        } => (*completed_bytes, *total_bytes),
        ModelDownloadEvent::Ready { model, .. } => (
            model
                .location
                .components()
                .iter()
                .map(|item| item.size_bytes)
                .sum(),
            model
                .location
                .components()
                .iter()
                .map(|item| item.size_bytes)
                .sum(),
        ),
        ModelDownloadEvent::Resolving { .. } => (0, 0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inventory::InventoryConfig;
    use crate::package_service::package_from_resolved;
    use icn_contracts::models::{
        ModelFile, ModelFileId, ModelFileRelationship, ModelFileRole, ModelPackageId,
        ModelPackageProperties, ModelPackageSource,
    };
    use icn_contracts::{
        CapabilityEvidence, ComponentRelationship, ComponentRole, EffectiveTemplateInputs,
        ReasoningCapability, ReasoningControlDomain, ReasoningDelimiters, ReasoningVisibility,
        ResolvedModel, TemplateAssessment, TemplateAssessor, TemplateCapabilities,
    };

    struct DownloadTestTemplateAssessor;

    impl TemplateAssessor for DownloadTestTemplateAssessor {
        fn cache_identity(&self) -> &str {
            "download-test-template-assessor"
        }

        fn assess(&self, _: &EffectiveTemplateInputs) -> Result<TemplateAssessment, String> {
            Ok(TemplateAssessment {
                capabilities: TemplateCapabilities {
                    string_content: true,
                    typed_content: false,
                    tools: false,
                    tool_calls: false,
                    parallel_tool_calls: false,
                    system_role: true,
                    preserve_reasoning: false,
                    object_arguments: false,
                    enable_thinking: false,
                },
                reasoning: ReasoningCapability::Supported {
                    control: ReasoningControlDomain::Effort {
                        levels: vec!["none".to_owned()],
                        default: Some("none".to_owned()),
                    },
                    visibility: ReasoningVisibility::Hidden,
                    delimiters: ReasoningDelimiters::Unavailable,
                    evidence: CapabilityEvidence::BoundedTemplateProbe {
                        fingerprint: "download-test".to_owned(),
                    },
                },
                fingerprint: "download-test".to_owned(),
            })
        }
    }

    fn model_component(contents: &[u8]) -> ModelComponent {
        let digest = format!("{:x}", Sha256::digest(contents));
        ModelComponent {
            path: PathBuf::from("model.gguf"),
            role: ComponentRole::Weights,
            size_bytes: contents.len() as u64,
            content: ContentIdentity::Sha256 { value: digest },
            shard_index: None,
            relationship: None,
        }
    }

    fn exact_package(path: &str, contents: &[u8]) -> ModelPackage {
        let sha256 = format!("{:x}", Sha256::digest(contents));
        ModelPackage {
            id: ModelPackageId("package_test".to_owned()),
            source: ModelPackageSource::HuggingFace {
                repository: "owner/repository".to_owned(),
                revision: "a".repeat(40),
            },
            files: vec![ModelFile {
                id: ModelFileId(format!("file_{sha256}")),
                path: PathBuf::from(path),
                role: ModelFileRole::Weights,
                size_bytes: contents.len() as u64,
                tensor_storage_bytes: None,
                sha256,
            }],
            relationships: Vec::new(),
            properties: ModelPackageProperties {
                format: "gguf".to_owned(),
                quantization: "test".to_owned(),
                quantization_name: "test".to_owned(),
                architecture: "test".to_owned(),
                maximum_context_length: 1,
                intrinsic_model_id: None,
                intrinsic_quality_id: None,
            },
        }
    }

    #[tokio::test]
    async fn publishing_a_revision_retains_other_snapshots_and_blobs() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let repository = directory.path().join("repository");
        tokio::fs::create_dir_all(repository.join("snapshots/old"))
            .await
            .expect("old snapshot");
        tokio::fs::create_dir_all(repository.join("snapshots/current"))
            .await
            .expect("current snapshot");
        tokio::fs::create_dir_all(repository.join("blobs"))
            .await
            .expect("blobs");
        tokio::fs::write(repository.join("blobs/model"), b"verified")
            .await
            .expect("blob");

        let component = model_component(b"current");
        let paths =
            DownloadComponentPaths::new(&repository.join("blobs"), &blob_key(&component.content));
        tokio::fs::write(&paths.blob, b"current")
            .await
            .expect("current blob");
        publish_package_snapshot(
            &repository,
            &repository.join("snapshots/current"),
            "current",
            &[component],
        )
        .await
        .expect("additive snapshot publication");

        assert!(repository.join("snapshots/old").is_dir());
        assert!(repository.join("snapshots/current").is_dir());
        assert!(repository.join("blobs/model").is_file());
    }

    #[tokio::test]
    async fn publishing_the_first_revision_creates_the_snapshot_directory() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let repository = directory.path().join("repository");
        let staged = repository.join(".incomplete/snapshot-current");
        let published = repository.join("snapshots/current");
        tokio::fs::create_dir_all(&staged)
            .await
            .expect("staged snapshot");
        tokio::fs::write(staged.join("model.gguf"), b"verified")
            .await
            .expect("staged model");

        let component = model_component(b"verified");
        let paths =
            DownloadComponentPaths::new(&repository.join("blobs"), &blob_key(&component.content));
        tokio::fs::create_dir_all(repository.join("blobs"))
            .await
            .expect("blob directory");
        tokio::fs::write(&paths.blob, b"verified")
            .await
            .expect("verified blob");

        publish_package_snapshot(&repository, &published, "current", &[component])
            .await
            .expect("first snapshot publication");

        assert!(!staged.exists());
        assert_eq!(
            tokio::fs::read(published.join("model.gguf"))
                .await
                .expect("published model"),
            b"verified"
        );
    }

    #[tokio::test]
    async fn publishing_another_package_into_the_same_revision_is_additive() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let repository = directory.path().join("repository");
        let snapshot = repository.join("snapshots/current");

        let mut target = model_component(b"target");
        target.path = PathBuf::from("target.gguf");
        let mut draft = model_component(b"draft");
        draft.path = PathBuf::from("draft.gguf");
        for (component, contents) in [
            (&target, b"target".as_slice()),
            (&draft, b"draft".as_slice()),
        ] {
            let paths = DownloadComponentPaths::new(
                &repository.join("blobs"),
                &blob_key(&component.content),
            );
            tokio::fs::create_dir_all(repository.join("blobs"))
                .await
                .expect("blob directory");
            tokio::fs::write(paths.blob, contents)
                .await
                .expect("verified blob");
        }

        publish_package_snapshot(&repository, &snapshot, "current", &[target])
            .await
            .expect("target publication");
        publish_package_snapshot(&repository, &snapshot, "current", &[draft])
            .await
            .expect("draft publication");

        assert_eq!(
            tokio::fs::read(snapshot.join("target.gguf")).await.unwrap(),
            b"target"
        );
        assert_eq!(
            tokio::fs::read(snapshot.join("draft.gguf")).await.unwrap(),
            b"draft"
        );
    }

    #[tokio::test]
    async fn installed_package_admission_does_not_create_an_upstream_operation() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let root = directory.path().join("models");
        let cache_root = directory.path().join("cache");
        let hf_cache = directory.path().join("hugging-face");
        let snapshot = hf_cache
            .join("models--owner--repository")
            .join("snapshots")
            .join("a".repeat(40));
        tokio::fs::create_dir_all(&snapshot)
            .await
            .expect("snapshot directory");
        let mut gguf = Vec::new();
        gguf.extend_from_slice(b"GGUF");
        gguf.extend_from_slice(&3_u32.to_le_bytes());
        gguf.extend_from_slice(&0_u64.to_le_bytes());
        gguf.extend_from_slice(&0_u64.to_le_bytes());
        gguf.resize(32, 0);
        tokio::fs::write(snapshot.join("model.gguf"), gguf)
            .await
            .expect("installed model");
        let mut config = InventoryConfig::with_roots(root, cache_root).expect("inventory config");
        config.hf_cache_dirs.push(hf_cache);
        let manager = ManagedModelStore::open_with_template_assessor(
            config,
            Some(Arc::new(DownloadTestTemplateAssessor)),
        )
        .await
        .expect("model manager");
        manager
            .ensure_installed_model_inventory()
            .await
            .expect("initial inventory");
        let model = manager
            .models
            .read()
            .expect("inventory lock")
            .values()
            .next()
            .cloned()
            .expect("installed inventory model");
        let resolved = ResolvedModel {
            components: crate::service::resolve_components(manager.root(), &model)
                .expect("resolved components"),
            model: model.clone(),
        };
        let package = package_from_resolved(&resolved).expect("installed package");

        let mut streams = manager
            .start_target_downloads(vec![package])
            .await
            .expect("admission");
        let event = streams
            .pop()
            .expect("download stream")
            .next()
            .await
            .expect("ready event");

        assert!(matches!(
            event,
            ModelDownloadEvent::Ready { model: ready, .. } if ready.id == model.id
        ));
        assert!(manager.operations.lock().await.is_empty());
    }

    #[test]
    fn streamed_integrity_matches_the_complete_source_digest() {
        let contents = b"streamed model contents";
        let component = model_component(contents);
        let mut integrity = DownloadIntegrity::empty(&component);

        integrity.update(&contents[..8]).expect("first chunk");
        integrity.update(&contents[8..]).expect("second chunk");

        integrity.verify(&component).expect("matching digest");
    }

    #[tokio::test]
    async fn resumed_integrity_restores_the_existing_prefix_without_reading_it() {
        let contents = b"resumed model contents";
        let split = 9;
        let component = model_component(contents);
        let directory = tempfile::tempdir().expect("temporary directory");
        let paths = DownloadComponentPaths::new(directory.path(), &blob_key(&component.content));
        tokio::fs::write(&paths.partial, &contents[..split])
            .await
            .expect("partial contents");
        let mut original = DownloadIntegrity::empty(&component);
        original
            .update(&contents[..split])
            .expect("downloaded prefix");
        atomic_json(&paths.checkpoint, &original.record(&component))
            .await
            .expect("integrity checkpoint");

        let mut integrity = recover_partial(&paths, &component)
            .await
            .expect("resume integrity");
        integrity
            .update(&contents[split..])
            .expect("remaining bytes");

        integrity.verify(&component).expect("matching digest");
    }

    #[tokio::test]
    async fn resume_truncates_bytes_beyond_the_durable_checkpoint() {
        let contents = b"checkpoint plus uncommitted tail";
        let split = 10;
        let component = model_component(contents);
        let directory = tempfile::tempdir().expect("temporary directory");
        let paths = DownloadComponentPaths::new(directory.path(), &blob_key(&component.content));
        tokio::fs::write(&paths.partial, contents)
            .await
            .expect("partial contents");
        let mut original = DownloadIntegrity::empty(&component);
        original
            .update(&contents[..split])
            .expect("checkpointed prefix");
        atomic_json(&paths.checkpoint, &original.record(&component))
            .await
            .expect("integrity checkpoint");

        let integrity = recover_partial(&paths, &component)
            .await
            .expect("resume integrity");

        assert_eq!(integrity.bytes, split as u64);
        assert_eq!(
            tokio::fs::metadata(&paths.partial)
                .await
                .expect("partial metadata")
                .len(),
            split as u64
        );
    }

    #[tokio::test]
    async fn completed_blob_uses_the_final_checkpoint_without_rehashing() {
        let contents = b"completed model contents";
        let component = model_component(contents);
        let directory = tempfile::tempdir().expect("temporary directory");
        let paths = DownloadComponentPaths::new(directory.path(), &blob_key(&component.content));
        tokio::fs::write(&paths.blob, contents)
            .await
            .expect("completed blob");
        let mut integrity = DownloadIntegrity::empty(&component);
        integrity.update(contents).expect("downloaded contents");
        atomic_json(&paths.checkpoint, &integrity.record(&component))
            .await
            .expect("final integrity checkpoint");

        assert!(
            recover_completed_blob(&paths, &component)
                .await
                .expect("completed blob recovery")
        );
        assert!(paths.checkpoint.is_file());
    }

    #[tokio::test]
    async fn partial_without_a_valid_checkpoint_restarts_cleanly() {
        let contents = b"untrusted partial model contents";
        let component = model_component(contents);
        let directory = tempfile::tempdir().expect("temporary directory");
        let paths = DownloadComponentPaths::new(directory.path(), &blob_key(&component.content));
        tokio::fs::write(&paths.partial, contents)
            .await
            .expect("partial contents");
        tokio::fs::write(&paths.checkpoint, b"not a checkpoint")
            .await
            .expect("invalid checkpoint");

        let integrity = recover_partial(&paths, &component)
            .await
            .expect("clean restart");

        assert_eq!(integrity.bytes, 0);
        assert!(!paths.partial.exists());
        assert!(!paths.checkpoint.exists());
    }

    #[test]
    fn streamed_integrity_rejects_mismatched_content() {
        let expected = b"expected model contents";
        let component = model_component(expected);
        let mut different = expected.to_vec();
        different[0] ^= 1;
        let mut integrity = DownloadIntegrity::empty(&component);
        integrity.update(&different).expect("streamed bytes");

        let error = integrity.verify(&component).expect_err("digest mismatch");

        assert!(matches!(error.kind, DownloadErrorKind::Integrity));
        assert!(!error.retryable());
        assert!(!error.resumable());
    }

    #[test]
    fn insufficient_disk_failure_preserves_required_and_available_bytes() {
        let failure = DownloadError {
            kind: DownloadErrorKind::InsufficientDiskSpace {
                required_bytes: 37_923_968_128,
                available_bytes: 33_440_665_600,
            },
            message: "insufficient disk space".to_owned(),
            retryable: false,
            resumable: true,
        };

        assert_eq!(
            failure
                .to_failure()
                .expect("test failure is not cancellation"),
            DownloadFailure::InsufficientDiskSpace {
                required_bytes: 37_923_968_128,
                available_bytes: 33_440_665_600,
            }
        );
    }

    #[test]
    fn equivalent_revision_requires_matching_path_size_and_sha256() {
        let expected = model_component(b"model contents");
        let ContentIdentity::Sha256 { value: sha256 } = &expected.content else {
            unreachable!("test component uses SHA-256")
        };
        let metadata = ResolvedRemoteMetadata {
            size: expected.size_bytes,
            sha256: Some(sha256.clone()),
        };
        validate_equivalent_file(
            "owner/repository",
            &"a".repeat(40),
            &"b".repeat(40),
            &expected,
            &metadata,
        )
        .expect("equivalent file");

        let different_size = ResolvedRemoteMetadata {
            size: expected.size_bytes + 1,
            sha256: Some(sha256.clone()),
        };
        assert!(matches!(
            validate_equivalent_file(
                "owner/repository",
                &"a".repeat(40),
                &"b".repeat(40),
                &expected,
                &different_size,
            )
            .expect_err("changed size")
            .kind,
            DownloadErrorKind::SourceUnavailable
        ));

        let changed = ResolvedRemoteMetadata {
            size: expected.size_bytes,
            sha256: Some("0".repeat(64)),
        };
        let failure = validate_equivalent_file(
            "owner/repository",
            &"a".repeat(40),
            &"b".repeat(40),
            &expected,
            &changed,
        )
        .expect_err("changed file");
        assert!(matches!(failure.kind, DownloadErrorKind::SourceUnavailable));
    }

    #[test]
    fn package_components_preserve_roles_and_relationships() {
        let mut package = exact_package("model.gguf", b"model contents");
        let projector_sha = format!("{:x}", Sha256::digest(b"projector contents"));
        let projector_id = ModelFileId(format!("file_{projector_sha}"));
        package.files.push(ModelFile {
            id: projector_id.clone(),
            path: PathBuf::from("projector.gguf"),
            role: ModelFileRole::Projector,
            size_bytes: 18,
            tensor_storage_bytes: None,
            sha256: projector_sha,
        });
        package
            .relationships
            .push(ModelFileRelationship::ProjectorFor {
                projector_file_id: projector_id,
                weights_file_id: package.files[0].id.clone(),
            });

        let package = ValidatedDownloadPackage::new(package).expect("valid package");
        let components = package.components();
        let projector = &components[1];

        assert_eq!(projector.role, ComponentRole::Projector);
        assert_eq!(projector.shard_index, None);
        assert_eq!(
            projector.relationship,
            Some(ComponentRelationship::ProjectorFor {
                projector: PathBuf::from("projector.gguf"),
                model: PathBuf::from("model.gguf"),
            })
        );
    }

    #[test]
    fn fallback_is_limited_to_definitive_missing_content() {
        let failure = |kind| DownloadError {
            kind,
            message: String::new(),
            retryable: false,
            resumable: false,
        };
        assert!(missing_upstream_content(&failure(
            DownloadErrorKind::MissingSource
        )));
        assert!(!missing_upstream_content(&failure(
            DownloadErrorKind::SourceUnavailable
        )));
        assert!(!missing_upstream_content(&failure(
            DownloadErrorKind::Network
        )));
    }
}
