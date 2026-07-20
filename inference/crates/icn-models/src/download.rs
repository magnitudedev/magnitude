use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use fs2::FileExt;
use futures_util::{StreamExt, stream};
use getrandom::fill;
use hf_hub::HFError;
use icn_contracts::{
    ContentIdentity, DownloadEventStream, DownloadFailure, DownloadFileProgress,
    DownloadModelRequest, DownloadStage, HardwareAssessment, HuggingFaceDownloadSource, Integrity,
    InventoryError, InventoryModel, InventoryProperties, ModelComponent, ModelDownloadEvent,
    ModelLocation, ModelSource, ModelStatus,
};
use sha2::{Digest, Sha256};
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};
use tokio::sync::watch;

use crate::identity::{content_id, model_id};
use crate::inventory::{ModelManager, build_model, hf_repo_dir, now};
use crate::manifest::{MANIFEST_VERSION, ManagedManifest, OperationComponent, OperationManifest};
use crate::validation::validate_download_request;

const MAX_ATTEMPTS: usize = 5;

pub(crate) struct DownloadOperation {
    sender: watch::Sender<ModelDownloadEvent>,
}

impl DownloadOperation {
    fn subscribe(&self) -> DownloadEventStream {
        watch_stream(self.sender.subscribe())
    }
}

#[derive(Debug)]
struct RemoteComponent {
    request: icn_contracts::DownloadComponent,
    size: u64,
    content: ContentIdentity,
    content_key: String,
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
    #[serde(rename = "blobId")]
    blob_id: Option<String>,
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
    content: ContentIdentity,
}

#[derive(Debug, thiserror::Error)]
#[error("{message}")]
struct DownloadError {
    code: &'static str,
    message: String,
    retryable: bool,
    resumable: bool,
}

impl ModelManager {
    pub(crate) async fn start_download(
        &self,
        request: DownloadModelRequest,
    ) -> Result<DownloadEventStream, InventoryError> {
        validate_download_request(&request)?;
        let key = request_key(&request);
        let mut operations = self.operations.lock().await;
        if let Some(operation) = operations.get(&key) {
            return Ok(operation.subscribe());
        }

        let operation_id = random_id("download")?;
        let HuggingFaceDownloadSource::HuggingFace {
            repository,
            revision,
        } = &request.source;
        let initial = ModelDownloadEvent::Resolving {
            operation_id: operation_id.clone(),
            repository: repository.clone(),
            revision: revision.clone(),
        };
        let (sender, _receiver) = watch::channel(initial);
        let operation = Arc::new(DownloadOperation { sender });
        let stream = operation.subscribe();
        operations.insert(key.clone(), Arc::clone(&operation));
        drop(operations);

        let manager = self.clone();
        tokio::spawn(async move {
            manager
                .run_download(key, operation_id, request, operation)
                .await;
        });
        Ok(stream)
    }

    async fn run_download(
        &self,
        operation_key: String,
        operation_id: String,
        request: DownloadModelRequest,
        operation: Arc<DownloadOperation>,
    ) {
        let result = self
            .run_download_inner(&operation_id, &request, &operation)
            .await;
        if let Err(failure) = result {
            let model_id = current_model_id(&operation.sender.borrow());
            if let Some(model_id) = model_id.as_ref() {
                persist_operation_failure(&self.config.root, model_id, failure.code).await;
            }
            if let Some(model_id) = model_id.as_ref()
                && let Ok(mut models) = self.models.write()
                && let Some(model) = models.get_mut(model_id)
            {
                let (completed_bytes, total_bytes) = progress_totals(&operation.sender.borrow());
                model.status = ModelStatus::Interrupted {
                    completed_bytes,
                    total_bytes,
                    resumable: failure.resumable,
                    reason: (!failure.resumable).then(|| failure.code.to_owned()),
                    last_error: failure.code.to_owned(),
                    updated_at: now(),
                };
                model.updated_at = now();
            }
            let (completed_bytes, total_bytes) = progress_totals(&operation.sender.borrow());
            operation.sender.send_replace(ModelDownloadEvent::Failed {
                operation_id,
                model_id,
                error: DownloadFailure {
                    code: failure.code.to_owned(),
                    message: failure.message,
                    retryable: failure.retryable,
                },
                completed_bytes,
                total_bytes,
                resumable: failure.resumable,
            });
        }
        self.operations.lock().await.remove(&operation_key);
    }

    async fn run_download_inner(
        &self,
        operation_id: &str,
        request: &DownloadModelRequest,
        operation: &DownloadOperation,
    ) -> Result<(), DownloadError> {
        let HuggingFaceDownloadSource::HuggingFace {
            repository,
            revision,
        } = &request.source;
        let (owner, name) = repository.split_once('/').ok_or_else(|| DownloadError {
            code: "invalid_request",
            message: "repository must be owner/name".to_owned(),
            retryable: false,
            resumable: false,
        })?;
        let repo = self.client.model(owner.to_owned(), name.to_owned());

        let api_metadata = hub_api_metadata(&self.client, repository, revision).await?;
        let commit = api_metadata.sha.clone().ok_or_else(|| DownloadError {
            code: "missing_metadata",
            message: "Hugging Face repository response did not include a commit".to_owned(),
            retryable: true,
            resumable: false,
        })?;
        let mut remote = Vec::with_capacity(request.components.len());
        for component in &request.components {
            let metadata =
                resolve_remote_metadata(&repo, &api_metadata, &commit, &component.path).await?;
            if metadata.size == 0 {
                return Err(DownloadError {
                    code: "missing_metadata",
                    message: format!(
                        "Hugging Face did not report a non-zero size for {}",
                        component.path.display()
                    ),
                    retryable: false,
                    resumable: false,
                });
            }
            let content = if let Some(expected) = component.expected_sha256.as_ref() {
                ContentIdentity::Sha256 {
                    value: expected.to_ascii_lowercase(),
                }
            } else {
                metadata.content
            };
            let content_key = blob_key(&content);
            remote.push(RemoteComponent {
                request: component.clone(),
                size: metadata.size,
                content,
                content_key,
            });
        }
        let components = remote
            .iter()
            .map(|component| ModelComponent {
                path: component.request.path.clone(),
                role: component.request.role.clone(),
                size_bytes: component.size,
                content: component.content.clone(),
                shard_index: component.request.shard_index,
                relationship: request.relationships.iter().find_map(|relationship| {
                    relationship_component(relationship)
                        .is_some_and(|path| path == component.request.path)
                        .then(|| relationship.clone())
                }),
            })
            .collect::<Vec<_>>();
        let content_id = content_id(&components);
        let repo_root = self.config.root.join("hub").join(hf_repo_dir(repository));
        let snapshot = repo_root.join("snapshots").join(&commit);
        let model_id = model_id("magnitude-cache", &snapshot, &content_id);
        let total_bytes: u64 = remote.iter().map(|component| component.size).sum();
        let previous = read_operation_manifest(&self.config.root, &model_id).await;
        let completed_bytes = resumable_bytes(&repo_root, &remote, previous.as_ref()).await;
        let missing_bytes = total_bytes.saturating_sub(completed_bytes);
        let available_bytes =
            fs2::available_space(&self.config.root).map_err(|error| DownloadError {
                code: "disk_inspection_failed",
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
        if missing_bytes.saturating_add(self.config.disk_reserve_bytes) > available_bytes {
            return Err(DownloadError {
                code: "insufficient_disk",
                message: format!(
                    "download requires {} bytes including reserve, but {} bytes are available",
                    missing_bytes.saturating_add(self.config.disk_reserve_bytes),
                    available_bytes
                ),
                retryable: false,
                resumable: true,
            });
        }

        if let Ok(models) = self.models.read()
            && let Some(existing) = models.get(&model_id)
            && matches!(
                existing.status,
                ModelStatus::Available { .. } | ModelStatus::Loaded { .. }
            )
        {
            operation.sender.send_replace(ModelDownloadEvent::Ready {
                operation_id: operation_id.to_owned(),
                model: Box::new(existing.clone()),
            });
            return Ok(());
        }

        let started_at = now();
        let planned = InventoryModel {
            id: model_id.clone(),
            content_id: content_id.clone(),
            created: started_at,
            name: repository.clone(),
            supported_parameters: Vec::new(),
            status: ModelStatus::Downloading {
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
            hardware: HardwareAssessment::NotAssessed {
                reason: "model_not_ready".to_owned(),
            },
            operations: Vec::new(),
            updated_at: started_at,
        };
        self.models
            .write()
            .map_err(|_| DownloadError {
                code: "internal",
                message: "inventory lock poisoned".to_owned(),
                retryable: false,
                resumable: true,
            })?
            .insert(model_id.clone(), planned);

        if let Some(first) = remote.first() {
            operation.sender.send_replace(ModelDownloadEvent::Progress {
                operation_id: operation_id.to_owned(),
                model_id: model_id.clone(),
                stage: DownloadStage::Queued,
                completed_bytes,
                total_bytes,
                file: DownloadFileProgress {
                    path: first.request.path.clone(),
                    completed_bytes: component_partial_len(&repo_root, first).await,
                    total_bytes: first.size,
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
                code: "internal",
                message: error.to_string(),
                retryable: false,
                resumable: true,
            })?;
        tokio::fs::create_dir_all(repo_root.join("blobs"))
            .await
            .map_err(download_io)?;
        tokio::fs::create_dir_all(&snapshot)
            .await
            .map_err(download_io)?;

        let lock_path = self
            .config
            .root
            .join("locks")
            .join(format!("{}.lock", model_id.0));
        let lock_file = acquire_lock(lock_path).await?;
        let mut operation_manifest = OperationManifest {
            version: MANIFEST_VERSION,
            operation_id: operation_id.to_owned(),
            model_id: model_id.clone(),
            content_id: content_id.clone(),
            repository: repository.clone(),
            requested_revision: revision.clone(),
            commit: commit.clone(),
            components: remote
                .iter()
                .map(|component| OperationComponent {
                    path: component.request.path.clone(),
                    role: component.request.role.clone(),
                    content: component.content.clone(),
                    shard_index: component.request.shard_index,
                    relationship: request.relationships.iter().find_map(|relationship| {
                        relationship_component(relationship)
                            .is_some_and(|path| path == component.request.path)
                            .then(|| relationship.clone())
                    }),
                    expected_size: component.size,
                    content_key: component.content_key.clone(),
                    completed_bytes: 0,
                })
                .collect(),
            stage: "downloading".to_owned(),
            started_at,
            updated_at: started_at,
            last_error: None,
        };
        persist_operation_manifest(&self.config.root, &operation_manifest).await?;

        let started = Instant::now();
        for (index, component) in remote.iter().enumerate() {
            let resumed_from = component_partial_len(&repo_root, component).await;
            let mut last_progress_emit = Instant::now()
                .checked_sub(Duration::from_millis(100))
                .unwrap_or_else(Instant::now);
            download_component_with_retry(
                &repo,
                &self.config.root,
                &commit,
                component,
                |file_completed| {
                    let timestamp = Instant::now();
                    if file_completed != component.size
                        && timestamp.duration_since(last_progress_emit) < Duration::from_millis(100)
                    {
                        return;
                    }
                    last_progress_emit = timestamp;
                    let previous_files = remote[..index].iter().map(|item| item.size).sum::<u64>();
                    let completed = previous_files.saturating_add(file_completed);
                    let elapsed = started.elapsed().as_secs_f64();
                    let rate = (elapsed > 0.0).then(|| completed as f64 / elapsed);
                    let stage = if file_completed == component.size {
                        DownloadStage::Verifying
                    } else {
                        DownloadStage::Downloading
                    };
                    operation.sender.send_replace(ModelDownloadEvent::Progress {
                        operation_id: operation_id.to_owned(),
                        model_id: model_id.clone(),
                        stage,
                        completed_bytes: completed,
                        total_bytes,
                        file: DownloadFileProgress {
                            path: component.request.path.clone(),
                            completed_bytes: file_completed,
                            total_bytes: component.size,
                        },
                        bytes_per_second: rate,
                        resumed_from_bytes: resumed_from,
                    });
                    if let Ok(mut models) = self.models.write()
                        && let Some(model) = models.get_mut(&model_id)
                    {
                        let updated_at = now();
                        model.status = ModelStatus::Downloading {
                            operation_id: operation_id.to_owned(),
                            stage,
                            completed_bytes: completed,
                            total_bytes,
                            current_component: Some(component.request.path.clone()),
                            started_at,
                            updated_at,
                        };
                        model.updated_at = updated_at;
                    }
                },
            )
            .await?;
            operation_manifest.components[index].completed_bytes = component.size;
            operation_manifest.updated_at = now();
            persist_operation_manifest(&self.config.root, &operation_manifest).await?;
            publish_snapshot_link(&repo_root, &snapshot, component).await?;
        }

        operation_manifest.stage = "verifying".to_owned();
        operation_manifest.updated_at = now();
        persist_operation_manifest(&self.config.root, &operation_manifest).await?;
        if let Some(last) = remote.last() {
            operation.sender.send_replace(ModelDownloadEvent::Progress {
                operation_id: operation_id.to_owned(),
                model_id: model_id.clone(),
                stage: DownloadStage::Verifying,
                completed_bytes: total_bytes,
                total_bytes,
                file: DownloadFileProgress {
                    path: last.request.path.clone(),
                    completed_bytes: last.size,
                    total_bytes: last.size,
                },
                bytes_per_second: None,
                resumed_from_bytes: 0,
            });
        }

        let manifest = ManagedManifest {
            version: MANIFEST_VERSION,
            model_id: model_id.clone(),
            content_id,
            repository: repository.clone(),
            requested_revision: revision.clone(),
            commit,
            components,
            created_at: started_at,
            ready_at: now(),
        };
        persist_managed_manifest(&self.config.root, &manifest).await?;
        let operation_path = operation_manifest_path(&self.config.root, &model_id);
        let _ = tokio::fs::remove_file(operation_path).await;
        drop(lock_file);
        let primary = manifest
            .components
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
                code: "publish_failed",
                message: "published model has no runnable weight component".to_owned(),
                retryable: false,
                resumable: false,
            })?;
        let model = build_model(
            manifest.model_id.clone(),
            manifest.content_id.clone(),
            manifest.created_at,
            manifest.ready_at,
            ModelSource::HuggingFace {
                repository: manifest.repository.clone(),
                requested_revision: manifest.requested_revision.clone(),
                commit: manifest.commit.clone(),
                metadata: None,
            },
            ModelLocation::MagnitudeCache {
                total_bytes: manifest.components.iter().map(|item| item.size_bytes).sum(),
                components: manifest.components.clone(),
                integrity: Integrity::Verified {
                    method: "manifest".to_owned(),
                },
            },
            &primary,
            true,
            &self.cache,
            self.template_assessor.as_deref(),
        )
        .map_err(|error| DownloadError {
            code: "inspection_failed",
            message: error.to_string(),
            retryable: true,
            resumable: true,
        })?;
        let ready = self
            .complete_and_publish_model(model)
            .await
            .map_err(|error| DownloadError {
                code: "publish_failed",
                message: error.to_string(),
                retryable: true,
                resumable: true,
            })?;
        operation.sender.send_replace(ModelDownloadEvent::Ready {
            operation_id: operation_id.to_owned(),
            model: Box::new(ready),
        });
        Ok(())
    }
}

async fn download_component_with_retry(
    repo: &hf_hub::HFRepository<hf_hub::RepoTypeModel>,
    root: &Path,
    commit: &str,
    component: &RemoteComponent,
    mut progress: impl FnMut(u64),
) -> Result<(), DownloadError> {
    for attempt in 0..MAX_ATTEMPTS {
        match download_component_once(repo, root, commit, component, &mut progress).await {
            Ok(()) => return Ok(()),
            Err(error) if error.retryable && attempt + 1 < MAX_ATTEMPTS => {
                tokio::time::sleep(std::time::Duration::from_secs(1_u64 << attempt.min(4))).await;
            }
            Err(error) => return Err(error),
        }
    }
    unreachable!("bounded retry loop returns on its final attempt")
}

async fn download_component_once(
    repo: &hf_hub::HFRepository<hf_hub::RepoTypeModel>,
    root: &Path,
    commit: &str,
    component: &RemoteComponent,
    progress: &mut impl FnMut(u64),
) -> Result<(), DownloadError> {
    let blobs = root
        .join("hub")
        .join(hf_repo_dir(&repo.repo_path()))
        .join("blobs");
    let blob = blobs.join(&component.content_key);
    if tokio::fs::metadata(&blob)
        .await
        .is_ok_and(|metadata| metadata.is_file() && metadata.len() == component.size)
    {
        verify_file(&blob, component).await?;
        progress(component.size);
        return Ok(());
    }
    let partial = blobs.join(format!("{}.incomplete", component.content_key));
    let mut offset = tokio::fs::metadata(&partial)
        .await
        .map(|metadata| metadata.len())
        .unwrap_or(0);
    if offset > component.size {
        quarantine(&partial).await?;
        offset = 0;
    }
    if offset == component.size {
        verify_file(&partial, component).await?;
        tokio::fs::rename(&partial, &blob)
            .await
            .map_err(download_io)?;
        progress(component.size);
        return Ok(());
    }

    let original_offset = offset;
    let (_reported_length, mut stream) = repo
        .download_file_stream()
        .filename(component.request.path.to_string_lossy().into_owned())
        .revision(commit.to_owned())
        .range(offset..component.size)
        .send()
        .await
        .map_err(map_hf_error)?;
    let std_file = open_partial(&partial)?;
    let mut file = tokio::fs::File::from_std(std_file);
    file.seek(std::io::SeekFrom::Start(offset))
        .await
        .map_err(download_io)?;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(map_hf_error)?;
        let chunk_len = u64::try_from(chunk.len()).map_err(|_| DownloadError {
            code: "size_overflow",
            message: "download chunk size overflows u64".to_owned(),
            retryable: false,
            resumable: false,
        })?;
        if offset
            .checked_add(chunk_len)
            .is_none_or(|next| next > component.size)
        {
            file.set_len(original_offset).await.map_err(download_io)?;
            return Err(DownloadError {
                code: "size_mismatch",
                message: format!(
                    "download exceeded expected size for {}",
                    component.request.path.display()
                ),
                retryable: true,
                resumable: true,
            });
        }
        file.write_all(&chunk).await.map_err(download_io)?;
        offset += chunk_len;
        progress(offset);
    }
    file.flush().await.map_err(download_io)?;
    file.sync_all().await.map_err(download_io)?;
    if offset != component.size {
        file.set_len(original_offset).await.map_err(download_io)?;
        return Err(DownloadError {
            code: "size_mismatch",
            message: format!(
                "download ended at {offset} bytes; expected {} for {}",
                component.size,
                component.request.path.display()
            ),
            retryable: true,
            resumable: true,
        });
    }
    drop(file);
    verify_file(&partial, component).await?;
    tokio::fs::rename(&partial, &blob)
        .await
        .map_err(download_io)?;
    Ok(())
}

async fn verify_file(path: &Path, component: &RemoteComponent) -> Result<(), DownloadError> {
    let metadata = tokio::fs::metadata(path).await.map_err(download_io)?;
    if !metadata.is_file() || metadata.len() != component.size {
        return Err(DownloadError {
            code: "size_mismatch",
            message: format!("unexpected size for {}", component.request.path.display()),
            retryable: false,
            resumable: false,
        });
    }
    let ContentIdentity::Sha256 { value: expected } = &component.content else {
        return Ok(());
    };
    let mut file = tokio::fs::File::open(path).await.map_err(download_io)?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer).await.map_err(download_io)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    let actual = format!("{:x}", digest.finalize());
    if &actual != expected {
        quarantine(path).await?;
        return Err(DownloadError {
            code: "integrity_failed",
            message: format!("SHA-256 mismatch for {}", component.request.path.display()),
            retryable: false,
            resumable: false,
        });
    }
    Ok(())
}

async fn publish_snapshot_link(
    repo_root: &Path,
    snapshot: &Path,
    component: &RemoteComponent,
) -> Result<(), DownloadError> {
    let destination = snapshot.join(&component.request.path);
    if let Some(parent) = destination.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(download_io)?;
    }
    let blob = repo_root.join("blobs").join(&component.content_key);
    let destination_clone = destination.clone();
    tokio::task::spawn_blocking(move || -> Result<(), DownloadError> {
        if destination_clone.exists() {
            let canonical = destination_clone.canonicalize().map_err(download_io)?;
            if canonical == blob.canonicalize().map_err(download_io)? {
                return Ok(());
            }
            return Err(DownloadError {
                code: "publication_conflict",
                message: format!(
                    "snapshot path already exists: {}",
                    destination_clone.display()
                ),
                retryable: false,
                resumable: true,
            });
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
        code: "publication_failed",
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

async fn persist_operation_manifest(
    root: &Path,
    manifest: &OperationManifest,
) -> Result<(), DownloadError> {
    atomic_json(&operation_manifest_path(root, &manifest.model_id), manifest).await
}

async fn persist_managed_manifest(
    root: &Path,
    manifest: &ManagedManifest,
) -> Result<(), DownloadError> {
    atomic_json(
        &root
            .join("installations")
            .join(format!("{}.json", manifest.model_id.0)),
        manifest,
    )
    .await
}

async fn atomic_json(path: &Path, value: &impl serde::Serialize) -> Result<(), DownloadError> {
    let bytes = serde_json::to_vec_pretty(value).map_err(|error| DownloadError {
        code: "serialization_failed",
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
    if let Some(parent) = path.parent() {
        let directory = tokio::fs::File::open(parent).await.map_err(download_io)?;
        directory.sync_all().await.map_err(download_io)?;
    }
    Ok(())
}

async fn read_operation_manifest(
    root: &Path,
    model_id: &icn_contracts::ModelId,
) -> Option<OperationManifest> {
    let bytes = tokio::fs::read(operation_manifest_path(root, model_id))
        .await
        .ok()?;
    serde_json::from_slice(&bytes).ok()
}

async fn persist_operation_failure(root: &Path, model_id: &icn_contracts::ModelId, code: &str) {
    let Some(mut manifest) = read_operation_manifest(root, model_id).await else {
        return;
    };
    manifest.last_error = Some(code.to_owned());
    manifest.updated_at = now();
    for component in &mut manifest.components {
        let blobs = root
            .join("hub")
            .join(hf_repo_dir(&manifest.repository))
            .join("blobs");
        let blob = blobs.join(&component.content_key);
        let partial = blobs.join(format!("{}.incomplete", component.content_key));
        component.completed_bytes = if tokio::fs::metadata(&blob)
            .await
            .is_ok_and(|metadata| metadata.is_file() && metadata.len() == component.expected_size)
        {
            component.expected_size
        } else {
            tokio::fs::metadata(partial)
                .await
                .map(|metadata| metadata.len().min(component.expected_size))
                .unwrap_or(0)
        };
    }
    let _ = persist_operation_manifest(root, &manifest).await;
}

fn operation_manifest_path(root: &Path, model_id: &icn_contracts::ModelId) -> PathBuf {
    root.join("operations").join(format!("{}.json", model_id.0))
}

async fn resumable_bytes(
    repo_root: &Path,
    components: &[RemoteComponent],
    previous: Option<&OperationManifest>,
) -> u64 {
    let mut total = 0_u64;
    for component in components {
        let blob = repo_root.join("blobs").join(&component.content_key);
        if tokio::fs::metadata(&blob)
            .await
            .is_ok_and(|metadata| metadata.is_file() && metadata.len() == component.size)
        {
            total = total.saturating_add(component.size);
            continue;
        }
        let matches_sidecar = previous.is_some_and(|manifest| {
            manifest.components.iter().any(|candidate| {
                candidate.path == component.request.path
                    && candidate.expected_size == component.size
                    && candidate.content_key == component.content_key
            })
        });
        let partial = component_partial(repo_root, component);
        let length = tokio::fs::metadata(&partial)
            .await
            .map(|metadata| metadata.len())
            .unwrap_or(0);
        if matches_sidecar || length == component.size {
            total = total.saturating_add(length.min(component.size));
        } else if length > 0 {
            let _ = quarantine(&partial).await;
        }
    }
    total
}

async fn component_partial_len(repo_root: &Path, component: &RemoteComponent) -> u64 {
    tokio::fs::metadata(component_partial(repo_root, component))
        .await
        .map(|metadata| metadata.len().min(component.size))
        .unwrap_or(0)
}

fn component_partial(repo_root: &Path, component: &RemoteComponent) -> PathBuf {
    repo_root
        .join("blobs")
        .join(format!("{}.incomplete", component.content_key))
}

async fn quarantine(path: &Path) -> Result<(), DownloadError> {
    if !path.exists() {
        return Ok(());
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
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(&path)
            .map_err(download_io)?;
        FileExt::lock_exclusive(&file).map_err(download_io)?;
        Ok(file)
    })
    .await
    .map_err(|error| DownloadError {
        code: "lock_failed",
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
            code: "unsafe_path",
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

fn request_key(request: &DownloadModelRequest) -> String {
    let bytes = serde_json::to_vec(request).expect("validated download requests serialize");
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
        code: "internal",
        message: error.to_string(),
        retryable: false,
        resumable: true,
    }
}

async fn hub_api_metadata(
    client: &hf_hub::HFClient,
    repository: &str,
    revision: &str,
) -> Result<HubApiModel, DownloadError> {
    let url = format!("{}/api/models/{repository}", client.endpoint());
    let http = reqwest::Client::builder().build().map_err(download_io)?;
    let mut request = http
        .get(url)
        .query(&[("revision", revision), ("blobs", "true")]);
    if let Some(token) = std::env::var_os("HF_TOKEN").and_then(|value| value.into_string().ok()) {
        request = request.bearer_auth(token);
    }
    let response = request.send().await.map_err(reqwest_download_error)?;
    let status = response.status();
    if !status.is_success() {
        return Err(DownloadError {
            code: match status.as_u16() {
                401 => "authentication_required",
                403 => "forbidden",
                404 => "repository_not_found",
                429 => "rate_limited",
                _ => "upstream_http_error",
            },
            message: format!("Hugging Face repository metadata returned HTTP {status}"),
            retryable: status.as_u16() == 429 || status.is_server_error(),
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
                    code: "revision_changed",
                    message: format!("{} resolved outside pinned commit", path.display()),
                    retryable: true,
                    resumable: false,
                });
            }
            let content = if let Some(value) = metadata.xet_hash {
                ContentIdentity::Xet { value }
            } else if metadata.etag.len() == 64
                && metadata.etag.bytes().all(|byte| byte.is_ascii_hexdigit())
            {
                ContentIdentity::Sha256 {
                    value: metadata.etag.to_ascii_lowercase(),
                }
            } else {
                ContentIdentity::GitOid {
                    value: metadata.etag,
                }
            };
            Ok(ResolvedRemoteMetadata {
                size: metadata.file_size,
                content,
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
                    code: "file_not_found",
                    message: format!("Hugging Face repository has no {}", path.display()),
                    retryable: false,
                    resumable: false,
                })?;
            let (size, content) = match sibling.lfs.as_ref() {
                Some(lfs) => (
                    lfs.size,
                    ContentIdentity::Sha256 {
                        value: lfs.sha256.to_ascii_lowercase(),
                    },
                ),
                None => (
                    sibling.size.unwrap_or(0),
                    sibling
                        .blob_id
                        .as_ref()
                        .map_or(ContentIdentity::Unknown, |value| ContentIdentity::GitOid {
                            value: value.clone(),
                        }),
                ),
            };
            Ok(ResolvedRemoteMetadata { size, content })
        }
        Err(error) => Err(map_hf_error(error)),
    }
}

fn reqwest_download_error(error: reqwest::Error) -> DownloadError {
    DownloadError {
        code: "transport_failed",
        message: error.to_string(),
        retryable: error.is_timeout() || error.is_connect() || error.is_request(),
        resumable: false,
    }
}

fn map_hf_error(error: HFError) -> DownloadError {
    let (code, retryable) = match &error {
        HFError::AuthRequired { .. } => ("authentication_required", false),
        HFError::Forbidden { .. } => ("forbidden", false),
        HFError::RepoNotFound { .. } => ("repository_not_found", false),
        HFError::RevisionNotFound { .. } => ("revision_not_found", false),
        HFError::EntryNotFound { .. } => ("file_not_found", false),
        HFError::RateLimited { .. } => ("rate_limited", true),
        HFError::Request { .. } | HFError::Xet { .. } => ("transport_failed", true),
        HFError::Http { context } => {
            let retryable = context.status.as_u16() == 408 || context.status.is_server_error();
            ("upstream_http_error", retryable)
        }
        HFError::Io(_) => ("io_failed", true),
        HFError::MalformedResponse { .. } => ("malformed_upstream_response", true),
        _ => ("upstream_failed", false),
    };
    DownloadError {
        code,
        message: error.to_string(),
        retryable,
        resumable: retryable,
    }
}

fn download_io(error: impl std::fmt::Display) -> DownloadError {
    DownloadError {
        code: "io_failed",
        message: error.to_string(),
        retryable: true,
        resumable: true,
    }
}

fn relationship_component(relationship: &icn_contracts::ComponentRelationship) -> Option<&Path> {
    match relationship {
        icn_contracts::ComponentRelationship::ProjectorFor { projector, .. } => Some(projector),
        icn_contracts::ComponentRelationship::DraftFor { draft, .. } => Some(draft),
        icn_contracts::ComponentRelationship::MtpFor { mtp, .. } => Some(mtp),
    }
    .map(PathBuf::as_path)
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
                ModelDownloadEvent::Ready { .. } | ModelDownloadEvent::Failed { .. }
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
