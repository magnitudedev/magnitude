use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use futures_util::StreamExt;
use futures_util::future::BoxFuture;
use getrandom::fill;
use icn_contracts::models::{
    DownloadAttempt, DownloadAttemptId, ModelDownloads, ModelDownloadsResponse, ModelFailure,
    ModelFileRelationship, ModelFileRole, ModelPackage, ModelPackageSource,
    StartModelDownloadRequest, StartModelDownloadResponse,
};
use icn_contracts::{
    ComponentRelationship, ComponentRole, DownloadComponent, DownloadModelRequest,
    HuggingFaceDownloadSource, InventoryError, ModelDownloadEvent, ModelInventory,
};
use serde::{Deserialize, Serialize};

use crate::inventory::ModelManager;

#[derive(Clone)]
pub struct ManagedModelDownloads {
    manager: Arc<ModelManager>,
    records: Arc<RwLock<BTreeMap<DownloadAttemptId, AttemptRecord>>>,
    starts: Arc<tokio::sync::Mutex<()>>,
    path: Arc<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AttemptRecord {
    attempt: DownloadAttempt,
    package: ModelPackage,
    #[serde(default)]
    sequence: u64,
}

impl ManagedModelDownloads {
    #[must_use]
    pub fn open(manager: Arc<ModelManager>) -> Self {
        let path = manager.config.root.join("download-attempts.json");
        let mut records = load_records(&path);
        for record in records.values_mut() {
            if matches!(
                record.attempt,
                DownloadAttempt::Pending { .. } | DownloadAttempt::Downloading { .. }
            ) {
                let (id, package_id) = attempt_identity(&record.attempt);
                let (completed_bytes, total_bytes) = attempt_progress(&record.attempt);
                record.attempt = DownloadAttempt::Failed {
                    id,
                    package_id,
                    completed_bytes,
                    total_bytes,
                    failure: ModelFailure {
                        code: "interrupted".to_owned(),
                        message: "download was interrupted when ICN stopped".to_owned(),
                        retryable: true,
                    },
                };
            }
        }
        persist_records(&path, &records);
        Self {
            manager,
            records: Arc::new(RwLock::new(records)),
            starts: Arc::new(tokio::sync::Mutex::new(())),
            path: Arc::new(path),
        }
    }

    fn update(&self, id: &DownloadAttemptId, attempt: DownloadAttempt) {
        let Ok(mut records) = self.records.write() else {
            return;
        };
        if let Some(record) = records.get_mut(id) {
            record.attempt = attempt;
            persist_records(&self.path, &records);
        }
    }

    async fn consume(
        self,
        id: DownloadAttemptId,
        package: ModelPackage,
        mut stream: icn_contracts::DownloadEventStream,
    ) {
        let mut completed_bytes = 0;
        let mut total_bytes = 0;
        let mut terminal = false;
        while let Some(event) = stream.next().await {
            let attempt = match event {
                ModelDownloadEvent::Resolving { .. } => DownloadAttempt::Pending {
                    id: id.clone(),
                    package_id: package.id.clone(),
                },
                ModelDownloadEvent::CheckingSpace {
                    completed_bytes,
                    total_bytes,
                    ..
                }
                | ModelDownloadEvent::Progress {
                    completed_bytes,
                    total_bytes,
                    ..
                } => DownloadAttempt::Downloading {
                    id: id.clone(),
                    package_id: package.id.clone(),
                    completed_bytes,
                    total_bytes,
                },
                ModelDownloadEvent::Ready { .. } => DownloadAttempt::Completed {
                    id: id.clone(),
                    package_id: package.id.clone(),
                },
                ModelDownloadEvent::Failed { error, .. } if error.code == "cancelled" => {
                    DownloadAttempt::Cancelled {
                        id: id.clone(),
                        package_id: package.id.clone(),
                    }
                }
                ModelDownloadEvent::Failed {
                    error,
                    completed_bytes,
                    total_bytes,
                    ..
                } => DownloadAttempt::Failed {
                    id: id.clone(),
                    package_id: package.id.clone(),
                    completed_bytes,
                    total_bytes,
                    failure: ModelFailure {
                        code: error.code,
                        message: error.message,
                        retryable: error.retryable,
                    },
                },
            };
            let is_terminal = matches!(
                attempt,
                DownloadAttempt::Completed { .. }
                    | DownloadAttempt::Failed { .. }
                    | DownloadAttempt::Cancelled { .. }
            );
            (completed_bytes, total_bytes) = attempt_progress(&attempt);
            self.update(&id, attempt);
            if is_terminal {
                terminal = true;
                break;
            }
        }
        if !terminal {
            self.update(
                &id,
                DownloadAttempt::Failed {
                    id: id.clone(),
                    package_id: package.id,
                    completed_bytes,
                    total_bytes,
                    failure: ModelFailure {
                        code: "stream_ended".to_owned(),
                        message: "download ended before reporting a terminal result".to_owned(),
                        retryable: true,
                    },
                },
            );
        }
    }
}

fn attempt_progress(attempt: &DownloadAttempt) -> (u64, u64) {
    match attempt {
        DownloadAttempt::Downloading {
            completed_bytes,
            total_bytes,
            ..
        }
        | DownloadAttempt::Failed {
            completed_bytes,
            total_bytes,
            ..
        } => (*completed_bytes, *total_bytes),
        DownloadAttempt::Pending { .. }
        | DownloadAttempt::Completed { .. }
        | DownloadAttempt::Cancelled { .. } => (0, 0),
    }
}

fn attempt_identity(
    attempt: &DownloadAttempt,
) -> (DownloadAttemptId, icn_contracts::models::ModelPackageId) {
    match attempt {
        DownloadAttempt::Pending { id, package_id }
        | DownloadAttempt::Downloading { id, package_id, .. }
        | DownloadAttempt::Completed { id, package_id }
        | DownloadAttempt::Failed { id, package_id, .. }
        | DownloadAttempt::Cancelled { id, package_id } => (id.clone(), package_id.clone()),
    }
}

fn random_attempt_id() -> Result<DownloadAttemptId, InventoryError> {
    let mut bytes = [0_u8; 16];
    fill(&mut bytes).map_err(|error| InventoryError::Internal(error.to_string()))?;
    Ok(DownloadAttemptId(format!(
        "download_{}",
        bytes
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    )))
}

fn legacy_request(package: &ModelPackage) -> Result<DownloadModelRequest, InventoryError> {
    let ModelPackageSource::HuggingFace {
        repository,
        revision,
    } = &package.source
    else {
        return Err(InventoryError::Unsupported(
            "only exact Hugging Face packages can be downloaded".to_owned(),
        ));
    };
    let shard_indices = package
        .relationships
        .iter()
        .filter_map(|relationship| match relationship {
            ModelFileRelationship::Shard { file_id, index, .. } => Some((file_id.clone(), *index)),
            _ => None,
        })
        .collect::<BTreeMap<_, _>>();
    let path_by_id = package
        .files
        .iter()
        .map(|file| (file.id.clone(), file.path.clone()))
        .collect::<BTreeMap<_, _>>();
    let components = package
        .files
        .iter()
        .map(|file| DownloadComponent {
            path: file.path.clone(),
            role: match file.role {
                ModelFileRole::Weights if shard_indices.contains_key(&file.id) => {
                    ComponentRole::Shard
                }
                ModelFileRole::Weights => ComponentRole::Weights,
                ModelFileRole::Projector => ComponentRole::Projector,
                ModelFileRole::Mtp => ComponentRole::Mtp,
                ModelFileRole::Auxiliary => ComponentRole::Auxiliary,
            },
            shard_index: shard_indices.get(&file.id).copied(),
            expected_sha256: Some(file.sha256.clone()),
        })
        .collect();
    let relationships = package
        .relationships
        .iter()
        .filter_map(|relationship| match relationship {
            ModelFileRelationship::Shard { .. } => None,
            ModelFileRelationship::ProjectorFor {
                projector_file_id,
                weights_file_id,
            } => Some(ComponentRelationship::ProjectorFor {
                projector: path_by_id.get(projector_file_id)?.clone(),
                model: path_by_id.get(weights_file_id)?.clone(),
            }),
            ModelFileRelationship::MtpFor {
                mtp_file_id,
                weights_file_id,
            } => Some(ComponentRelationship::MtpFor {
                mtp: path_by_id.get(mtp_file_id)?.clone(),
                model: path_by_id.get(weights_file_id)?.clone(),
            }),
        })
        .collect();
    Ok(DownloadModelRequest {
        source: HuggingFaceDownloadSource::HuggingFace {
            repository: repository.clone(),
            revision: revision.clone(),
        },
        components,
        relationships,
    })
}

fn load_records(path: &Path) -> BTreeMap<DownloadAttemptId, AttemptRecord> {
    let mut records = fs::read(path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Vec<AttemptRecord>>(&bytes).ok())
        .unwrap_or_default();
    for (index, record) in records.iter_mut().enumerate() {
        if record.sequence == 0 {
            record.sequence = u64::try_from(index).unwrap_or(u64::MAX).saturating_add(1);
        }
    }
    records
        .into_iter()
        .map(|record| {
            let (id, _) = attempt_identity(&record.attempt);
            (id, record)
        })
        .collect()
}

fn persist_records(path: &Path, records: &BTreeMap<DownloadAttemptId, AttemptRecord>) {
    let Some(parent) = path.parent() else {
        return;
    };
    if fs::create_dir_all(parent).is_err() {
        return;
    }
    let temporary = path.with_extension("json.tmp");
    let values = records.values().collect::<Vec<_>>();
    if serde_json::to_vec(&values)
        .ok()
        .and_then(|bytes| fs::write(&temporary, bytes).ok())
        .is_some()
    {
        let _ = fs::rename(temporary, path);
    }
}

impl ModelDownloads for ManagedModelDownloads {
    fn start(
        &self,
        request: StartModelDownloadRequest,
    ) -> BoxFuture<'_, Result<StartModelDownloadResponse, InventoryError>> {
        Box::pin(async move {
            let _start_guard = self.starts.lock().await;
            if let Some(attempt) = self
                .records
                .read()
                .map_err(|_| {
                    InventoryError::Internal("download registry lock poisoned".to_owned())
                })?
                .values()
                .find(|record| {
                    record.package.id == request.package.id
                        && matches!(
                            record.attempt,
                            DownloadAttempt::Pending { .. } | DownloadAttempt::Downloading { .. }
                        )
                })
                .map(|record| record.attempt.clone())
            {
                return Ok(StartModelDownloadResponse { attempt });
            }
            let legacy = legacy_request(&request.package)?;
            let stream = self.manager.download(legacy).await?;
            let id = random_attempt_id()?;
            let attempt = DownloadAttempt::Pending {
                id: id.clone(),
                package_id: request.package.id.clone(),
            };
            {
                let mut records = self.records.write().map_err(|_| {
                    InventoryError::Internal("download registry lock poisoned".to_owned())
                })?;
                let sequence = records
                    .values()
                    .map(|record| record.sequence)
                    .max()
                    .unwrap_or(0)
                    .saturating_add(1);
                records.insert(
                    id.clone(),
                    AttemptRecord {
                        attempt: attempt.clone(),
                        package: request.package.clone(),
                        sequence,
                    },
                );
                persist_records(&self.path, &records);
            }
            tokio::spawn(self.clone().consume(id, request.package, stream));
            Ok(StartModelDownloadResponse { attempt })
        })
    }

    fn list_attempts(&self) -> BoxFuture<'_, Result<ModelDownloadsResponse, InventoryError>> {
        Box::pin(async move {
            let records = self.records.read().map_err(|_| {
                InventoryError::Internal("download registry lock poisoned".to_owned())
            })?;
            let mut attempts = records.values().collect::<Vec<_>>();
            attempts.sort_by_key(|record| record.sequence);
            Ok(ModelDownloadsResponse {
                attempts: attempts
                    .into_iter()
                    .map(|record| record.attempt.clone())
                    .collect(),
            })
        })
    }

    fn get_attempt(
        &self,
        id: &DownloadAttemptId,
    ) -> BoxFuture<'_, Result<DownloadAttempt, InventoryError>> {
        let id = id.clone();
        Box::pin(async move {
            self.records
                .read()
                .map_err(|_| {
                    InventoryError::Internal("download registry lock poisoned".to_owned())
                })?
                .get(&id)
                .map(|record| record.attempt.clone())
                .ok_or_else(|| InventoryError::NotFound(id.0))
        })
    }

    fn cancel(
        &self,
        id: &DownloadAttemptId,
    ) -> BoxFuture<'_, Result<DownloadAttempt, InventoryError>> {
        let id = id.clone();
        Box::pin(async move {
            let record = self
                .records
                .read()
                .map_err(|_| {
                    InventoryError::Internal("download registry lock poisoned".to_owned())
                })?
                .get(&id)
                .cloned()
                .ok_or_else(|| InventoryError::NotFound(id.0.clone()))?;
            if !matches!(
                record.attempt,
                DownloadAttempt::Pending { .. } | DownloadAttempt::Downloading { .. }
            ) {
                return Ok(record.attempt);
            }
            self.manager
                .cancel_download(&legacy_request(&record.package)?)
                .await?;
            let attempt = DownloadAttempt::Cancelled {
                id: id.clone(),
                package_id: record.package.id,
            };
            self.update(&id, attempt.clone());
            Ok(attempt)
        })
    }
}
