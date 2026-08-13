use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use futures_util::StreamExt;
use futures_util::future::BoxFuture;
use getrandom::fill;
use icn_contracts::models::{
    ModelDownload, ModelDownloadId, ModelDownloadState, ModelDownloads, ModelDownloadsResponse,
    ModelPackage, ModelPackageId, ServableModelBundle, StartModelDownloadRequest,
    StartModelDownloadResponse,
};
use icn_contracts::{DownloadFailure, DownloadStage, InventoryError, ModelDownloadEvent};
use serde::{Deserialize, Serialize};

use crate::inventory::ModelManager;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
struct DownloadAttemptId(String);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "_tag", rename_all = "PascalCase")]
enum DownloadAttempt {
    #[serde(rename_all = "camelCase")]
    Pending {
        id: DownloadAttemptId,
        package_id: ModelPackageId,
    },
    #[serde(rename_all = "camelCase")]
    Downloading {
        id: DownloadAttemptId,
        package_id: ModelPackageId,
        stage: DownloadStage,
        completed_bytes: u64,
        total_bytes: u64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        bytes_per_second: Option<u64>,
    },
    #[serde(rename_all = "camelCase")]
    Completed {
        id: DownloadAttemptId,
        package_id: ModelPackageId,
    },
    #[serde(rename_all = "camelCase")]
    Failed {
        id: DownloadAttemptId,
        package_id: ModelPackageId,
        completed_bytes: u64,
        total_bytes: u64,
        failure: DownloadFailure,
    },
    #[serde(rename_all = "camelCase")]
    Cancelled {
        id: DownloadAttemptId,
        package_id: ModelPackageId,
        completed_bytes: u64,
        total_bytes: u64,
    },
}

#[derive(Clone)]
pub struct ManagedModelDownloads {
    manager: Arc<ModelManager>,
    records: Arc<RwLock<BTreeMap<DownloadAttemptId, AttemptRecord>>>,
    downloads: Arc<RwLock<BTreeMap<ModelDownloadId, DownloadRecord>>>,
    starts: Arc<tokio::sync::Mutex<()>>,
    path: Arc<PathBuf>,
    downloads_path: Arc<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AttemptRecord {
    attempt: DownloadAttempt,
    package: ModelPackage,
    #[serde(default)]
    sequence: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct DownloadRecord {
    id: ModelDownloadId,
    bundle: ServableModelBundle,
    attempt_ids: Vec<DownloadAttemptId>,
    installed_package_ids_at_admission: Vec<ModelPackageId>,
    cancelled: bool,
    failure_acknowledged: bool,
    sequence: u64,
}

impl ManagedModelDownloads {
    pub async fn open(manager: Arc<ModelManager>) -> Result<Self, InventoryError> {
        let path = manager.config.root.join("download-attempts.json");
        let downloads_path = manager.config.root.join("model-downloads.json");
        let mut records = load_records(&path);
        for record in records.values_mut() {
            let interrupted = matches!(
                record.attempt,
                DownloadAttempt::Pending { .. } | DownloadAttempt::Downloading { .. }
            );
            if interrupted {
                let (id, package_id) = attempt_identity(&record.attempt);
                let (completed_bytes, total_bytes) = attempt_progress(&record.attempt);
                record.attempt = DownloadAttempt::Failed {
                    id,
                    package_id,
                    completed_bytes,
                    total_bytes,
                    failure: DownloadFailure::Interrupted,
                };
            }
        }
        persist_records(&path, &records);
        let downloads = load_download_records(&downloads_path);
        Ok(Self {
            manager,
            records: Arc::new(RwLock::new(records)),
            downloads: Arc::new(RwLock::new(downloads)),
            starts: Arc::new(tokio::sync::Mutex::new(())),
            path: Arc::new(path),
            downloads_path: Arc::new(downloads_path),
        })
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
                } => DownloadAttempt::Downloading {
                    id: id.clone(),
                    package_id: package.id.clone(),
                    stage: DownloadStage::CheckingSpace,
                    completed_bytes,
                    total_bytes,
                    bytes_per_second: None,
                },
                ModelDownloadEvent::Progress {
                    completed_bytes,
                    total_bytes,
                    stage,
                    bytes_per_second,
                    ..
                } => DownloadAttempt::Downloading {
                    id: id.clone(),
                    package_id: package.id.clone(),
                    stage,
                    completed_bytes,
                    total_bytes,
                    bytes_per_second: bytes_per_second.map(|value| value.round() as u64),
                },
                ModelDownloadEvent::Ready { .. } => DownloadAttempt::Completed {
                    id: id.clone(),
                    package_id: package.id.clone(),
                },
                ModelDownloadEvent::Cancelled {
                    completed_bytes,
                    total_bytes,
                    ..
                } => DownloadAttempt::Cancelled {
                    id: id.clone(),
                    package_id: package.id.clone(),
                    completed_bytes,
                    total_bytes,
                },
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
                    failure: error,
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
                    failure: DownloadFailure::Interrupted,
                },
            );
        }
    }
}

fn bundle_packages(bundle: &ServableModelBundle) -> Vec<&ModelPackage> {
    match bundle {
        ServableModelBundle::Standalone { package } => vec![package],
        ServableModelBundle::SpeculativeDecoding {
            target,
            draft_source,
            ..
        } => match draft_source {
            icn_contracts::models::SpeculativeDraftSource::Embedded => vec![target],
            icn_contracts::models::SpeculativeDraftSource::Separate { draft } => {
                vec![target, draft]
            }
        },
    }
}

fn package_bytes(package: &ModelPackage) -> u64 {
    package.files.iter().map(|file| file.size_bytes).sum()
}

fn model_download(
    record: &DownloadRecord,
    attempts: &BTreeMap<DownloadAttemptId, AttemptRecord>,
) -> ModelDownload {
    let packages = bundle_packages(&record.bundle);
    let total_bytes = packages.iter().map(|package| package_bytes(package)).sum();
    let admitted = record
        .attempt_ids
        .iter()
        .filter_map(|id| attempts.get(id).map(|attempt| &attempt.attempt))
        .collect::<Vec<_>>();
    let attempted_bytes = admitted
        .iter()
        .map(|attempt| match attempt {
            DownloadAttempt::Completed { package_id, .. } => packages
                .iter()
                .find(|package| package.id == *package_id)
                .map(|package| package_bytes(package))
                .unwrap_or(0),
            DownloadAttempt::Downloading {
                completed_bytes, ..
            }
            | DownloadAttempt::Failed {
                completed_bytes, ..
            } => *completed_bytes,
            DownloadAttempt::Cancelled {
                completed_bytes, ..
            } => *completed_bytes,
            DownloadAttempt::Pending { .. } => 0,
        })
        .sum::<u64>();
    let installed_bytes = packages
        .iter()
        .filter(|package| {
            record
                .installed_package_ids_at_admission
                .contains(&package.id)
        })
        .map(|package| package_bytes(package))
        .sum::<u64>();
    let completed_bytes = installed_bytes
        .saturating_add(attempted_bytes)
        .min(total_bytes);
    let missing_attempt = admitted.len() != record.attempt_ids.len();
    let state = if record.cancelled {
        ModelDownloadState::Cancelled {
            completed_bytes,
            total_bytes,
        }
    } else if missing_attempt {
        ModelDownloadState::Failed {
            completed_bytes,
            total_bytes,
            failure: DownloadFailure::Internal {
                message: "model download references missing package-attempt state".to_owned(),
            },
            acknowledged: record.failure_acknowledged,
        }
    } else if let Some(failure) = admitted.iter().find_map(|attempt| match attempt {
        DownloadAttempt::Failed { failure, .. } => Some(failure),
        _ => None,
    }) {
        ModelDownloadState::Failed {
            completed_bytes,
            total_bytes,
            failure: failure.clone(),
            acknowledged: record.failure_acknowledged,
        }
    } else if admitted
        .iter()
        .any(|attempt| matches!(attempt, DownloadAttempt::Cancelled { .. }))
    {
        ModelDownloadState::Cancelled {
            completed_bytes,
            total_bytes,
        }
    } else if admitted
        .iter()
        .any(|attempt| matches!(attempt, DownloadAttempt::Downloading { .. }))
    {
        let active = admitted
            .iter()
            .filter_map(|attempt| match attempt {
                DownloadAttempt::Downloading {
                    stage,
                    bytes_per_second,
                    ..
                } => Some((*stage, *bytes_per_second)),
                _ => None,
            })
            .collect::<Vec<_>>();
        ModelDownloadState::Downloading {
            stage: active
                .first()
                .map(|(stage, _)| *stage)
                .unwrap_or(DownloadStage::Queued),
            completed_bytes,
            total_bytes,
            bytes_per_second: active
                .iter()
                .filter_map(|(_, rate)| *rate)
                .reduce(u64::saturating_add),
        }
    } else if admitted
        .iter()
        .any(|attempt| matches!(attempt, DownloadAttempt::Pending { .. }))
    {
        ModelDownloadState::Pending {
            completed_bytes,
            total_bytes,
        }
    } else {
        ModelDownloadState::Completed
    };
    ModelDownload {
        id: record.id.clone(),
        bundle: record.bundle.clone(),
        state,
    }
}

fn attempt_has_other_live_download(
    attempt_id: &DownloadAttemptId,
    excluded_id: &ModelDownloadId,
    downloads: &BTreeMap<ModelDownloadId, DownloadRecord>,
    attempts: &BTreeMap<DownloadAttemptId, AttemptRecord>,
) -> bool {
    downloads.values().any(|other| {
        other.id != *excluded_id
            && !other.cancelled
            && other.attempt_ids.contains(attempt_id)
            && matches!(
                model_download(other, attempts).state,
                ModelDownloadState::Pending { .. } | ModelDownloadState::Downloading { .. }
            )
    })
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
        DownloadAttempt::Cancelled {
            completed_bytes,
            total_bytes,
            ..
        } => (*completed_bytes, *total_bytes),
        DownloadAttempt::Pending { .. } | DownloadAttempt::Completed { .. } => (0, 0),
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
        | DownloadAttempt::Cancelled { id, package_id, .. } => (id.clone(), package_id.clone()),
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

fn random_download_id() -> Result<ModelDownloadId, InventoryError> {
    let mut bytes = [0_u8; 16];
    fill(&mut bytes).map_err(|error| InventoryError::Internal(error.to_string()))?;
    Ok(ModelDownloadId(format!(
        "model_download_{}",
        bytes
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>()
    )))
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

fn persist_records_result(
    path: &Path,
    records: &BTreeMap<DownloadAttemptId, AttemptRecord>,
) -> Result<(), InventoryError> {
    let parent = path.parent().ok_or_else(|| {
        InventoryError::Io("download attempt registry has no parent directory".to_owned())
    })?;
    fs::create_dir_all(parent).map_err(|error| InventoryError::Io(error.to_string()))?;
    let temporary = path.with_extension("json.tmp");
    let values = records.values().collect::<Vec<_>>();
    let bytes =
        serde_json::to_vec(&values).map_err(|error| InventoryError::Internal(error.to_string()))?;
    fs::write(&temporary, bytes).map_err(|error| InventoryError::Io(error.to_string()))?;
    fs::rename(temporary, path).map_err(|error| InventoryError::Io(error.to_string()))?;
    Ok(())
}

fn persist_records(path: &Path, records: &BTreeMap<DownloadAttemptId, AttemptRecord>) {
    let _ = persist_records_result(path, records);
}

fn load_download_records(path: &Path) -> BTreeMap<ModelDownloadId, DownloadRecord> {
    fs::read(path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Vec<DownloadRecord>>(&bytes).ok())
        .unwrap_or_default()
        .into_iter()
        .map(|record| (record.id.clone(), record))
        .collect()
}

fn persist_download_records_result(
    path: &Path,
    records: &BTreeMap<ModelDownloadId, DownloadRecord>,
) -> Result<(), InventoryError> {
    let parent = path.parent().ok_or_else(|| {
        InventoryError::Io("model download registry has no parent directory".to_owned())
    })?;
    fs::create_dir_all(parent).map_err(|error| InventoryError::Io(error.to_string()))?;
    let temporary = path.with_extension("json.tmp");
    let mut values = records.values().collect::<Vec<_>>();
    values.sort_by_key(|record| record.sequence);
    let bytes =
        serde_json::to_vec(&values).map_err(|error| InventoryError::Internal(error.to_string()))?;
    fs::write(&temporary, bytes).map_err(|error| InventoryError::Io(error.to_string()))?;
    fs::rename(temporary, path).map_err(|error| InventoryError::Io(error.to_string()))?;
    Ok(())
}

impl ModelDownloads for ManagedModelDownloads {
    fn start(
        &self,
        request: StartModelDownloadRequest,
    ) -> BoxFuture<'_, Result<StartModelDownloadResponse, InventoryError>> {
        Box::pin(async move {
            let _start_guard = self.starts.lock().await;
            let packages = bundle_packages(&request.bundle)
                .into_iter()
                .cloned()
                .collect::<Vec<_>>();
            if packages
                .iter()
                .map(|package| &package.id)
                .collect::<BTreeSet<_>>()
                .len()
                != packages.len()
            {
                return Err(InventoryError::InvalidRequest(
                    "a separate speculative draft must be distinct from its target".to_owned(),
                ));
            }
            let active = {
                let existing = self.records.read().map_err(|_| {
                    InventoryError::Internal("download registry lock poisoned".to_owned())
                })?;
                packages
                    .iter()
                    .map(|package| {
                        existing
                            .values()
                            .find(|record| {
                                record.package.id == package.id
                                    && matches!(
                                        record.attempt,
                                        DownloadAttempt::Pending { .. }
                                            | DownloadAttempt::Downloading { .. }
                                    )
                            })
                            .map(|record| record.attempt.clone())
                    })
                    .collect::<Vec<_>>()
            };
            let candidates = packages
                .iter()
                .zip(&active)
                .filter_map(|(package, attempt)| attempt.is_none().then_some(package.clone()))
                .collect::<Vec<_>>();
            let mut missing = Vec::new();
            let mut installed_package_ids_at_admission = Vec::new();
            for package in candidates {
                match self.manager.installed_package(&package.id).await {
                    Ok(_) => installed_package_ids_at_admission.push(package.id),
                    Err(InventoryError::NotFound(_)) | Err(InventoryError::NotReady(_)) => {
                        missing.push(package);
                    }
                    Err(error) => return Err(error),
                }
            }
            let new_attempts = missing
                .iter()
                .map(|package| {
                    let id = random_attempt_id()?;
                    let attempt = DownloadAttempt::Pending {
                        id: id.clone(),
                        package_id: package.id.clone(),
                    };
                    Ok((id, attempt))
                })
                .collect::<Result<Vec<_>, InventoryError>>()?;
            let streams = self.manager.start_target_downloads(missing.clone()).await?;
            if streams.len() != missing.len() {
                return Err(InventoryError::Internal(
                    "download admission returned an unexpected number of streams".to_owned(),
                ));
            }
            let mut admitted = active.into_iter().flatten().collect::<Vec<_>>();
            {
                let mut records = self.records.write().map_err(|_| {
                    InventoryError::Internal("download registry lock poisoned".to_owned())
                })?;
                let mut sequence = records
                    .values()
                    .map(|record| record.sequence)
                    .max()
                    .unwrap_or(0)
                    .saturating_add(1);
                for (package, (id, attempt)) in missing.iter().zip(&new_attempts) {
                    records.insert(
                        id.clone(),
                        AttemptRecord {
                            attempt: attempt.clone(),
                            package: package.clone(),
                            sequence,
                        },
                    );
                    sequence = sequence.saturating_add(1);
                }
                if let Err(error) = persist_records_result(&self.path, &records) {
                    for (id, _) in &new_attempts {
                        records.remove(id);
                    }
                    return Err(error);
                }
            }
            for ((package, (id, attempt)), stream) in missing
                .into_iter()
                .zip(new_attempts.into_iter())
                .zip(streams)
            {
                tokio::spawn(self.clone().consume(id, package, stream));
                admitted.push(attempt);
            }
            let download = if admitted.is_empty() {
                None
            } else {
                let id = random_download_id()?;
                let mut record = DownloadRecord {
                    id: id.clone(),
                    bundle: request.bundle,
                    attempt_ids: admitted
                        .iter()
                        .map(|attempt| attempt_identity(attempt).0)
                        .collect(),
                    installed_package_ids_at_admission,
                    cancelled: false,
                    failure_acknowledged: false,
                    sequence: 0,
                };
                let attempts = self.records.read().map_err(|_| {
                    InventoryError::Internal("download registry lock poisoned".to_owned())
                })?;
                let projected = model_download(&record, &attempts);
                drop(attempts);
                let mut downloads = self.downloads.write().map_err(|_| {
                    InventoryError::Internal("model download registry lock poisoned".to_owned())
                })?;
                record.sequence = downloads
                    .values()
                    .map(|record| record.sequence)
                    .max()
                    .unwrap_or(0)
                    .saturating_add(1);
                downloads.insert(id.clone(), record);
                if let Err(error) =
                    persist_download_records_result(&self.downloads_path, &downloads)
                {
                    downloads.remove(&id);
                    return Err(error);
                }
                Some(projected)
            };
            Ok(StartModelDownloadResponse { download })
        })
    }

    fn list(&self) -> BoxFuture<'_, Result<ModelDownloadsResponse, InventoryError>> {
        Box::pin(async move {
            let records = self.records.read().map_err(|_| {
                InventoryError::Internal("download registry lock poisoned".to_owned())
            })?;
            let downloads = self.downloads.read().map_err(|_| {
                InventoryError::Internal("model download registry lock poisoned".to_owned())
            })?;
            let mut downloads = downloads.values().collect::<Vec<_>>();
            downloads.sort_by_key(|record| record.sequence);
            Ok(ModelDownloadsResponse {
                downloads: downloads
                    .into_iter()
                    .map(|record| model_download(record, &records))
                    .collect(),
            })
        })
    }

    fn cancel(&self, id: &ModelDownloadId) -> BoxFuture<'_, Result<ModelDownload, InventoryError>> {
        let id = id.clone();
        Box::pin(async move {
            let (record, unshared_attempt_ids) = {
                let attempts = self.records.read().map_err(|_| {
                    InventoryError::Internal("download registry lock poisoned".to_owned())
                })?;
                let mut downloads = self.downloads.write().map_err(|_| {
                    InventoryError::Internal("model download registry lock poisoned".to_owned())
                })?;
                let current = downloads
                    .get(&id)
                    .cloned()
                    .ok_or_else(|| InventoryError::NotFound(id.0.clone()))?;
                if !matches!(
                    model_download(&current, &attempts).state,
                    ModelDownloadState::Pending { .. } | ModelDownloadState::Downloading { .. }
                ) {
                    (current, Vec::new())
                } else {
                    downloads
                        .get_mut(&id)
                        .expect("active model download must remain retained")
                        .cancelled = true;
                    persist_download_records_result(&self.downloads_path, &downloads)?;
                    let record = downloads
                        .get(&id)
                        .cloned()
                        .expect("cancelled model download must remain retained");
                    let unshared_attempt_ids = record
                        .attempt_ids
                        .iter()
                        .filter(|attempt_id| {
                            !attempt_has_other_live_download(attempt_id, &id, &downloads, &attempts)
                        })
                        .cloned()
                        .collect::<Vec<_>>();
                    (record, unshared_attempt_ids)
                }
            };
            for attempt_id in unshared_attempt_ids {
                let attempt = self
                    .records
                    .read()
                    .map_err(|_| {
                        InventoryError::Internal("download registry lock poisoned".to_owned())
                    })?
                    .get(&attempt_id)
                    .cloned();
                let Some(attempt) = attempt else {
                    continue;
                };
                if !matches!(
                    attempt.attempt,
                    DownloadAttempt::Pending { .. } | DownloadAttempt::Downloading { .. }
                ) {
                    continue;
                }
                match self.manager.cancel_package_download(&attempt.package).await {
                    Ok(()) => {
                        let (completed_bytes, total_bytes) = attempt_progress(&attempt.attempt);
                        self.update(
                            &attempt_id,
                            DownloadAttempt::Cancelled {
                                id: attempt_id.clone(),
                                package_id: attempt.package.id,
                                completed_bytes,
                                total_bytes,
                            },
                        )
                    }
                    // The operation may have crossed its terminal boundary after the
                    // attempt snapshot above. The model-download cancellation is already
                    // durable and remains successful in that race.
                    Err(InventoryError::NotFound(_)) => {}
                    Err(error) => return Err(error),
                }
            }
            let attempts = self.records.read().map_err(|_| {
                InventoryError::Internal("download registry lock poisoned".to_owned())
            })?;
            Ok(model_download(&record, &attempts))
        })
    }

    fn acknowledge_failure(
        &self,
        id: &ModelDownloadId,
    ) -> BoxFuture<'_, Result<ModelDownload, InventoryError>> {
        let id = id.clone();
        Box::pin(async move {
            let attempts = self.records.read().map_err(|_| {
                InventoryError::Internal("download registry lock poisoned".to_owned())
            })?;
            let mut downloads = self.downloads.write().map_err(|_| {
                InventoryError::Internal("model download registry lock poisoned".to_owned())
            })?;
            let record = downloads
                .get(&id)
                .cloned()
                .ok_or_else(|| InventoryError::NotFound(id.0.clone()))?;
            if !matches!(
                model_download(&record, &attempts).state,
                ModelDownloadState::Failed { .. }
            ) {
                return Err(InventoryError::InvalidRequest(format!(
                    "model download {} has not failed",
                    id.0
                )));
            }
            downloads
                .get_mut(&id)
                .expect("failed model download must remain retained")
                .failure_acknowledged = true;
            persist_download_records_result(&self.downloads_path, &downloads)?;
            Ok(model_download(
                downloads
                    .get(&id)
                    .expect("acknowledged model download must remain retained"),
                &attempts,
            ))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use icn_contracts::models::{
        ModelFile, ModelFileId, ModelFileRole, ModelPackageId, ModelPackageProperties,
        ModelPackageSource, SpeculativeDraftSource, SpeculativeMethod,
    };

    fn package(id: &str, size_bytes: u64) -> ModelPackage {
        ModelPackage {
            id: ModelPackageId(id.to_owned()),
            source: ModelPackageSource::HuggingFace {
                repository: "owner/repository".to_owned(),
                revision: "a".repeat(40),
            },
            files: vec![ModelFile {
                id: ModelFileId(format!("file_{}", "b".repeat(64))),
                path: PathBuf::from("model.gguf"),
                role: ModelFileRole::Weights,
                size_bytes,
                tensor_storage_bytes: None,
                sha256: "b".repeat(64),
            }],
            relationships: Vec::new(),
            properties: ModelPackageProperties {
                format: "gguf".to_owned(),
                quantization: "Q4".to_owned(),
                quantization_name: "4-bit".to_owned(),
                architecture: "test".to_owned(),
                maximum_context_length: 4096,
            },
        }
    }

    fn paired_bundle(target: ModelPackage, draft: ModelPackage) -> ServableModelBundle {
        ServableModelBundle::SpeculativeDecoding {
            target,
            draft_source: SpeculativeDraftSource::Separate { draft },
            method: SpeculativeMethod::DSpark,
        }
    }

    #[test]
    fn bundle_progress_counts_installed_packages_without_changing_download_identity() {
        let target = package("package_target", 10);
        let draft = package("package_draft", 20);
        let attempt_id = DownloadAttemptId("attempt_draft".to_owned());
        let attempts = BTreeMap::from([(
            attempt_id.clone(),
            AttemptRecord {
                attempt: DownloadAttempt::Downloading {
                    id: attempt_id.clone(),
                    package_id: draft.id.clone(),
                    stage: DownloadStage::Downloading,
                    completed_bytes: 5,
                    total_bytes: 20,
                    bytes_per_second: Some(4),
                },
                package: draft.clone(),
                sequence: 1,
            },
        )]);
        let id = ModelDownloadId("model_download_test".to_owned());
        let record = DownloadRecord {
            id: id.clone(),
            bundle: paired_bundle(target, draft),
            attempt_ids: vec![attempt_id],
            installed_package_ids_at_admission: vec![ModelPackageId("package_target".to_owned())],
            cancelled: false,
            failure_acknowledged: false,
            sequence: 1,
        };

        let projected = model_download(&record, &attempts);
        assert_eq!(projected.id, id);
        assert_eq!(
            projected.state,
            ModelDownloadState::Downloading {
                stage: DownloadStage::Downloading,
                completed_bytes: 15,
                total_bytes: 30,
                bytes_per_second: Some(4),
            }
        );
    }

    #[test]
    fn model_download_records_round_trip_with_their_stable_identity() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let path = directory.path().join("model-downloads.json");
        let id = ModelDownloadId("model_download_test".to_owned());
        let records = BTreeMap::from([(
            id.clone(),
            DownloadRecord {
                id: id.clone(),
                bundle: ServableModelBundle::Standalone {
                    package: package("package_test", 10),
                },
                attempt_ids: vec![DownloadAttemptId("attempt_test".to_owned())],
                installed_package_ids_at_admission: Vec::new(),
                cancelled: false,
                failure_acknowledged: false,
                sequence: 1,
            },
        )]);

        persist_download_records_result(&path, &records).expect("persist model downloads");
        assert_eq!(
            load_download_records(&path)
                .get(&id)
                .map(|record| &record.id),
            Some(&id)
        );
    }

    #[test]
    fn model_download_preserves_structured_package_failure() {
        let package = package("package_test", 10);
        let attempt_id = DownloadAttemptId("attempt_test".to_owned());
        let attempts = BTreeMap::from([(
            attempt_id.clone(),
            AttemptRecord {
                attempt: DownloadAttempt::Failed {
                    id: attempt_id.clone(),
                    package_id: package.id.clone(),
                    completed_bytes: 0,
                    total_bytes: 10,
                    failure: DownloadFailure::InsufficientDiskSpace {
                        required_bytes: 12,
                        available_bytes: 8,
                    },
                },
                package: package.clone(),
                sequence: 1,
            },
        )]);
        let record = DownloadRecord {
            id: ModelDownloadId("model_download_test".to_owned()),
            bundle: ServableModelBundle::Standalone { package },
            attempt_ids: vec![attempt_id],
            installed_package_ids_at_admission: Vec::new(),
            cancelled: false,
            failure_acknowledged: false,
            sequence: 1,
        };

        assert!(matches!(
            model_download(&record, &attempts).state,
            ModelDownloadState::Failed {
                failure: DownloadFailure::InsufficientDiskSpace {
                    required_bytes: 12,
                    available_bytes: 8,
                },
                acknowledged: false,
                ..
            }
        ));
    }

    #[test]
    fn missing_package_attempt_state_fails_instead_of_implying_completion() {
        let record = DownloadRecord {
            id: ModelDownloadId("model_download_test".to_owned()),
            bundle: ServableModelBundle::Standalone {
                package: package("package_test", 10),
            },
            attempt_ids: vec![DownloadAttemptId("attempt_missing".to_owned())],
            installed_package_ids_at_admission: Vec::new(),
            cancelled: false,
            failure_acknowledged: false,
            sequence: 1,
        };

        assert!(matches!(
            model_download(&record, &BTreeMap::new()).state,
            ModelDownloadState::Failed {
                failure: DownloadFailure::Internal { message },
                ..
            } if message == "model download references missing package-attempt state"
        ));
    }

    #[test]
    fn shared_package_work_is_retained_only_for_another_live_bundle_download() {
        let package = package("package_test", 10);
        let attempt_id = DownloadAttemptId("attempt_test".to_owned());
        let attempts = BTreeMap::from([(
            attempt_id.clone(),
            AttemptRecord {
                attempt: DownloadAttempt::Pending {
                    id: attempt_id.clone(),
                    package_id: package.id.clone(),
                },
                package: package.clone(),
                sequence: 1,
            },
        )]);
        let first_id = ModelDownloadId("model_download_first".to_owned());
        let second_id = ModelDownloadId("model_download_second".to_owned());
        let record = |id: ModelDownloadId| DownloadRecord {
            id,
            bundle: ServableModelBundle::Standalone {
                package: package.clone(),
            },
            attempt_ids: vec![attempt_id.clone()],
            installed_package_ids_at_admission: Vec::new(),
            cancelled: false,
            failure_acknowledged: false,
            sequence: 1,
        };
        let mut downloads = BTreeMap::from([
            (first_id.clone(), record(first_id.clone())),
            (second_id.clone(), record(second_id.clone())),
        ]);

        assert!(attempt_has_other_live_download(
            &attempt_id,
            &first_id,
            &downloads,
            &attempts,
        ));
        downloads
            .get_mut(&second_id)
            .expect("second download")
            .cancelled = true;
        assert!(!attempt_has_other_live_download(
            &attempt_id,
            &first_id,
            &downloads,
            &attempts,
        ));
    }

    #[test]
    fn cancelled_bundle_preserves_progress_reached_before_cancellation() {
        let package = package("package_test", 20);
        let attempt_id = DownloadAttemptId("attempt_test".to_owned());
        let attempts = BTreeMap::from([(
            attempt_id.clone(),
            AttemptRecord {
                attempt: DownloadAttempt::Cancelled {
                    id: attempt_id.clone(),
                    package_id: package.id.clone(),
                    completed_bytes: 7,
                    total_bytes: 20,
                },
                package: package.clone(),
                sequence: 1,
            },
        )]);
        let record = DownloadRecord {
            id: ModelDownloadId("model_download_test".to_owned()),
            bundle: ServableModelBundle::Standalone { package },
            attempt_ids: vec![attempt_id],
            installed_package_ids_at_admission: Vec::new(),
            cancelled: true,
            failure_acknowledged: false,
            sequence: 1,
        };

        assert_eq!(
            model_download(&record, &attempts).state,
            ModelDownloadState::Cancelled {
                completed_bytes: 7,
                total_bytes: 20,
            }
        );
    }
}
