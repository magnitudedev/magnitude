use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock, Weak};

use futures_util::StreamExt as _;
use futures_util::future::BoxFuture;
use futures_util::stream::BoxStream;
use magnitude_service_contracts::InventoryError;
use magnitude_service_contracts::models::{
    CatalogInstallationAdmission, CatalogInstallationOperation, CatalogInstallationOperationId,
    CatalogInstallationOperationState, CatalogInstallationProgress, CatalogInstallationRemoval,
    CatalogInstallationRetentionReason, CatalogInstallations, CatalogInstallationsInvalidation,
    CatalogInstallationsResponse, CatalogModelOptimizer, CatalogOptimizationProgress,
    CatalogPackageRemover, ModelDownload, ModelDownloadId, ModelDownloadState, ModelDownloads,
    ModelId, StartModelDownloadRequest,
};

use crate::ManagedModelDownloads;
use crate::catalog_models::reject_unavailable;
use crate::model_domains::ModelDomainResolver;

#[derive(Debug, Clone)]
struct OperationBinding {
    operation_id: CatalogInstallationOperationId,
    model_id: ModelId,
    download_id: ModelDownloadId,
    optimization: Option<CatalogOptimizationProgress>,
}

/// Catalog installations: a download, then kernel tuning for the installed model. Each admitted
/// installation has one job that follows its download and, once the download is verified and
/// published, runs the optimizer. The operation is `Optimizing` while that runs.
pub struct ManagedCatalogInstallations {
    this: Weak<Self>,
    resolver: Arc<ModelDomainResolver>,
    downloads: Arc<ManagedModelDownloads>,
    remover: Arc<dyn CatalogPackageRemover>,
    optimizer: Arc<dyn CatalogModelOptimizer>,
    operations: RwLock<Vec<OperationBinding>>,
    jobs: Mutex<HashMap<CatalogInstallationOperationId, tokio::task::JoinHandle<()>>>,
    mutations: tokio::sync::Mutex<()>,
    revision: Arc<AtomicU64>,
    changes: tokio::sync::broadcast::Sender<CatalogInstallationsInvalidation>,
}

impl ManagedCatalogInstallations {
    pub(crate) fn new(
        resolver: Arc<ModelDomainResolver>,
        downloads: Arc<ManagedModelDownloads>,
        remover: Arc<dyn CatalogPackageRemover>,
        optimizer: Arc<dyn CatalogModelOptimizer>,
    ) -> Arc<Self> {
        let (changes, _) = tokio::sync::broadcast::channel(64);
        Arc::new_cyclic(|this| Self {
            this: this.clone(),
            resolver,
            downloads,
            remover,
            optimizer,
            operations: RwLock::new(Vec::new()),
            jobs: Mutex::new(HashMap::new()),
            mutations: tokio::sync::Mutex::new(()),
            revision: Arc::new(AtomicU64::new(0)),
            changes,
        })
    }

    fn invalidate(&self) {
        let revision = self.revision.fetch_add(1, Ordering::AcqRel).saturating_add(1);
        let _ = self.changes.send(CatalogInstallationsInvalidation { revision });
    }

    fn set_optimization(
        &self,
        id: &CatalogInstallationOperationId,
        optimization: Option<CatalogOptimizationProgress>,
    ) {
        let Ok(mut operations) = self.operations.write() else {
            return;
        };
        if let Some(binding) = operations.iter_mut().find(|binding| binding.operation_id == *id) {
            if binding.optimization == optimization {
                return;
            }
            binding.optimization = optimization;
            drop(operations);
            self.invalidate();
        }
    }

    async fn run(self: Arc<Self>, id: CatalogInstallationOperationId, binding: OperationBinding) {
        self.follow(&id, binding).await;
        self.jobs
            .lock()
            .expect("catalog installation job lock")
            .remove(&id);
        self.set_optimization(&id, None);
    }

    /// Follow the download to its end; a completed download is optimized.
    async fn follow(self: &Arc<Self>, id: &CatalogInstallationOperationId, binding: OperationBinding) {
        let mut changes = self.downloads.watch();
        loop {
            let state = match self.downloads.list().await {
                Ok(response) => response
                    .downloads
                    .into_iter()
                    .find(|download| download.id == binding.download_id)
                    .map(|download| download.state),
                Err(error) => {
                    tracing::warn!(operation_id = %id.0, %error, "catalog installation could not read its download");
                    None
                }
            };
            match state {
                Some(ModelDownloadState::Completed) => break,
                Some(ModelDownloadState::Pending { .. } | ModelDownloadState::Downloading { .. }) => {
                    if changes.next().await.is_none() {
                        return;
                    }
                }
                _ => return,
            }
        }
        self.set_optimization(id, Some(CatalogOptimizationProgress::preparing(None)));
        let this = Arc::downgrade(self);
        let progress_id = id.clone();
        self.optimizer
            .optimize_catalog_model(
                binding.model_id,
                Box::new(move |progress| {
                    if let Some(installations) = this.upgrade() {
                        installations.set_optimization(&progress_id, Some(progress));
                    }
                }),
            )
            .await;
    }

    /// Stop the operation's optimization, if it is optimizing, and wait until it has stopped.
    /// Returns whether it was optimizing.
    async fn stop_optimization(&self, id: &CatalogInstallationOperationId) -> Result<bool, InventoryError> {
        if self.binding(id)?.optimization.is_none() {
            return Ok(false);
        }
        let job = self
            .jobs
            .lock()
            .expect("catalog installation job lock")
            .remove(id);
        if let Some(job) = job {
            job.abort();
            let _ = job.await;
        }
        self.set_optimization(id, None);
        Ok(true)
    }

    fn binding(
        &self,
        id: &CatalogInstallationOperationId,
    ) -> Result<OperationBinding, InventoryError> {
        self.operations
            .read()
            .map_err(|_| InventoryError::Internal("catalog installation lock poisoned".to_owned()))?
            .iter()
            .find(|binding| binding.operation_id == *id)
            .cloned()
            .ok_or_else(|| InventoryError::NotFound(id.0.clone()))
    }

    async fn operation(
        &self,
        id: &CatalogInstallationOperationId,
    ) -> Result<CatalogInstallationOperation, InventoryError> {
        let binding = self.binding(id)?;
        let download = self
            .downloads
            .list()
            .await?
            .downloads
            .into_iter()
            .find(|download| download.id == binding.download_id)
            .ok_or_else(|| InventoryError::NotFound(id.0.clone()))?;
        Ok(operation_from_download(id.clone(), binding, download))
    }

    pub(crate) async fn cleanup_model(&self, model_id: &ModelId) -> Result<(), InventoryError> {
        let package_ids = self.resolver.catalog_cleanup_package_ids(model_id)?;
        if !package_ids.is_empty() {
            self.remover.remove_catalog_packages(package_ids).await?;
        }
        Ok(())
    }

    pub(crate) async fn install(
        &self,
        id: &ModelId,
    ) -> Result<CatalogInstallationAdmission, InventoryError> {
        let _mutation = self.mutations.lock().await;
        let definition = self.resolver.catalog_definition(id)?;
        reject_unavailable(id, definition)?;
        let existing_operation_ids = self
            .operations
            .read()
            .map_err(|_| InventoryError::Internal("catalog installation lock poisoned".to_owned()))?
            .iter()
            .filter(|binding| binding.model_id == *id)
            .map(|binding| binding.operation_id.clone())
            .collect::<Vec<_>>();
        for operation_id in existing_operation_ids {
            let operation = self.operation(&operation_id).await?;
            if matches!(
                operation.state,
                CatalogInstallationOperationState::Pending { .. }
                    | CatalogInstallationOperationState::Running { .. }
                    | CatalogInstallationOperationState::Optimizing { .. }
            ) {
                return Err(InventoryError::ModelOperation {
                    code: "catalog_installation_active".to_owned(),
                    message: format!("catalog model {id} already has an active installation"),
                    retryable: true,
                });
            }
        }
        let started = self
            .downloads
            .start(StartModelDownloadRequest {
                bundle: definition.configuration.bundle.clone(),
            })
            .await?;
        let Some(download) = started.download else {
            self.cleanup_model(id).await?;
            return Ok(CatalogInstallationAdmission::Current);
        };
        let operation_id = CatalogInstallationOperationId(download.id.0.clone());
        let binding = OperationBinding {
            operation_id: operation_id.clone(),
            model_id: id.clone(),
            download_id: download.id,
            optimization: None,
        };
        self.operations
            .write()
            .map_err(|_| InventoryError::Internal("catalog installation lock poisoned".to_owned()))?
            .push(binding.clone());
        let this = self.this.upgrade().expect("catalog installations are alive while in use");
        // Held across the spawn, so a job that ends at once removes its own entry after it.
        let mut jobs = self.jobs.lock().expect("catalog installation job lock");
        let job = tokio::spawn(this.run(operation_id.clone(), binding));
        jobs.insert(operation_id.clone(), job);
        drop(jobs);
        Ok(CatalogInstallationAdmission::Admitted { operation_id })
    }

    pub(crate) async fn remove(
        &self,
        id: &ModelId,
    ) -> Result<CatalogInstallationRemoval, InventoryError> {
        let _mutation = self.mutations.lock().await;
        self.resolver.catalog_definition(id)?;
        let operation_ids = self
            .operations
            .read()
            .map_err(|_| InventoryError::Internal("catalog installation lock poisoned".to_owned()))?
            .iter()
            .filter(|binding| binding.model_id == *id)
            .map(|binding| binding.operation_id.clone())
            .collect::<Vec<_>>();
        for operation_id in operation_ids {
            // An installed model's optimization yields to its removal.
            self.stop_optimization(&operation_id).await?;
            let operation = self.operation(&operation_id).await?;
            if matches!(
                operation.state,
                CatalogInstallationOperationState::Pending { .. }
                    | CatalogInstallationOperationState::Running { .. }
            ) {
                return Err(InventoryError::ModelOperation {
                    code: "catalog_installation_active".to_owned(),
                    message: format!(
                        "catalog model {id} has an active installation; cancel it before removal"
                    ),
                    retryable: false,
                });
            }
        }
        let plan = self.resolver.catalog_removal_plan(id)?;
        if !plan.installed {
            return Err(InventoryError::ModelOperation {
                code: "catalog_model_not_installed".to_owned(),
                message: format!("catalog model {id} is not installed"),
                retryable: false,
            });
        }
        if plan.externally_owned {
            return Ok(CatalogInstallationRemoval::Retained {
                reason: CatalogInstallationRetentionReason::ExternalOwnership,
            });
        }
        if plan.shared {
            return Ok(CatalogInstallationRemoval::Retained {
                reason: CatalogInstallationRetentionReason::SharedMaterial,
            });
        }
        let reclaimed_bytes = self
            .remover
            .remove_catalog_packages(plan.package_ids)
            .await?;
        Ok(CatalogInstallationRemoval::Removed { reclaimed_bytes })
    }
}

impl CatalogInstallations for ManagedCatalogInstallations {
    /// The current revision, then every change to an operation's download or optimization.
    fn watch_catalog_installations(&self) -> BoxStream<'static, CatalogInstallationsInvalidation> {
        let revision = Arc::clone(&self.revision);
        let downloads = self.downloads.watch().map(move |_| CatalogInstallationsInvalidation {
            revision: revision.fetch_add(1, Ordering::AcqRel).saturating_add(1),
        });
        let optimizations =
            futures_util::stream::unfold(self.changes.subscribe(), |mut receiver| async move {
                loop {
                    match receiver.recv().await {
                        Ok(event) => return Some((event, receiver)),
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                        Err(tokio::sync::broadcast::error::RecvError::Closed) => return None,
                    }
                }
            });
        Box::pin(futures_util::stream::select(downloads, optimizations))
    }

    fn list_catalog_installations(
        &self,
    ) -> BoxFuture<'_, Result<CatalogInstallationsResponse, InventoryError>> {
        Box::pin(async move {
            let ids = self
                .operations
                .read()
                .map_err(|_| {
                    InventoryError::Internal("catalog installation lock poisoned".to_owned())
                })?
                .iter()
                .map(|binding| binding.operation_id.clone())
                .collect::<Vec<_>>();
            let mut operations = Vec::with_capacity(ids.len());
            for id in ids {
                operations.push(self.operation(&id).await?);
            }
            Ok(CatalogInstallationsResponse { operations })
        })
    }

    fn cancel_catalog_installation(
        &self,
        id: &CatalogInstallationOperationId,
    ) -> BoxFuture<'_, Result<CatalogInstallationOperation, InventoryError>> {
        let id = id.clone();
        Box::pin(async move {
            // Cancelling an optimization leaves the model installed; its next load tunes.
            if self.stop_optimization(&id).await? {
                return self.operation(&id).await;
            }
            let binding = self.binding(&id)?;
            let download = self.downloads.cancel(&binding.download_id).await?;
            Ok(operation_from_download(id, binding, download))
        })
    }

    fn acknowledge_catalog_installation_failure(
        &self,
        id: &CatalogInstallationOperationId,
    ) -> BoxFuture<'_, Result<CatalogInstallationOperation, InventoryError>> {
        let id = id.clone();
        Box::pin(async move {
            let binding = self.binding(&id)?;
            let download = self
                .downloads
                .acknowledge_failure(&binding.download_id)
                .await?;
            Ok(operation_from_download(id, binding, download))
        })
    }
}

fn operation_from_download(
    operation_id: CatalogInstallationOperationId,
    binding: OperationBinding,
    download: ModelDownload,
) -> CatalogInstallationOperation {
    let progress =
        |stage, completed_bytes, total_bytes, bytes_per_second| CatalogInstallationProgress {
            stage,
            completed_bytes,
            total_bytes,
            bytes_per_second,
        };
    let state = match download.state {
        ModelDownloadState::Pending {
            completed_bytes,
            total_bytes,
        } => CatalogInstallationOperationState::Pending {
            progress: progress(
                magnitude_service_contracts::DownloadStage::Queued,
                completed_bytes,
                total_bytes,
                None,
            ),
        },
        ModelDownloadState::Downloading {
            stage,
            completed_bytes,
            total_bytes,
            bytes_per_second,
        } => CatalogInstallationOperationState::Running {
            progress: progress(stage, completed_bytes, total_bytes, bytes_per_second),
        },
        ModelDownloadState::Completed => match binding.optimization {
            Some(progress) => CatalogInstallationOperationState::Optimizing { progress },
            None => CatalogInstallationOperationState::Completed,
        },
        ModelDownloadState::Failed {
            completed_bytes,
            total_bytes,
            failure,
            acknowledged,
        } => CatalogInstallationOperationState::Failed {
            progress: progress(
                magnitude_service_contracts::DownloadStage::Downloading,
                completed_bytes,
                total_bytes,
                None,
            ),
            failure,
            acknowledged,
        },
        ModelDownloadState::Cancelled {
            completed_bytes,
            total_bytes,
        } => CatalogInstallationOperationState::Cancelled {
            progress: progress(
                magnitude_service_contracts::DownloadStage::Queued,
                completed_bytes,
                total_bytes,
                None,
            ),
        },
    };
    CatalogInstallationOperation {
        operation_id,
        model_id: binding.model_id,
        state,
    }
}
