use std::sync::{Arc, RwLock};

use futures_util::future::BoxFuture;
use icn_contracts::InventoryError;
use icn_contracts::models::{
    CatalogInstallationAdmission, CatalogInstallationOperation, CatalogInstallationOperationId,
    CatalogInstallationOperationState, CatalogInstallationProgress, CatalogInstallationRemoval,
    CatalogInstallationRetentionReason, CatalogInstallations, CatalogInstallationsResponse,
    CatalogPackageRemover, ModelDownload, ModelDownloadId, ModelDownloadState, ModelDownloads,
    ModelId, StartModelDownloadRequest,
};

use crate::ManagedModelDownloads;
use crate::model_domains::ModelDomainResolver;

#[derive(Debug, Clone)]
struct OperationBinding {
    operation_id: CatalogInstallationOperationId,
    model_id: ModelId,
    download_id: ModelDownloadId,
}

pub struct ManagedCatalogInstallations {
    resolver: Arc<ModelDomainResolver>,
    downloads: Arc<ManagedModelDownloads>,
    remover: Arc<dyn CatalogPackageRemover>,
    operations: RwLock<Vec<OperationBinding>>,
    mutations: tokio::sync::Mutex<()>,
}

impl ManagedCatalogInstallations {
    pub(crate) fn new(
        resolver: Arc<ModelDomainResolver>,
        downloads: Arc<ManagedModelDownloads>,
        remover: Arc<dyn CatalogPackageRemover>,
    ) -> Arc<Self> {
        Arc::new(Self {
            resolver,
            downloads,
            remover,
            operations: RwLock::new(Vec::new()),
            mutations: tokio::sync::Mutex::new(()),
        })
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
        Ok(operation_from_download(
            id.clone(),
            binding.model_id,
            download,
        ))
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
        self.operations
            .write()
            .map_err(|_| InventoryError::Internal("catalog installation lock poisoned".to_owned()))?
            .push(OperationBinding {
                operation_id: operation_id.clone(),
                model_id: id.clone(),
                download_id: download.id,
            });
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
            let binding = self.binding(&id)?;
            let download = self.downloads.cancel(&binding.download_id).await?;
            Ok(operation_from_download(id, binding.model_id, download))
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
            Ok(operation_from_download(id, binding.model_id, download))
        })
    }
}

fn operation_from_download(
    operation_id: CatalogInstallationOperationId,
    model_id: ModelId,
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
                icn_contracts::DownloadStage::Queued,
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
        ModelDownloadState::Completed => CatalogInstallationOperationState::Completed,
        ModelDownloadState::Failed {
            completed_bytes,
            total_bytes,
            failure,
            acknowledged,
        } => CatalogInstallationOperationState::Failed {
            progress: progress(
                icn_contracts::DownloadStage::Downloading,
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
                icn_contracts::DownloadStage::Queued,
                completed_bytes,
                total_bytes,
                None,
            ),
        },
    };
    CatalogInstallationOperation {
        operation_id,
        model_id,
        state,
    }
}
