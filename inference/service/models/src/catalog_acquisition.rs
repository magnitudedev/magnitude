//! Acquisition of one exact catalog bundle into a managed model store, for tools that run a
//! catalog model outside ICN (evaluation). It converges the bundle's packages to present through
//! the store's own download and verification, and reports the verified files.

use std::path::{Path, PathBuf};

use futures_util::StreamExt;
use magnitude_service_contracts::models::{
    ModelFile, ModelFileRole, ModelPackage, ModelPackageSource, RecommendableModel,
    ServableModelBundle, SpeculativeDraftSource, SpeculativeMethod,
};
use magnitude_service_contracts::{
    DownloadEventStream, InventoryError, ModelDownloadEvent, ModelSource,
};

use crate::inventory::{
    ManagedModelStore, catalog_components_present, components_for_catalog_package, hf_repo_dir,
};

/// One verified file of an acquired catalog bundle.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AcquiredFile {
    /// Absolute path inside the managed store's snapshot for the package's revision.
    pub path: PathBuf,
    pub sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AcquiredDrafter {
    pub method: SpeculativeMethod,
    /// The separate draft GGUF, or the target itself when the draft is embedded.
    pub file: AcquiredFile,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AcquiredCatalogModel {
    /// The primary weights GGUF (its first shard when sharded).
    pub target: AcquiredFile,
    pub drafter: Option<AcquiredDrafter>,
    pub projector: Option<AcquiredFile>,
}

impl ManagedModelStore {
    /// Make every package of the catalog model's bundle present in this store, downloading only
    /// what is missing, and return the verified files.
    ///
    /// A package whose content-addressed blobs and snapshot links already match its exact files
    /// is reused without network access or rehashing; anything else goes through the store's
    /// ordinary package download, which verifies size and SHA-256 before publication.
    pub async fn acquire_catalog_model(
        &self,
        model: &RecommendableModel,
        progress: impl Fn(&ModelDownloadEvent) + Sync,
    ) -> Result<AcquiredCatalogModel, InventoryError> {
        let (target, draft, method) = bundle_parts(&model.configuration.bundle);
        let target_snapshot = self.acquire_package(target, &progress).await?;
        let draft_snapshot = match draft {
            Some(draft) => Some(self.acquire_package(draft, &progress).await?),
            None => None,
        };
        let target_file = acquired(&target_snapshot, primary_weights(target)?);
        let projector = target
            .files
            .iter()
            .find(|file| file.role == ModelFileRole::Projector)
            .map(|file| acquired(&target_snapshot, file));
        let drafter = match (method, draft, draft_snapshot) {
            (Some(method), Some(draft), Some(snapshot)) => Some(AcquiredDrafter {
                method: method.clone(),
                file: acquired(&snapshot, primary_weights(draft)?),
            }),
            (Some(method), None, None) => Some(AcquiredDrafter {
                method: method.clone(),
                file: target_file.clone(),
            }),
            _ => None,
        };
        Ok(AcquiredCatalogModel {
            target: target_file,
            drafter,
            projector,
        })
    }

    /// Converge one package to present and return its snapshot directory.
    async fn acquire_package(
        &self,
        package: &ModelPackage,
        progress: &(impl Fn(&ModelDownloadEvent) + Sync),
    ) -> Result<PathBuf, InventoryError> {
        let ModelPackageSource::HuggingFace {
            repository,
            revision,
        } = &package.source
        else {
            return Err(InventoryError::InvalidRequest(format!(
                "catalog package {} has no Hugging Face source",
                package.id.0
            )));
        };
        if let Some(snapshot) = self.present_snapshot(package, repository, revision)? {
            return Ok(snapshot);
        }
        let stream = self
            .start_target_downloads(vec![package.clone()])
            .await?
            .pop()
            .ok_or_else(|| {
                InventoryError::Internal("a package download admitted no operation".to_owned())
            })?;
        let commit = await_download(stream, progress).await?;
        self.present_snapshot(package, repository, &commit)?
            .ok_or_else(|| {
                InventoryError::Integrity(format!(
                    "package {} completed but its files are not present at {repository}@{commit}",
                    package.id.0
                ))
            })
    }

    fn present_snapshot(
        &self,
        package: &ModelPackage,
        repository: &str,
        commit: &str,
    ) -> Result<Option<PathBuf>, InventoryError> {
        let repository_root = self.root().join("hub").join(hf_repo_dir(repository));
        let snapshot = repository_root.join("snapshots").join(commit);
        let components = components_for_catalog_package(package)?;
        Ok(catalog_components_present(&snapshot, &repository_root, &components).then_some(snapshot))
    }
}

/// Follow one package download to its terminal event; a ready download yields the commit its
/// snapshot was published under.
async fn await_download(
    mut stream: DownloadEventStream,
    progress: &(impl Fn(&ModelDownloadEvent) + Sync),
) -> Result<String, InventoryError> {
    while let Some(event) = stream.next().await {
        progress(&event);
        match event {
            ModelDownloadEvent::Ready { model, .. } => {
                let ModelSource::HuggingFace { commit, .. } = model.source else {
                    return Err(InventoryError::Internal(
                        "a catalog package download completed from a non-Hugging Face source"
                            .to_owned(),
                    ));
                };
                return Ok(commit);
            }
            ModelDownloadEvent::Failed { error, .. } => {
                return Err(InventoryError::Upstream(format!(
                    "package download failed: {error:?}"
                )));
            }
            ModelDownloadEvent::Cancelled { .. } => {
                return Err(InventoryError::Io("package download was cancelled".to_owned()));
            }
            ModelDownloadEvent::Resolving { .. }
            | ModelDownloadEvent::CheckingSpace { .. }
            | ModelDownloadEvent::Progress { .. } => {}
        }
    }
    Err(InventoryError::Io(
        "package download ended without an outcome".to_owned(),
    ))
}

fn bundle_parts(
    bundle: &ServableModelBundle,
) -> (&ModelPackage, Option<&ModelPackage>, Option<&SpeculativeMethod>) {
    match bundle {
        ServableModelBundle::Standalone { package } => (package, None, None),
        ServableModelBundle::SpeculativeDecoding {
            target,
            draft_source,
            method,
        } => match draft_source {
            SpeculativeDraftSource::Embedded => (target, None, Some(method)),
            SpeculativeDraftSource::Separate { draft } => (target, Some(draft), Some(method)),
        },
    }
}

/// A package's primary weights file: the sole weights file, or the first shard.
fn primary_weights(package: &ModelPackage) -> Result<&ModelFile, InventoryError> {
    package
        .files
        .iter()
        .filter(|file| file.role == ModelFileRole::Weights)
        .min_by(|left, right| left.path.cmp(&right.path))
        .ok_or_else(|| {
            InventoryError::Integrity(format!("package {} has no weights", package.id.0))
        })
}

fn acquired(snapshot: &Path, file: &ModelFile) -> AcquiredFile {
    AcquiredFile {
        path: snapshot.join(&file.path),
        sha256: file.sha256.clone(),
    }
}
