use std::collections::BTreeMap;
use std::sync::Arc;

use futures_util::future::BoxFuture;
use futures_util::stream::BoxStream;
use icn_contracts::InventoryError;
use icn_contracts::models::{
    DiscoveredModel, DiscoveredModelCatalogAttribution, DiscoveredModelState, DiscoveredModels,
    DiscoveredModelsResponse, EffectiveModel, HuggingFaceArtifactSelector, HuggingFaceRepositoryId,
    InstalledCatalogAttribution, InstalledModelPackage, ModelDomainInvalidation, ModelFailure,
    ModelFileRole, ModelId, ModelPackageId, ModelPackageInstallationOrigin, ServingProfile,
};

use crate::inventory::InstalledPackageSnapshot;
use crate::model_domains::{ModelDomainResolver, domain_changes};
use crate::model_projection::effective_model;

const DEFAULT_EXTERNAL_CONTEXT: u32 = 100_000;

#[derive(Clone)]
pub(crate) struct DiscoveryCandidate {
    package: InstalledModelPackage,
    commit: String,
    current: bool,
}

pub(crate) enum SelectedDiscovery {
    Ready(DiscoveryCandidate),
    Ambiguous(ModelFailure),
}

impl SelectedDiscovery {
    pub(crate) fn ready(self) -> Option<InstalledModelPackage> {
        match self {
            Self::Ready(candidate) => Some(candidate.package),
            Self::Ambiguous(_) => None,
        }
    }
}

pub(crate) fn selected_discovered_packages(
    installed: &InstalledPackageSnapshot,
) -> BTreeMap<ModelId, SelectedDiscovery> {
    let mut candidates = BTreeMap::<ModelId, Vec<DiscoveryCandidate>>::new();
    for record in installed.records.values() {
        if record.installed.origin != ModelPackageInstallationOrigin::HuggingFaceCache {
            continue;
        }
        let icn_contracts::ModelSource::HuggingFace {
            repository,
            requested_revision,
            commit,
            ..
        } = &record.model.source
        else {
            continue;
        };
        let Some(selector) = record
            .installed
            .package
            .files
            .iter()
            .filter(|file| file.role == ModelFileRole::Weights)
            .filter_map(|file| file.path.to_str().map(str::to_owned))
            .min()
            .and_then(|path| HuggingFaceArtifactSelector::new(path).ok())
        else {
            continue;
        };
        let Ok(repository) = HuggingFaceRepositoryId::new(repository.clone()) else {
            continue;
        };
        candidates
            .entry(ModelId::hugging_face(&repository, &selector))
            .or_default()
            .push(DiscoveryCandidate {
                package: record.installed.clone(),
                commit: commit.clone(),
                current: requested_revision == "main",
            });
    }

    candidates
        .into_iter()
        .map(|(id, grouped)| {
            let mut distinct = BTreeMap::<ModelPackageId, DiscoveryCandidate>::new();
            for candidate in grouped {
                distinct
                    .entry(candidate.package.package.id.clone())
                    .and_modify(|existing| {
                        if candidate.current && !existing.current
                            || candidate.current == existing.current
                                && candidate.commit < existing.commit
                        {
                            *existing = candidate.clone();
                        }
                    })
                    .or_insert(candidate);
            }
            let current = distinct
                .values()
                .filter(|candidate| candidate.current)
                .cloned()
                .collect::<Vec<_>>();
            let selected = if current.len() == 1 {
                SelectedDiscovery::Ready(current.into_iter().next().expect("one current candidate"))
            } else if current.is_empty() && distinct.len() == 1 {
                SelectedDiscovery::Ready(distinct.into_values().next().expect("one candidate"))
            } else {
                SelectedDiscovery::Ambiguous(ModelFailure {
                    code: "ambiguous_hugging_face_artifact".to_owned(),
                    message: "Multiple installed revisions provide this Hugging Face artifact"
                        .to_owned(),
                    retryable: false,
                })
            };
            (id, selected)
        })
        .collect()
}

fn discovered_models(installed: &InstalledPackageSnapshot) -> Vec<DiscoveredModel> {
    selected_discovered_packages(installed)
        .into_iter()
        .filter_map(|(id, selected)| {
            let selected = match selected {
                SelectedDiscovery::Ready(selected) => selected,
                SelectedDiscovery::Ambiguous(failure) => {
                    return Some(DiscoveredModel {
                        id,
                        state: DiscoveredModelState::Ambiguous { failure },
                    });
                }
            };
            let catalog_attribution = match &selected.package.catalog_attribution {
                InstalledCatalogAttribution::NotCatalogTarget => {
                    DiscoveredModelCatalogAttribution::NotInCatalog
                }
                InstalledCatalogAttribution::Failed { failure } => {
                    DiscoveredModelCatalogAttribution::Failed {
                        failure: failure.clone(),
                    }
                }
                InstalledCatalogAttribution::Attributed { .. } => return None,
            };
            let profile = discovered_profile(&selected.package);
            let (effective, installation) = effective_model(&selected.package, profile);
            let state = match effective {
                EffectiveModel::Ready { model } => DiscoveredModelState::Ready {
                    selected_revision: selected.commit,
                    installation,
                    model,
                    catalog_attribution,
                },
                EffectiveModel::Unavailable { failure } => DiscoveredModelState::Unavailable {
                    selected_revision: selected.commit,
                    installation,
                    failure,
                },
            };
            Some(DiscoveredModel { id, state })
        })
        .collect()
}

pub(crate) fn discovered_profile(installed: &InstalledModelPackage) -> ServingProfile {
    ServingProfile {
        context_length: installed
            .package
            .properties
            .maximum_context_length
            .unwrap_or(DEFAULT_EXTERNAL_CONTEXT)
            .min(DEFAULT_EXTERNAL_CONTEXT),
    }
}

pub struct ManagedDiscoveredModels {
    resolver: Arc<ModelDomainResolver>,
    changes: tokio::sync::broadcast::Sender<ModelDomainInvalidation>,
}

impl ManagedDiscoveredModels {
    pub(crate) fn new(resolver: Arc<ModelDomainResolver>) -> Arc<Self> {
        let (changes, _) = tokio::sync::broadcast::channel(32);
        Arc::new(Self { resolver, changes })
    }

    pub(crate) fn installed_packages_changed(&self, revision: u64) {
        let _ = self.changes.send(ModelDomainInvalidation { revision });
    }

    fn snapshot(&self) -> Result<DiscoveredModelsResponse, InventoryError> {
        let installed = self.resolver.inventory.installed_packages_response()?;
        let records = self
            .resolver
            .inventory
            .installed_packages
            .read()
            .map_err(|_| InventoryError::Internal("installed package lock poisoned".to_owned()))?;
        Ok(DiscoveredModelsResponse {
            revision: installed.revision,
            reconciliation_complete: installed.reconciliation_complete,
            models: discovered_models(&records),
        })
    }
}

impl DiscoveredModels for ManagedDiscoveredModels {
    fn list_discovered(&self) -> BoxFuture<'_, Result<DiscoveredModelsResponse, InventoryError>> {
        Box::pin(async { self.snapshot() })
    }

    fn refresh_discovery(&self) -> BoxFuture<'_, Result<DiscoveredModelsResponse, InventoryError>> {
        Box::pin(async move {
            self.resolver.inventory.ensure_model_inventory().await?;
            self.snapshot()
        })
    }

    fn watch_discovery(&self) -> BoxStream<'static, ModelDomainInvalidation> {
        domain_changes(
            self.resolver.revision().unwrap_or_default(),
            self.changes.subscribe(),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use icn_contracts::models::{
        CatalogBaseId, CatalogVariantId, ModelCapabilities, ModelFile, ModelFileId, ModelPackage,
        ModelPackageInspection, ModelPackageProperties, ModelPackageSource,
        ModelReasoningCapabilities,
    };
    use icn_contracts::{
        ComponentRole, ContentId, ContentIdentity, Integrity, InventoryEntryId, InventoryModel,
        InventoryProperties, ModelAvailability, ModelComponent, ModelLocation, ModelSource,
    };

    use super::*;
    use crate::inventory::{InstalledPackageRecord, InstalledPackageSnapshot};

    fn discovery_record(
        occurrence: &str,
        repository: &str,
        selector: &str,
        requested_revision: &str,
        commit: &str,
        package_id: &str,
        catalog_attribution: InstalledCatalogAttribution,
    ) -> (InventoryEntryId, InstalledPackageRecord) {
        let model_file = ModelFile {
            id: ModelFileId(format!("file-{package_id}")),
            path: PathBuf::from(selector),
            role: ModelFileRole::Weights,
            size_bytes: 10,
            tensor_storage_bytes: Some(10),
            sha256: "a".repeat(64),
        };
        let package = ModelPackage {
            id: ModelPackageId(package_id.to_owned()),
            source: ModelPackageSource::HuggingFace {
                repository: repository.to_owned(),
                revision: commit.to_owned(),
            },
            files: vec![model_file],
            relationships: Vec::new(),
            properties: ModelPackageProperties {
                format: "gguf".to_owned(),
                quantization: "Q4_K_M".to_owned(),
                quantization_name: "4-bit".to_owned(),
                architecture: "test".to_owned(),
                maximum_context_length: Some(32_768),
                intrinsic_model_id: None,
                intrinsic_quality_id: None,
            },
        };
        let installed = InstalledModelPackage {
            path: PathBuf::from(format!("/cache/{occurrence}/{selector}")),
            package,
            origin: ModelPackageInstallationOrigin::HuggingFaceCache,
            inspection: ModelPackageInspection::Inspected {
                capabilities: ModelCapabilities {
                    vision: false,
                    tools: true,
                    structured_output: true,
                    reasoning: ModelReasoningCapabilities {
                        supported: false,
                        efforts: Vec::new(),
                        default_effort: None,
                    },
                },
            },
            catalog_attribution,
        };
        let id = InventoryEntryId(occurrence.to_owned());
        let component = ModelComponent {
            path: PathBuf::from(selector),
            role: ComponentRole::Weights,
            size_bytes: 10,
            content: ContentIdentity::Sha256 {
                value: "a".repeat(64),
            },
            shard_index: None,
            relationship: None,
        };
        let model = InventoryModel {
            id: id.clone(),
            content_id: ContentId(format!("content-{occurrence}")),
            created: 1,
            name: selector.to_owned(),
            supported_parameters: Vec::new(),
            availability: ModelAvailability::Available { ready_at: 1 },
            source: ModelSource::HuggingFace {
                repository: repository.to_owned(),
                requested_revision: requested_revision.to_owned(),
                commit: commit.to_owned(),
                metadata: None,
            },
            location: ModelLocation::HuggingFaceCache {
                cache_root: PathBuf::from(format!("/cache/{occurrence}")),
                repository: repository.to_owned(),
                commit: commit.to_owned(),
                components: vec![component],
                total_bytes: 10,
                integrity: Integrity::Unverified {
                    reason: "test".to_owned(),
                },
            },
            properties: InventoryProperties::Pending,
            operations: Vec::new(),
            updated_at: 1,
        };
        (id, InstalledPackageRecord { installed, model })
    }

    #[test]
    fn keeps_identical_content_from_different_repositories() {
        let first = discovery_record(
            "first",
            "owner/first",
            "model.gguf",
            "main",
            "commit-a",
            "same-package",
            InstalledCatalogAttribution::NotCatalogTarget,
        );
        let second = discovery_record(
            "second",
            "owner/second",
            "model.gguf",
            "main",
            "commit-b",
            "same-package",
            InstalledCatalogAttribution::NotCatalogTarget,
        );
        let snapshot = InstalledPackageSnapshot {
            records: BTreeMap::from([first, second]),
        };
        let ids = discovered_models(&snapshot)
            .into_iter()
            .map(|model| model.id.to_string())
            .collect::<Vec<_>>();
        assert_eq!(
            ids,
            vec![
                "hf:owner/first/model.gguf".to_owned(),
                "hf:owner/second/model.gguf".to_owned(),
            ]
        );
    }

    #[test]
    fn collapses_identical_revisions_and_prefers_current_selection() {
        let stale = discovery_record(
            "a-stale",
            "owner/repo",
            "model.gguf",
            "commit-a",
            "commit-a",
            "same-package",
            InstalledCatalogAttribution::NotCatalogTarget,
        );
        let current = discovery_record(
            "z-current",
            "owner/repo",
            "model.gguf",
            "main",
            "commit-b",
            "same-package",
            InstalledCatalogAttribution::NotCatalogTarget,
        );
        let models = discovered_models(&InstalledPackageSnapshot {
            records: BTreeMap::from([stale, current]),
        });
        let [
            DiscoveredModel {
                state:
                    DiscoveredModelState::Ready {
                        selected_revision, ..
                    },
                ..
            },
        ] = models.as_slice()
        else {
            panic!("identical occurrences must collapse to one ready discovery")
        };
        assert_eq!(selected_revision, "commit-b");
    }

    #[test]
    fn excludes_material_attributed_to_an_exact_catalog_variant() {
        let attributed = discovery_record(
            "catalog",
            "owner/repo",
            "model.gguf",
            "main",
            "commit",
            "package",
            InstalledCatalogAttribution::Attributed {
                model_id: CatalogBaseId::new("catalog").expect("base"),
                variant_id: CatalogVariantId::new("gguf:q4").expect("variant"),
            },
        );
        let snapshot = InstalledPackageSnapshot {
            records: BTreeMap::from([attributed]),
        };
        assert!(discovered_models(&snapshot).is_empty());
    }

    #[test]
    fn preserves_failed_catalog_attribution_on_a_ready_discovery() {
        let failed = discovery_record(
            "failed",
            "owner/repo",
            "model.gguf",
            "main",
            "commit",
            "package",
            InstalledCatalogAttribution::Failed {
                failure: ModelFailure {
                    code: "catalog_target_package_ambiguous".to_owned(),
                    message: "ambiguous target".to_owned(),
                    retryable: false,
                },
            },
        );
        let models = discovered_models(&InstalledPackageSnapshot {
            records: BTreeMap::from([failed]),
        });
        assert!(matches!(
            models.as_slice(),
            [DiscoveredModel {
                state: DiscoveredModelState::Ready {
                    catalog_attribution: DiscoveredModelCatalogAttribution::Failed { .. },
                    ..
                },
                ..
            }]
        ));
    }
}
