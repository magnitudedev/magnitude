use std::collections::BTreeMap;
use std::sync::Arc;

use futures_util::future::BoxFuture;
use futures_util::stream::BoxStream;
use icn_contracts::InventoryError;
use icn_contracts::models::{
    DiscoveredModel, DiscoveredModelCatalogAttribution, DiscoveredModelState, DiscoveredModels,
    DiscoveredModelsResponse, EffectiveModel, HuggingFaceArtifactSelector, HuggingFaceRepositoryId,
    InstalledCatalogAttribution, InstalledModelPackage, ModelDomainInvalidation, ModelFileRole,
    ModelId, ModelPackageInstallationOrigin, ServingProfile,
};

use crate::inventory::InstalledPackageSnapshot;
use crate::model_domains::{ModelDomainResolver, domain_changes};
use crate::model_projection::effective_model;

const DEFAULT_EXTERNAL_CONTEXT: u32 = 100_000;

struct DiscoveryCandidate {
    package: InstalledModelPackage,
    commit: String,
    current: bool,
    modified_at: u64,
}

fn candidate_preference(candidate: &DiscoveryCandidate) -> (bool, u64, &str) {
    (candidate.current, candidate.modified_at, &candidate.commit)
}

pub(crate) fn selected_discovered_packages(
    installed: &InstalledPackageSnapshot,
) -> BTreeMap<ModelId, InstalledModelPackage> {
    let mut selected = BTreeMap::<ModelId, DiscoveryCandidate>::new();
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
        let id = ModelId::hugging_face(&repository, &selector);
        let candidate = DiscoveryCandidate {
            package: record.installed.clone(),
            commit: commit.clone(),
            current: requested_revision == "main",
            modified_at: record.model.updated_at,
        };
        match selected.entry(id) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                entry.insert(candidate);
            }
            std::collections::btree_map::Entry::Occupied(mut entry)
                if candidate_preference(&candidate) > candidate_preference(entry.get()) =>
            {
                entry.insert(candidate);
            }
            std::collections::btree_map::Entry::Occupied(_) => {}
        }
    }

    selected
        .into_iter()
        .map(|(id, candidate)| (id, candidate.package))
        .collect()
}

fn discovered_models(installed: &InstalledPackageSnapshot) -> Vec<DiscoveredModel> {
    selected_discovered_packages(installed)
        .into_iter()
        .filter_map(|(id, selected)| {
            let catalog_attribution = match &selected.catalog_attribution {
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
            let profile = discovered_profile(&selected);
            let (effective, installation) = effective_model(&selected, profile);
            let state = match effective {
                EffectiveModel::Ready { model } => DiscoveredModelState::Ready {
                    installation,
                    model,
                    catalog_attribution,
                },
                EffectiveModel::Unavailable { failure } => DiscoveredModelState::Unavailable {
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
        CatalogBaseId, CatalogVariantId, ModelCapabilities, ModelFailure, ModelFile, ModelFileId,
        ModelPackage, ModelPackageId, ModelPackageInspection, ModelPackageProperties,
        ModelPackageSource, ModelReasoningCapabilities,
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
        let selected = selected_discovered_packages(&InstalledPackageSnapshot {
            records: BTreeMap::from([stale, current]),
        });
        assert_eq!(selected.len(), 1);
        let candidate = selected.values().next().expect("one selected discovery");
        assert!(matches!(
            &candidate.package.source,
            ModelPackageSource::HuggingFace { revision, .. } if revision == "commit-b"
        ));
    }

    #[test]
    fn prefers_the_most_recent_distinct_revision_without_a_current_ref() {
        let older = discovery_record(
            "older",
            "owner/repo",
            "model.gguf",
            "commit-a",
            "commit-a",
            "package-a",
            InstalledCatalogAttribution::NotCatalogTarget,
        );
        let mut newer = discovery_record(
            "newer",
            "owner/repo",
            "model.gguf",
            "commit-b",
            "commit-b",
            "package-b",
            InstalledCatalogAttribution::NotCatalogTarget,
        );
        newer.1.model.updated_at = 2;
        let selected = selected_discovered_packages(&InstalledPackageSnapshot {
            records: BTreeMap::from([older, newer]),
        });
        assert_eq!(selected.len(), 1);
        let candidate = selected.values().next().expect("one selected discovery");
        assert!(matches!(
            &candidate.package.source,
            ModelPackageSource::HuggingFace { revision, .. } if revision == "commit-b"
        ));
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
