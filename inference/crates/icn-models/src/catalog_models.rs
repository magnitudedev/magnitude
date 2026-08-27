use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use futures_util::future::BoxFuture;
use icn_contracts::InventoryError;
use icn_contracts::models::{
    CatalogModelEffectiveConfiguration, CatalogModelInstallation, CatalogModelLocalState,
    CatalogModelUpdateState, CatalogModels, CatalogPackageRemover, CatalogPackageRole,
    InferenceModel, InstallCatalogModelRequest, InstallModelResponse, InstalledCatalogAttribution,
    InstalledModelPackage, ModelDownloads, ModelFailure, ModelPackageId, ModelServingConfiguration,
    RecommendableModel, RecommendableModelCatalog, ServableModelBundle, StartModelDownloadRequest,
};

use crate::ManagedModelDownloads;
use crate::inventory::{
    InstalledPackagesObserver, ManagedModelStore, catalog_packages, catalog_target,
};

#[derive(Clone)]
pub struct CatalogModelResolver {
    inventory: Arc<ManagedModelStore>,
    catalog: RecommendableModelCatalog,
}

impl CatalogModelResolver {
    #[must_use]
    pub fn new(inventory: Arc<ManagedModelStore>, catalog: RecommendableModelCatalog) -> Arc<Self> {
        Arc::new(Self { inventory, catalog })
    }

    pub fn serving_configuration(
        &self,
        canonical_model_id: &str,
    ) -> Result<ModelServingConfiguration, InventoryError> {
        let model = self
            .snapshot()?
            .models
            .into_iter()
            .find(|model| model.id == canonical_model_id)
            .ok_or_else(|| {
                InventoryError::NotFound(format!("model {canonical_model_id} is not available"))
            })?;
        match model.local_state {
            CatalogModelLocalState::Installed {
                installation:
                    CatalogModelInstallation {
                        effective_configuration:
                            CatalogModelEffectiveConfiguration::Runnable { configuration },
                        ..
                    },
                ..
            } => Ok(configuration),
            CatalogModelLocalState::Installed {
                installation:
                    CatalogModelInstallation {
                        effective_configuration:
                            CatalogModelEffectiveConfiguration::Unavailable { failure },
                        ..
                    },
                ..
            } => Err(InventoryError::ModelOperation {
                code: failure.code,
                message: failure.message,
                retryable: failure.retryable,
            }),
            CatalogModelLocalState::NotInstalled => Ok(model.desired_configuration),
        }
    }

    fn snapshot(&self) -> Result<icn_contracts::models::ModelsResponse, InventoryError> {
        let installed = self.inventory.installed_packages_response()?;
        let present = installed
            .packages
            .iter()
            .map(|entry| (entry.package.id.clone(), entry))
            .collect::<BTreeMap<_, _>>();
        let affiliations = self
            .inventory
            .catalog_affiliations
            .read()
            .map_err(|_| InventoryError::Internal("catalog affiliation lock poisoned".to_owned()))?
            .entries()
            .cloned()
            .collect::<Vec<_>>();

        let catalog_models = self
            .catalog
            .models
            .iter()
            .map(|model| catalog_model(model, &present, &affiliations))
            .collect::<Vec<_>>();
        Ok(icn_contracts::models::ModelsResponse {
            revision: installed.revision,
            reconciliation_complete: installed.reconciliation_complete,
            models: catalog_models,
            diagnostics: self.catalog.diagnostics.clone(),
        })
    }
}

#[derive(Clone)]
pub struct ManagedCatalogModels {
    resolver: Arc<CatalogModelResolver>,
    downloads: Arc<ManagedModelDownloads>,
    remover: Arc<dyn CatalogPackageRemover>,
    maintenance_generation: Arc<AtomicU64>,
    maintenance_running: Arc<AtomicBool>,
}

impl ManagedCatalogModels {
    #[must_use]
    pub fn new(
        resolver: Arc<CatalogModelResolver>,
        downloads: Arc<ManagedModelDownloads>,
        remover: Arc<dyn CatalogPackageRemover>,
    ) -> Result<Arc<Self>, InventoryError> {
        let service = Arc::new(Self {
            resolver,
            downloads,
            remover,
            maintenance_generation: Arc::new(AtomicU64::new(0)),
            maintenance_running: Arc::new(AtomicBool::new(false)),
        });
        let observer: Arc<dyn InstalledPackagesObserver> = service.clone();
        service
            .resolver
            .inventory
            .set_installed_packages_observer(Arc::downgrade(&observer))?;
        service.request_maintenance();
        Ok(service)
    }

    fn snapshot(&self) -> Result<icn_contracts::models::ModelsResponse, InventoryError> {
        self.resolver.snapshot()
    }

    async fn cleanup(&self, model: &InferenceModel) -> Result<(), InventoryError> {
        for package_id in superseded_packages_ready_for_removal(model) {
            self.remover
                .remove_catalog_package(package_id.clone())
                .await?;
        }
        Ok(())
    }

    fn request_maintenance(&self) {
        self.maintenance_generation.fetch_add(1, Ordering::AcqRel);
        if self
            .maintenance_running
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }
        let service = self.clone();
        tokio::spawn(async move { service.run_maintenance().await });
    }

    async fn run_maintenance(self) {
        loop {
            let observed = self.maintenance_generation.load(Ordering::Acquire);
            match self.snapshot() {
                Ok(snapshot) => {
                    for model in &snapshot.models {
                        if let Err(error) = self.cleanup(model).await {
                            tracing::warn!(
                                model_id = %model.model_id.0,
                                variant_id = %model.variant_id.0,
                                %error,
                                "catalog package cleanup failed"
                            );
                        }
                    }
                }
                Err(error) => tracing::warn!(%error, "catalog package maintenance snapshot failed"),
            }

            if self.maintenance_generation.load(Ordering::Acquire) != observed {
                continue;
            }
            self.maintenance_running.store(false, Ordering::Release);
            if self.maintenance_generation.load(Ordering::Acquire) == observed
                || self
                    .maintenance_running
                    .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                    .is_err()
            {
                return;
            }
        }
    }
}

fn superseded_packages_ready_for_removal(model: &InferenceModel) -> &[ModelPackageId] {
    let CatalogModelLocalState::Installed {
        update_state:
            CatalogModelUpdateState::Available {
                missing_package_ids,
                superseded_package_ids,
            },
        ..
    } = &model.local_state
    else {
        return &[];
    };
    if missing_package_ids.is_empty() {
        superseded_package_ids
    } else {
        &[]
    }
}

impl InstalledPackagesObserver for ManagedCatalogModels {
    fn installed_packages_changed(&self) {
        self.request_maintenance();
    }
}

fn catalog_model(
    definition: &RecommendableModel,
    present: &BTreeMap<ModelPackageId, &InstalledModelPackage>,
    affiliations: &[icn_contracts::models::CatalogPackageAffiliation],
) -> InferenceModel {
    let desired_ids = catalog_packages(definition)
        .map(|(package, _)| package.id.clone())
        .collect::<BTreeSet<_>>();
    let missing_desired_package_ids = desired_ids
        .iter()
        .filter(|package_id| !present.contains_key(*package_id))
        .cloned()
        .collect::<Vec<_>>();
    let superseded_package_ids = affiliations
        .iter()
        .filter(|affiliation| {
            affiliation.model_id == definition.model_id
                && affiliation.variant_id == definition.variant_id
                && present.contains_key(&affiliation.package_id)
                && !desired_ids.contains(&affiliation.package_id)
        })
        .map(|affiliation| affiliation.package_id.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let attributed_targets = present
        .values()
        .filter(|entry| {
            matches!(
                &entry.catalog_attribution,
                InstalledCatalogAttribution::Attributed { model_id, variant_id }
                    if model_id == &definition.model_id && variant_id == &definition.variant_id
            ) || affiliations.iter().any(|affiliation| {
                affiliation.model_id == definition.model_id
                    && affiliation.variant_id == definition.variant_id
                    && affiliation.package_id == entry.package.id
                    && affiliation.role == CatalogPackageRole::Target
            })
        })
        .copied()
        .collect::<Vec<_>>();
    let present_packages = present
        .values()
        .filter(|entry| {
            matches!(
                &entry.catalog_attribution,
                InstalledCatalogAttribution::Attributed { model_id, variant_id }
                    if model_id == &definition.model_id && variant_id == &definition.variant_id
            ) || affiliations.iter().any(|affiliation| {
                affiliation.model_id == definition.model_id
                    && affiliation.variant_id == definition.variant_id
                    && affiliation.package_id == entry.package.id
            })
        })
        .copied()
        .cloned()
        .collect::<Vec<_>>();

    let local_state = if attributed_targets.is_empty() {
        CatalogModelLocalState::NotInstalled
    } else {
        let effective_configuration = if missing_desired_package_ids.is_empty() {
            CatalogModelEffectiveConfiguration::Runnable {
                configuration: definition.configuration.clone(),
            }
        } else {
            let desired_target_id = &catalog_target(definition).id;
            let fallback = attributed_targets
                .iter()
                .find(|entry| entry.package.id == *desired_target_id)
                .copied()
                .or_else(|| (attributed_targets.len() == 1).then_some(attributed_targets[0]));
            match fallback {
                Some(entry) => {
                    let bundle = ServableModelBundle::Standalone {
                        package: entry.package.clone(),
                    };
                    let profile = definition.configuration.profile.clone();
                    CatalogModelEffectiveConfiguration::Runnable {
                        configuration: ModelServingConfiguration {
                            bundle,
                            profile,
                        },
                    }
                }
                None => CatalogModelEffectiveConfiguration::Unavailable {
                    failure: ModelFailure {
                        code: "catalog_installed_targets_ambiguous".to_owned(),
                        message: "Multiple superseded catalog targets are installed and no current target is present"
                            .to_owned(),
                        retryable: true,
                    },
                },
            }
        };
        let update_state =
            if missing_desired_package_ids.is_empty() && superseded_package_ids.is_empty() {
                CatalogModelUpdateState::Current
            } else {
                CatalogModelUpdateState::Available {
                    missing_package_ids: missing_desired_package_ids,
                    superseded_package_ids,
                }
            };
        CatalogModelLocalState::Installed {
            installation: CatalogModelInstallation {
                effective_configuration,
                packages: present_packages,
            },
            update_state,
        }
    };

    InferenceModel {
        id: format!("{}:{}", definition.model_id.0, definition.variant_id.0),
        model_id: definition.model_id.clone(),
        variant_id: definition.variant_id.clone(),
        desired_configuration: definition.configuration.clone(),
        display_name: definition.display_name.clone(),
        variant_label: definition.variant_label.clone(),
        description: definition.description.clone(),
        release_date: definition.release_date.clone(),
        license: definition.license.clone(),
        capabilities: definition.capabilities.clone(),
        parameterization: definition.parameterization.clone(),
        intelligence: definition.intelligence.clone(),
        fidelity_rank: definition.fidelity_rank,
        quantization_aware: definition.quantization_aware,
        local_state,
    }
}

impl CatalogModels for ManagedCatalogModels {
    fn list(&self) -> BoxFuture<'_, Result<icn_contracts::models::ModelsResponse, InventoryError>> {
        Box::pin(async { self.snapshot() })
    }

    fn install(
        &self,
        request: InstallCatalogModelRequest,
    ) -> BoxFuture<'_, Result<InstallModelResponse, InventoryError>> {
        Box::pin(async move {
            let model = self
                .snapshot()?
                .models
                .into_iter()
                .find(|model| {
                    model.model_id == request.model_id && model.variant_id == request.variant_id
                })
                .ok_or_else(|| {
                    InventoryError::NotFound(format!(
                        "catalog model {} variant {}",
                        request.model_id.0, request.variant_id.0
                    ))
                })?;
            let response = self
                .downloads
                .start(StartModelDownloadRequest {
                    bundle: model.desired_configuration.bundle.clone(),
                })
                .await?;
            if let Some(download) = response.download {
                return Ok(InstallModelResponse::DownloadAdmitted {
                    download_id: download.id,
                });
            }
            self.cleanup(&model).await?;
            Ok(InstallModelResponse::Current)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    use icn_contracts::models::{CatalogIntelligence, IntelligenceProvenance};

    use icn_contracts::models::{
        CatalogModelId, CatalogPackageAffiliation, CatalogPackageRole, CatalogVariantId,
        ModelCapabilities, ModelPackage, ModelPackageInspection, ModelPackageInstallationOrigin,
        ModelPackageProperties, ModelPackageSource, ModelReasoningCapabilities, ModelReleaseDate,
        ServingProfile,
    };

    fn package(id: &str) -> ModelPackage {
        ModelPackage {
            id: ModelPackageId(id.to_owned()),
            source: ModelPackageSource::Local {
                path: PathBuf::from(format!("/{id}.gguf")),
            },
            files: Vec::new(),
            relationships: Vec::new(),
            properties: ModelPackageProperties {
                format: "gguf".to_owned(),
                quantization: "Q4".to_owned(),
                quantization_name: "4-bit".to_owned(),
                architecture: "test".to_owned(),
                maximum_context_length: Some(4_096),
                intrinsic_model_id: Some("catalog".to_owned()),
                intrinsic_quality_id: Some("Q4".to_owned()),
            },
        }
    }

    fn definition(desired: ModelPackage) -> RecommendableModel {
        RecommendableModel {
            model_id: CatalogModelId("catalog".to_owned()),
            variant_id: CatalogVariantId("gguf:q4".to_owned()),
            configuration: ModelServingConfiguration {
                bundle: ServableModelBundle::Standalone { package: desired },
                profile: ServingProfile {
                    context_length: 4_096,
                },
            },
            display_name: "Catalog".to_owned(),
            variant_label: "Q4".to_owned(),
            description: String::new(),
            release_date: ModelReleaseDate::new("2026-01-01").expect("valid test date"),
            license: "test".to_owned(),
            capabilities: ModelCapabilities {
                vision: false,
                tools: false,
                structured_output: false,
                reasoning: ModelReasoningCapabilities {
                    supported: false,
                    efforts: Vec::new(),
                    default_effort: None,
                },
            },
            parameterization: icn_contracts::models::ModelParameterization::Dense {
                total_parameters: 8_000_000_000,
            },
            intelligence: CatalogIntelligence {
                score: 1.0,
                provenance: IntelligenceProvenance::ArtificialAnalysisIntelligenceIndex {
                    methodology_version: "test".to_owned(),
                    as_of_date: "2026-01-01".to_owned(),
                    url: "https://example.com/model".to_owned(),
                },
            },
            fidelity_rank: 1,
            quantization_aware: false,
        }
    }

    fn installed(model_package: ModelPackage) -> InstalledModelPackage {
        InstalledModelPackage {
            path: PathBuf::from(format!("/installed/{}", model_package.id.0)),
            package: model_package,
            origin: ModelPackageInstallationOrigin::Magnitude,
            inspection: ModelPackageInspection::Pending,
            catalog_attribution: InstalledCatalogAttribution::Attributed {
                model_id: CatalogModelId("catalog".to_owned()),
                variant_id: CatalogVariantId("gguf:q4".to_owned()),
            },
        }
    }

    #[test]
    fn attributed_prior_target_remains_visible_and_runnable_when_desired_artifact_is_missing() {
        let desired = package("desired");
        let mut prior = installed(package("prior"));
        prior.catalog_attribution = InstalledCatalogAttribution::NotCatalogTarget;
        let present = BTreeMap::from([(prior.package.id.clone(), &prior)]);
        let affiliations = vec![CatalogPackageAffiliation {
            model_id: CatalogModelId("catalog".to_owned()),
            variant_id: CatalogVariantId("gguf:q4".to_owned()),
            package_id: prior.package.id.clone(),
            repository: "owner/prior".to_owned(),
            role: CatalogPackageRole::Target,
        }];

        let projected = catalog_model(&definition(desired.clone()), &present, &affiliations);
        assert_eq!(projected.release_date.as_str(), "2026-01-01");
        assert!(superseded_packages_ready_for_removal(&projected).is_empty());
        let CatalogModelLocalState::Installed {
            installation,
            update_state,
        } = projected.local_state
        else {
            panic!("attributed target must be installed");
        };
        let CatalogModelUpdateState::Available {
            missing_package_ids,
            superseded_package_ids,
        } = update_state
        else {
            panic!("prior target must expose an available update");
        };
        assert_eq!(missing_package_ids, vec![desired.id]);
        assert_eq!(superseded_package_ids, vec![prior.package.id.clone()]);
        assert_eq!(installation.packages, vec![prior.clone()]);
        let CatalogModelEffectiveConfiguration::Runnable { configuration } =
            installation.effective_configuration
        else {
            panic!("unique prior target must remain runnable");
        };
        let ServableModelBundle::Standalone { package } = configuration.bundle else {
            panic!("prior target fallback must be standalone");
        };
        assert_eq!(package.id, prior.package.id);
    }

    #[test]
    fn superseded_packages_become_removable_only_after_every_desired_package_is_present() {
        let desired = installed(package("desired"));
        let prior = installed(package("prior"));
        let present = BTreeMap::from([
            (desired.package.id.clone(), &desired),
            (prior.package.id.clone(), &prior),
        ]);
        let affiliations = vec![CatalogPackageAffiliation {
            model_id: CatalogModelId("catalog".to_owned()),
            variant_id: CatalogVariantId("gguf:q4".to_owned()),
            package_id: prior.package.id.clone(),
            repository: "owner/prior".to_owned(),
            role: CatalogPackageRole::Target,
        }];

        let projected = catalog_model(
            &definition(desired.package.clone()),
            &present,
            &affiliations,
        );

        assert_eq!(
            superseded_packages_ready_for_removal(&projected),
            std::slice::from_ref(&prior.package.id),
        );
    }
}
