use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Weak};

use futures_util::future::BoxFuture;
use futures_util::stream::BoxStream;
use icn_contracts::InventoryError;
use icn_contracts::models::{
    CatalogInstallationAdmission, CatalogInstallationRemoval, CatalogModel, CatalogModelState,
    CatalogModelUpdate, CatalogModels, CatalogModelsResponse, CatalogPackageRole, EffectiveModel,
    InstalledCatalogAttribution, InstalledModelPackage, ModelAssessmentSubject,
    ModelDomainInvalidation, ModelFailure, ModelId, ModelInstallation, ModelInstallationOwnership,
    ModelPackageId, ModelPackageInstallationOrigin, ModelServingConfiguration, ParsedModelId,
    RecommendableModel, ServableModelBundle,
};

use crate::catalog_installations::ManagedCatalogInstallations;
use crate::discovered_models::{
    SelectedDiscovery, discovered_profile, selected_discovered_packages,
};
use crate::inventory::{InstalledPackagesObserver, catalog_packages, catalog_target};
use crate::model_domains::{ModelDomainResolver, domain_changes};
use crate::model_projection::{
    bundle_packages, effective_configuration_model, effective_model, primary_model_path,
    ready_model,
};

pub struct CatalogRemovalPlan {
    pub package_ids: Vec<ModelPackageId>,
    pub installed: bool,
    pub externally_owned: bool,
    pub shared: bool,
}

fn removal_plan(
    ids: &BTreeSet<ModelPackageId>,
    target_ids: &BTreeSet<ModelPackageId>,
    external_ids: &BTreeSet<ModelPackageId>,
    shared_ids: &BTreeSet<ModelPackageId>,
) -> CatalogRemovalPlan {
    CatalogRemovalPlan {
        package_ids: if target_ids.is_empty() {
            Vec::new()
        } else {
            ids.difference(external_ids)
                .filter(|package_id| !shared_ids.contains(*package_id))
                .cloned()
                .collect()
        },
        installed: !target_ids.is_empty(),
        externally_owned: !target_ids.is_disjoint(external_ids),
        shared: !target_ids.is_disjoint(shared_ids),
    }
}

impl ModelDomainResolver {
    pub fn serving_configuration(
        &self,
        model_id: &ModelId,
    ) -> Result<ModelServingConfiguration, InventoryError> {
        match model_id.parsed() {
            ParsedModelId::Catalog { .. } => self.effective_catalog_configuration(model_id),
            ParsedModelId::HuggingFace { .. } => {
                let installed = self.inventory.installed_packages.read().map_err(|_| {
                    InventoryError::Internal("installed package lock poisoned".to_owned())
                })?;
                let selected = selected_discovered_packages(&installed)
                    .remove(model_id)
                    .and_then(SelectedDiscovery::ready)
                    .ok_or_else(|| InventoryError::NotFound(model_id.to_string()))?;
                if matches!(
                    selected.catalog_attribution,
                    InstalledCatalogAttribution::Attributed { .. }
                ) {
                    return Err(InventoryError::NotFound(model_id.to_string()));
                }
                let profile = discovered_profile(&selected);
                match effective_model(&selected, profile.clone()).0 {
                    EffectiveModel::Ready { .. } => Ok(ModelServingConfiguration {
                        bundle: ServableModelBundle::Standalone {
                            package: selected.package,
                        },
                        profile,
                    }),
                    EffectiveModel::Unavailable { failure } => {
                        Err(InventoryError::ModelOperation {
                            code: failure.code,
                            message: failure.message,
                            retryable: failure.retryable,
                        })
                    }
                }
            }
        }
    }

    pub fn assessment_configuration(
        &self,
        subject: &ModelAssessmentSubject,
    ) -> Result<ModelServingConfiguration, InventoryError> {
        match subject {
            ModelAssessmentSubject::Catalog {
                model_id,
                selection,
            } => match selection {
                icn_contracts::models::CatalogModelSelection::Desired => {
                    Ok(self.catalog_definition(model_id)?.configuration.clone())
                }
                icn_contracts::models::CatalogModelSelection::Effective => {
                    self.effective_catalog_configuration(model_id)
                }
            },
            ModelAssessmentSubject::Discovery { model_id } => self.serving_configuration(model_id),
        }
    }

    fn effective_catalog_configuration(
        &self,
        model_id: &ModelId,
    ) -> Result<ModelServingConfiguration, InventoryError> {
        let definition = self.catalog_definition(model_id)?;
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
        match catalog_resolution(definition, &present, &affiliations).state {
            CatalogMaterialState::NotInstalled => Err(InventoryError::NotReady(format!(
                "catalog model {model_id} is not installed"
            ))),
            CatalogMaterialState::Installed {
                configuration: Some(configuration),
                effective: EffectiveModel::Ready { .. },
                ..
            } => Ok(configuration),
            CatalogMaterialState::Installed {
                effective: EffectiveModel::Unavailable { failure },
                ..
            } => Err(InventoryError::ModelOperation {
                code: failure.code,
                message: failure.message,
                retryable: failure.retryable,
            }),
            CatalogMaterialState::Installed {
                configuration: None,
                effective: EffectiveModel::Ready { .. },
                ..
            } => Err(InventoryError::Internal(
                "ready catalog material has no serving configuration".to_owned(),
            )),
        }
    }

    pub fn revision(&self) -> Result<u64, InventoryError> {
        Ok(self.inventory.installed_packages_response()?.revision)
    }

    pub fn catalog_snapshot(&self) -> Result<CatalogModelsResponse, InventoryError> {
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
        let occurrence_origins = self
            .inventory
            .installed_packages
            .read()
            .map_err(|_| InventoryError::Internal("installed package lock poisoned".to_owned()))?
            .records
            .values()
            .fold(
                BTreeMap::<ModelPackageId, InstallationOrigins>::new(),
                |mut origins, record| {
                    origins
                        .entry(record.installed.package.id.clone())
                        .or_default()
                        .include(record.installed.origin);
                    origins
                },
            );
        Ok(CatalogModelsResponse {
            revision: installed.revision,
            reconciliation_complete: installed.reconciliation_complete,
            models: self
                .catalog
                .models
                .iter()
                .map(|definition| {
                    catalog_model(definition, &present, &affiliations, &occurrence_origins)
                })
                .collect(),
        })
    }

    pub(crate) fn catalog_definition(
        &self,
        id: &ModelId,
    ) -> Result<&RecommendableModel, InventoryError> {
        self.catalog
            .models
            .iter()
            .find(|definition| catalog_id(definition) == *id)
            .ok_or_else(|| InventoryError::NotFound(id.to_string()))
    }

    pub fn catalog_removal_plan(&self, id: &ModelId) -> Result<CatalogRemovalPlan, InventoryError> {
        let definition = self.catalog_definition(id)?;
        let installed = self.inventory.installed_packages_response()?;
        let affiliations = self
            .inventory
            .catalog_affiliations
            .read()
            .map_err(|_| InventoryError::Internal("catalog affiliation lock poisoned".to_owned()))?
            .entries()
            .cloned()
            .collect::<Vec<_>>();
        let ids = installed
            .packages
            .iter()
            .filter(|entry| {
                catalog_packages(definition).any(|(package, _)| package.id == entry.package.id)
                    || affiliations.iter().any(|affiliation| {
                        affiliation.model_id == definition.model_id
                            && affiliation.variant_id == definition.variant_id
                            && affiliation.package_id == entry.package.id
                    })
            })
            .map(|entry| entry.package.id.clone())
            .collect::<BTreeSet<_>>();
        let occurrences =
            self.inventory.installed_packages.read().map_err(|_| {
                InventoryError::Internal("installed package lock poisoned".to_owned())
            })?;
        let external_ids = occurrences
            .records
            .values()
            .filter(|record| {
                ids.contains(&record.installed.package.id)
                    && record.installed.origin == ModelPackageInstallationOrigin::HuggingFaceCache
            })
            .map(|record| record.installed.package.id.clone())
            .collect::<BTreeSet<_>>();
        let shared_ids = ids
            .iter()
            .filter(|package_id| {
                affiliations.iter().any(|affiliation| {
                    affiliation.package_id == **package_id
                        && (affiliation.model_id != definition.model_id
                            || affiliation.variant_id != definition.variant_id)
                }) || self.catalog.models.iter().any(|candidate| {
                    catalog_id(candidate) != *id
                        && catalog_packages(candidate)
                            .any(|(package, _)| package.id == **package_id)
                })
            })
            .cloned()
            .collect::<BTreeSet<_>>();
        let target_ids = ids
            .iter()
            .filter(|package_id| {
                **package_id == catalog_target(definition).id
                    || affiliations.iter().any(|affiliation| {
                        affiliation.model_id == definition.model_id
                            && affiliation.variant_id == definition.variant_id
                            && affiliation.package_id == **package_id
                            && affiliation.role == CatalogPackageRole::Target
                    })
            })
            .cloned()
            .collect::<BTreeSet<_>>();
        Ok(removal_plan(&ids, &target_ids, &external_ids, &shared_ids))
    }

    pub(crate) fn catalog_cleanup_package_ids(
        &self,
        id: &ModelId,
    ) -> Result<Vec<ModelPackageId>, InventoryError> {
        let definition = self.catalog_definition(id)?;
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
        let resolution = catalog_resolution(definition, &present, &affiliations);
        if !resolution.missing_desired_package_ids.is_empty() {
            return Ok(Vec::new());
        }
        Ok(resolution
            .superseded_package_ids
            .into_iter()
            .filter(|package_id| {
                present
                    .get(package_id)
                    .is_some_and(|entry| entry.origin == ModelPackageInstallationOrigin::Magnitude)
            })
            .filter(|package_id| {
                !self.catalog.models.iter().any(|candidate| {
                    catalog_id(candidate) != *id
                        && catalog_packages(candidate).any(|(package, _)| package.id == *package_id)
                })
            })
            .filter(|package_id| {
                !affiliations.iter().any(|affiliation| {
                    affiliation.package_id == *package_id
                        && (affiliation.model_id != definition.model_id
                            || affiliation.variant_id != definition.variant_id)
                })
            })
            .collect())
    }
}

#[derive(Clone)]
pub struct ManagedCatalogModels {
    resolver: Arc<ModelDomainResolver>,
    installations: Arc<ManagedCatalogInstallations>,
    maintenance_generation: Arc<AtomicU64>,
    maintenance_running: Arc<AtomicBool>,
    changes: tokio::sync::broadcast::Sender<ModelDomainInvalidation>,
    inventory_observer: Arc<dyn InstalledPackagesObserver>,
}

impl ManagedCatalogModels {
    pub(crate) fn new(
        resolver: Arc<ModelDomainResolver>,
        installations: Arc<ManagedCatalogInstallations>,
        make_inventory_observer: impl FnOnce(Weak<Self>) -> Arc<dyn InstalledPackagesObserver>,
    ) -> Arc<Self> {
        let (changes, _) = tokio::sync::broadcast::channel(32);
        Arc::new_cyclic(|catalog| Self {
            resolver,
            installations,
            maintenance_generation: Arc::new(AtomicU64::new(0)),
            maintenance_running: Arc::new(AtomicBool::new(false)),
            changes,
            inventory_observer: make_inventory_observer(catalog.clone()),
        })
    }

    pub(crate) fn installed_packages_changed(&self, revision: u64) {
        let _ = self.changes.send(ModelDomainInvalidation { revision });
        self.request_maintenance();
    }

    pub(crate) fn resolver(&self) -> &Arc<ModelDomainResolver> {
        &self.resolver
    }

    pub(crate) fn inventory_observer(&self) -> &Arc<dyn InstalledPackagesObserver> {
        &self.inventory_observer
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
            for definition in &self.resolver.catalog.models {
                let model_id = catalog_id(definition);
                if let Err(error) = self.installations.cleanup_model(&model_id).await {
                    tracing::warn!(%model_id, %error, "catalog package cleanup failed");
                }
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

impl CatalogModels for ManagedCatalogModels {
    fn list_catalog(&self) -> BoxFuture<'_, Result<CatalogModelsResponse, InventoryError>> {
        Box::pin(async { self.resolver.catalog_snapshot() })
    }

    fn install_catalog_model(
        &self,
        id: &ModelId,
    ) -> BoxFuture<'_, Result<CatalogInstallationAdmission, InventoryError>> {
        let id = id.clone();
        Box::pin(async move { self.installations.install(&id).await })
    }

    fn remove_catalog_model_installation(
        &self,
        id: &ModelId,
    ) -> BoxFuture<'_, Result<CatalogInstallationRemoval, InventoryError>> {
        let id = id.clone();
        Box::pin(async move { self.installations.remove(&id).await })
    }

    fn watch_catalog(&self) -> BoxStream<'static, ModelDomainInvalidation> {
        domain_changes(
            self.resolver.revision().unwrap_or_default(),
            self.changes.subscribe(),
        )
    }
}

fn catalog_model(
    definition: &RecommendableModel,
    present: &BTreeMap<ModelPackageId, &InstalledModelPackage>,
    affiliations: &[icn_contracts::models::CatalogPackageAffiliation],
    occurrence_origins: &BTreeMap<ModelPackageId, InstallationOrigins>,
) -> CatalogModel {
    let resolution = catalog_resolution(definition, present, affiliations);
    CatalogModel {
        id: catalog_id(definition),
        desired: ready_model(
            &definition.configuration.bundle,
            definition.configuration.profile.clone(),
            definition.capabilities.clone(),
        ),
        display_name: definition.display_name.clone(),
        variant_label: definition.variant_label.clone(),
        description: definition.description.clone(),
        release_date: definition.release_date.clone(),
        license: definition.license.clone(),
        source_urls: bundle_packages(&definition.configuration.bundle)
            .filter_map(|package| match &package.source {
                icn_contracts::models::ModelPackageSource::HuggingFace { repository, .. } => {
                    Some(format!("https://huggingface.co/{repository}"))
                }
                icn_contracts::models::ModelPackageSource::Local { .. } => None,
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect(),
        parameterization: definition.parameterization.clone(),
        intelligence: definition.intelligence.clone(),
        fidelity_rank: definition.fidelity_rank,
        quantization_aware: definition.quantization_aware,
        local_state: match resolution.state {
            CatalogMaterialState::NotInstalled => CatalogModelState::NotInstalled,
            CatalogMaterialState::Installed {
                selected_target,
                effective,
                ..
            } => CatalogModelState::Installed {
                effective,
                installation: installation(
                    &resolution.installed_packages,
                    selected_target,
                    occurrence_origins,
                ),
                update_state: if resolution.required_download_bytes == 0 {
                    CatalogModelUpdate::Current
                } else {
                    CatalogModelUpdate::Available {
                        required_download_bytes: resolution.required_download_bytes,
                    }
                },
            },
        },
    }
}

struct CatalogResolution<'a> {
    installed_packages: Vec<&'a InstalledModelPackage>,
    missing_desired_package_ids: Vec<ModelPackageId>,
    superseded_package_ids: Vec<ModelPackageId>,
    required_download_bytes: u64,
    state: CatalogMaterialState<'a>,
}

#[derive(Clone, Copy, Default)]
struct InstallationOrigins {
    magnitude: bool,
    external: bool,
}

impl InstallationOrigins {
    fn include(&mut self, origin: ModelPackageInstallationOrigin) {
        match origin {
            ModelPackageInstallationOrigin::Magnitude => self.magnitude = true,
            ModelPackageInstallationOrigin::HuggingFaceCache => self.external = true,
        }
    }
}

enum CatalogMaterialState<'a> {
    NotInstalled,
    Installed {
        selected_target: Option<&'a InstalledModelPackage>,
        configuration: Option<ModelServingConfiguration>,
        effective: EffectiveModel,
    },
}

fn catalog_resolution<'a>(
    definition: &RecommendableModel,
    present: &BTreeMap<ModelPackageId, &'a InstalledModelPackage>,
    affiliations: &[icn_contracts::models::CatalogPackageAffiliation],
) -> CatalogResolution<'a> {
    let desired_packages = catalog_packages(definition).collect::<Vec<_>>();
    let desired_ids = desired_packages
        .iter()
        .map(|(package, _)| package.id.clone())
        .collect::<BTreeSet<_>>();
    let missing_desired_package_ids = desired_ids
        .iter()
        .filter(|package_id| !present.contains_key(*package_id))
        .cloned()
        .collect::<Vec<_>>();
    let required_download_bytes = desired_packages
        .iter()
        .filter(|(package, _)| !present.contains_key(&package.id))
        .flat_map(|(package, _)| &package.files)
        .map(|file| file.size_bytes)
        .sum();
    let superseded_package_ids = affiliations
        .iter()
        .filter(|affiliation| {
            affiliation.model_id == definition.model_id
                && affiliation.variant_id == definition.variant_id
                && !desired_ids.contains(&affiliation.package_id)
                && present.contains_key(&affiliation.package_id)
        })
        .map(|affiliation| affiliation.package_id.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let installed_packages = present
        .values()
        .filter(|entry| {
            if missing_desired_package_ids.is_empty() {
                return desired_ids.contains(&entry.package.id);
            }
            desired_ids.contains(&entry.package.id)
                || matches!(
                    &entry.catalog_attribution,
                    InstalledCatalogAttribution::Attributed { model_id, variant_id }
                        if model_id == &definition.model_id && variant_id == &definition.variant_id
                )
                || affiliations.iter().any(|affiliation| {
                    affiliation.model_id == definition.model_id
                        && affiliation.variant_id == definition.variant_id
                        && affiliation.package_id == entry.package.id
                })
        })
        .copied()
        .collect::<Vec<_>>();
    let targets = present
        .values()
        .filter(|entry| {
            entry.package.id == catalog_target(definition).id
                || matches!(
                    &entry.catalog_attribution,
                    InstalledCatalogAttribution::Attributed { model_id, variant_id }
                        if model_id == &definition.model_id && variant_id == &definition.variant_id
                )
                || affiliations.iter().any(|affiliation| {
                    affiliation.model_id == definition.model_id
                        && affiliation.variant_id == definition.variant_id
                        && affiliation.package_id == entry.package.id
                        && affiliation.role == CatalogPackageRole::Target
                })
        })
        .copied()
        .collect::<Vec<_>>();
    let desired_target = targets
        .iter()
        .find(|entry| entry.package.id == catalog_target(definition).id)
        .copied();
    let selected_target = desired_target.or(match targets.as_slice() {
        [target] => Some(*target),
        _ => None,
    });
    let state = if targets.is_empty() {
        CatalogMaterialState::NotInstalled
    } else if missing_desired_package_ids.is_empty() {
        let configuration = definition.configuration.clone();
        CatalogMaterialState::Installed {
            selected_target: desired_target,
            effective: effective_configuration_model(&configuration, present),
            configuration: Some(configuration),
        }
    } else if let Some(target) = selected_target {
        let configuration = ModelServingConfiguration {
            bundle: ServableModelBundle::Standalone {
                package: target.package.clone(),
            },
            profile: definition.configuration.profile.clone(),
        };
        CatalogMaterialState::Installed {
            selected_target: Some(target),
            effective: effective_configuration_model(&configuration, present),
            configuration: Some(configuration),
        }
    } else {
        CatalogMaterialState::Installed {
            selected_target: None,
            configuration: None,
            effective: EffectiveModel::Unavailable {
                failure: ModelFailure {
                    code: "catalog_installed_targets_ambiguous".to_owned(),
                    message: "Multiple superseded catalog targets are installed and no current target is present"
                        .to_owned(),
                    retryable: true,
                },
            },
        }
    };
    CatalogResolution {
        installed_packages,
        missing_desired_package_ids,
        superseded_package_ids,
        required_download_bytes,
        state,
    }
}

fn catalog_id(definition: &RecommendableModel) -> ModelId {
    ModelId::catalog(&definition.model_id, &definition.variant_id)
}

fn installation(
    installed: &[&InstalledModelPackage],
    selected_target: Option<&InstalledModelPackage>,
    occurrence_origins: &BTreeMap<ModelPackageId, InstallationOrigins>,
) -> ModelInstallation {
    let installed_bytes = installed
        .iter()
        .map(|entry| {
            entry
                .package
                .files
                .iter()
                .map(|file| file.size_bytes)
                .sum::<u64>()
        })
        .sum();
    let origins = installed
        .iter()
        .fold(InstallationOrigins::default(), |mut origins, entry| {
            if let Some(occurrences) = occurrence_origins.get(&entry.package.id) {
                origins.magnitude |= occurrences.magnitude;
                origins.external |= occurrences.external;
            } else {
                origins.include(entry.origin);
            }
            origins
        });
    let ownership = match (origins.magnitude, origins.external) {
        (true, false) => ModelInstallationOwnership::Magnitude,
        (false, true) => ModelInstallationOwnership::ExternalHuggingFace,
        (true, true) => ModelInstallationOwnership::Mixed,
        (false, false) => unreachable!("an installation contains at least one package"),
    };
    match selected_target {
        Some(target) => ModelInstallation::Resolved {
            installed_bytes,
            primary_path: primary_model_path(target),
            ownership,
        },
        None => ModelInstallation::Unresolved {
            installed_bytes,
            ownership,
        },
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use icn_contracts::models::{
        CatalogBaseId, CatalogIntelligence, CatalogPackageAffiliation, CatalogVariantId,
        IntelligenceProvenance, ModelCapabilities, ModelFile, ModelFileId, ModelFileRole,
        ModelPackage, ModelPackageInspection, ModelPackageProperties, ModelPackageSource,
        ModelParameterization, ModelReasoningCapabilities, ModelReleaseDate,
        ResolvedModelInstallation, ServingProfile, SpeculativeDraftSource, SpeculativeMethod,
    };

    use super::*;
    use crate::model_projection::resolved_installation;

    fn capabilities() -> ModelCapabilities {
        ModelCapabilities {
            vision: false,
            tools: true,
            structured_output: true,
            reasoning: ModelReasoningCapabilities {
                supported: false,
                efforts: Vec::new(),
                default_effort: None,
            },
        }
    }

    fn package(id: &str, bytes: u64) -> ModelPackage {
        ModelPackage {
            id: ModelPackageId(id.to_owned()),
            source: ModelPackageSource::Local {
                path: PathBuf::from(format!("/{id}.gguf")),
            },
            files: vec![ModelFile {
                id: ModelFileId(format!("file-{id}")),
                path: PathBuf::from(format!("{id}.gguf")),
                role: ModelFileRole::Weights,
                size_bytes: bytes,
                tensor_storage_bytes: Some(bytes),
                sha256: "a".repeat(64),
            }],
            relationships: Vec::new(),
            properties: ModelPackageProperties {
                format: "gguf".to_owned(),
                quantization: "Q4_K_M".to_owned(),
                quantization_name: "4-bit".to_owned(),
                architecture: "test".to_owned(),
                maximum_context_length: Some(32_768),
                intrinsic_model_id: Some("catalog".to_owned()),
                intrinsic_quality_id: Some("Q4".to_owned()),
            },
        }
    }

    fn installed(
        package: ModelPackage,
        origin: ModelPackageInstallationOrigin,
    ) -> InstalledModelPackage {
        InstalledModelPackage {
            path: PathBuf::from(format!("/installed/{}", package.id.0)),
            package,
            origin,
            inspection: ModelPackageInspection::Inspected {
                capabilities: capabilities(),
            },
            catalog_attribution: InstalledCatalogAttribution::NotCatalogTarget,
        }
    }

    fn definition(bundle: ServableModelBundle) -> RecommendableModel {
        RecommendableModel {
            model_id: CatalogBaseId::new("catalog").expect("catalog base"),
            variant_id: CatalogVariantId::new("gguf:q4").expect("catalog variant"),
            configuration: ModelServingConfiguration {
                bundle,
                profile: ServingProfile {
                    context_length: 32_768,
                },
            },
            display_name: "Catalog".to_owned(),
            variant_label: "Q4".to_owned(),
            description: "Catalog model".to_owned(),
            release_date: ModelReleaseDate::new("2026-01-01").expect("release date"),
            license: "test".to_owned(),
            capabilities: capabilities(),
            parameterization: ModelParameterization::Dense {
                total_parameters: 1_000_000,
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

    fn affiliation(
        package_id: &ModelPackageId,
        role: CatalogPackageRole,
    ) -> CatalogPackageAffiliation {
        CatalogPackageAffiliation {
            model_id: CatalogBaseId::new("catalog").expect("catalog base"),
            variant_id: CatalogVariantId::new("gguf:q4").expect("catalog variant"),
            package_id: package_id.clone(),
            repository: "owner/repo".to_owned(),
            role,
        }
    }

    #[test]
    fn removal_retains_protected_dependencies_without_blocking_a_managed_target() {
        let target = ModelPackageId("target".to_owned());
        let external_dependency = ModelPackageId("external-dependency".to_owned());
        let shared_dependency = ModelPackageId("shared-dependency".to_owned());
        let ids = BTreeSet::from([
            target.clone(),
            external_dependency.clone(),
            shared_dependency.clone(),
        ]);
        let plan = removal_plan(
            &ids,
            &BTreeSet::from([target.clone()]),
            &BTreeSet::from([external_dependency]),
            &BTreeSet::from([shared_dependency]),
        );
        assert!(plan.installed);
        assert!(!plan.externally_owned);
        assert!(!plan.shared);
        assert_eq!(plan.package_ids, vec![target]);
    }

    #[test]
    fn removal_is_retained_when_the_target_itself_is_external_or_shared() {
        let target = ModelPackageId("target".to_owned());
        let ids = BTreeSet::from([target.clone()]);
        let external = removal_plan(
            &ids,
            &ids,
            &BTreeSet::from([target.clone()]),
            &BTreeSet::new(),
        );
        assert!(external.externally_owned);
        let shared = removal_plan(&ids, &ids, &BTreeSet::new(), &BTreeSet::from([target]));
        assert!(shared.shared);
    }

    #[test]
    fn removal_does_not_treat_dependency_only_material_as_an_installation() {
        let dependency = ModelPackageId("dependency".to_owned());
        let plan = removal_plan(
            &BTreeSet::from([dependency]),
            &BTreeSet::new(),
            &BTreeSet::new(),
            &BTreeSet::new(),
        );
        assert!(!plan.installed);
        assert!(plan.package_ids.is_empty());
    }

    #[test]
    fn ready_model_exposes_serving_facts_without_exposing_bundle_structure() {
        let target = package("target", 20);
        let draft = package("draft", 5);
        let ready = ready_model(
            &ServableModelBundle::SpeculativeDecoding {
                target,
                draft_source: SpeculativeDraftSource::Separate { draft },
                method: SpeculativeMethod::DFlash,
            },
            ServingProfile {
                context_length: 32_768,
            },
            capabilities(),
        );
        assert_eq!(ready.metadata.storage_bytes, 25);
        assert_eq!(ready.speculative_method, Some(SpeculativeMethod::DFlash));
    }

    #[test]
    fn resolved_installation_points_to_the_primary_model_file() {
        let temporary = tempfile::tempdir().expect("temporary directory");
        let mut installed = installed(
            package("model", 20),
            ModelPackageInstallationOrigin::HuggingFaceCache,
        );
        installed.path = temporary.path().to_path_buf();
        let ResolvedModelInstallation::Resolved { primary_path, .. } =
            resolved_installation(&installed);
        assert_eq!(primary_path, temporary.path().join("model.gguf"));
    }

    #[test]
    fn catalog_resolution_with_no_installed_target_is_not_installed() {
        let definition = definition(ServableModelBundle::Standalone {
            package: package("target", 20),
        });
        let resolution = catalog_resolution(&definition, &BTreeMap::new(), &[]);
        assert!(matches!(
            resolution.state,
            CatalogMaterialState::NotInstalled
        ));
    }

    #[test]
    fn prior_target_remains_ready_while_desired_material_is_missing() {
        let desired = package("desired", 20);
        let prior = installed(
            package("prior", 10),
            ModelPackageInstallationOrigin::Magnitude,
        );
        let present = BTreeMap::from([(prior.package.id.clone(), &prior)]);
        let model = catalog_model(
            &definition(ServableModelBundle::Standalone { package: desired }),
            &present,
            &[affiliation(&prior.package.id, CatalogPackageRole::Target)],
            &BTreeMap::new(),
        );
        let CatalogModelState::Installed {
            effective: EffectiveModel::Ready { .. },
            installation:
                ModelInstallation::Resolved {
                    installed_bytes, ..
                },
            update_state:
                CatalogModelUpdate::Available {
                    required_download_bytes,
                },
        } = model.local_state
        else {
            panic!("the unique prior target must remain a ready installed fallback")
        };
        assert_eq!(installed_bytes, 10);
        assert_eq!(required_download_bytes, 20);
    }

    #[test]
    fn desired_target_falls_back_to_standalone_until_separate_draft_is_installed() {
        let target_package = package("target", 20);
        let draft_package = package("draft", 5);
        let target = installed(
            target_package.clone(),
            ModelPackageInstallationOrigin::Magnitude,
        );
        let definition = definition(ServableModelBundle::SpeculativeDecoding {
            target: target_package,
            draft_source: SpeculativeDraftSource::Separate {
                draft: draft_package,
            },
            method: SpeculativeMethod::DSpark,
        });
        let present = BTreeMap::from([(target.package.id.clone(), &target)]);
        let resolution = catalog_resolution(&definition, &present, &[]);
        let CatalogMaterialState::Installed {
            configuration: Some(configuration),
            effective: EffectiveModel::Ready { .. },
            ..
        } = resolution.state
        else {
            panic!("the installed target must remain runnable as a standalone fallback")
        };
        assert!(matches!(
            configuration.bundle,
            ServableModelBundle::Standalone { .. }
        ));
        assert_eq!(resolution.required_download_bytes, 5);
    }

    #[test]
    fn complete_bundle_readiness_requires_every_dependency_to_be_inspected() {
        let target_package = package("target", 20);
        let draft_package = package("draft", 5);
        let target = installed(
            target_package.clone(),
            ModelPackageInstallationOrigin::Magnitude,
        );
        let mut draft = installed(
            draft_package.clone(),
            ModelPackageInstallationOrigin::HuggingFaceCache,
        );
        draft.inspection = ModelPackageInspection::Invalid {
            failure: ModelFailure {
                code: "invalid_draft".to_owned(),
                message: "draft is invalid".to_owned(),
                retryable: false,
            },
        };
        let definition = definition(ServableModelBundle::SpeculativeDecoding {
            target: target_package,
            draft_source: SpeculativeDraftSource::Separate {
                draft: draft_package,
            },
            method: SpeculativeMethod::DSpark,
        });
        let present = BTreeMap::from([
            (target.package.id.clone(), &target),
            (draft.package.id.clone(), &draft),
        ]);
        let model = catalog_model(&definition, &present, &[], &BTreeMap::new());
        let CatalogModelState::Installed {
            effective: EffectiveModel::Unavailable { failure },
            installation:
                ModelInstallation::Resolved {
                    installed_bytes,
                    ownership,
                    ..
                },
            update_state: CatalogModelUpdate::Current,
        } = model.local_state
        else {
            panic!("an invalid installed dependency must make the effective bundle unavailable")
        };
        assert_eq!(failure.code, "invalid_draft");
        assert_eq!(installed_bytes, 25);
        assert_eq!(ownership, ModelInstallationOwnership::Mixed);
    }

    #[test]
    fn multiple_prior_targets_are_installed_but_explicitly_unavailable() {
        let desired = package("desired", 20);
        let first = installed(
            package("first", 10),
            ModelPackageInstallationOrigin::Magnitude,
        );
        let second = installed(
            package("second", 11),
            ModelPackageInstallationOrigin::Magnitude,
        );
        let present = BTreeMap::from([
            (first.package.id.clone(), &first),
            (second.package.id.clone(), &second),
        ]);
        let model = catalog_model(
            &definition(ServableModelBundle::Standalone { package: desired }),
            &present,
            &[
                affiliation(&first.package.id, CatalogPackageRole::Target),
                affiliation(&second.package.id, CatalogPackageRole::Target),
            ],
            &BTreeMap::new(),
        );
        let CatalogModelState::Installed {
            effective: EffectiveModel::Unavailable { failure },
            installation:
                ModelInstallation::Unresolved {
                    installed_bytes, ..
                },
            ..
        } = model.local_state
        else {
            panic!("ambiguous prior targets are a genuine installed-but-unavailable state")
        };
        assert_eq!(failure.code, "catalog_installed_targets_ambiguous");
        assert_eq!(installed_bytes, 21);
    }

    #[test]
    fn superseded_material_does_not_leave_a_complete_desired_bundle_updateable() {
        let desired_package = package("desired", 20);
        let desired = installed(
            desired_package.clone(),
            ModelPackageInstallationOrigin::Magnitude,
        );
        let prior = installed(
            package("prior", 10),
            ModelPackageInstallationOrigin::Magnitude,
        );
        let present = BTreeMap::from([
            (desired.package.id.clone(), &desired),
            (prior.package.id.clone(), &prior),
        ]);
        let model = catalog_model(
            &definition(ServableModelBundle::Standalone {
                package: desired_package,
            }),
            &present,
            &[affiliation(&prior.package.id, CatalogPackageRole::Target)],
            &BTreeMap::new(),
        );
        let CatalogModelState::Installed {
            update_state: CatalogModelUpdate::Current,
            installation:
                ModelInstallation::Resolved {
                    installed_bytes, ..
                },
            ..
        } = model.local_state
        else {
            panic!("complete desired material must be current while cleanup runs independently")
        };
        assert_eq!(installed_bytes, 20);
    }

    #[test]
    fn catalog_installation_reports_mixed_ownership_for_duplicate_occurrences() {
        let desired_package = package("desired", 20);
        let desired = installed(
            desired_package.clone(),
            ModelPackageInstallationOrigin::Magnitude,
        );
        let present = BTreeMap::from([(desired.package.id.clone(), &desired)]);
        let origins = BTreeMap::from([(
            desired.package.id.clone(),
            InstallationOrigins {
                magnitude: true,
                external: true,
            },
        )]);
        let model = catalog_model(
            &definition(ServableModelBundle::Standalone {
                package: desired_package,
            }),
            &present,
            &[],
            &origins,
        );
        assert!(matches!(
            model.local_state,
            CatalogModelState::Installed {
                installation: ModelInstallation::Resolved {
                    ownership: ModelInstallationOwnership::Mixed,
                    ..
                },
                ..
            }
        ));
    }
}
