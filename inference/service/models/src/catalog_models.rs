use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Weak};

use futures_util::future::BoxFuture;
use futures_util::stream::BoxStream;
use magnitude_service_contracts::InventoryError;
use magnitude_service_contracts::models::{
    CatalogBaseId, CatalogComponent, CatalogInstallationAdmission, CatalogInstallationRemoval,
    CatalogMissingComponent, CatalogModel, CatalogModelState, CatalogVariantId,
    CatalogModelUpdate, CatalogModels, CatalogModelsResponse, CatalogPackageRole, CatalogSupport,
    CatalogUpdatePart,
    EffectiveModel,
    InstalledCatalogAttribution, InstalledModelPackage, ModelAssessmentSubject,
    ModelDomainInvalidation, ModelFailure, ModelId, ModelInstallation, ModelInstallationOwnership,
    ModelFileId, ModelFileRole, ModelPackage, ModelPackageId, ModelPackageInstallationOrigin,
    ModelServingConfiguration, ParsedModelId, RecommendableModel, ServableModelBundle,
    SpeculativeDraftSource, SpeculativeMethod,
};

use crate::catalog_installations::ManagedCatalogInstallations;
use crate::custom_models::{CatalogDrafterFit, DrafterContent};
use crate::inventory::{
    InstalledPackageRecord, InstalledPackagesObserver, catalog_packages, catalog_projector_package,
    catalog_store_packages, catalog_target,
};
use crate::model_domains::{ModelDomainResolver, domain_changes};
use crate::model_projection::{
    bundle_packages, effective_configuration_model, primary_model_path,
    ready_model,
};

pub struct CatalogRemovalPlan {
    pub package_ids: Vec<ModelPackageId>,
    pub installed: bool,
    pub externally_owned: bool,
    pub stores_packages: bool,
    pub kept_components: Vec<CatalogComponent>,
    pub shared: bool,
}

/// Removal deletes Magnitude's own copies of the model's packages and never another app's files.
/// A target with no copy of Magnitude's own is externally owned: only Magnitude's own components
/// of it (its projector package and drafter) are removed, and the found target stays. A target
/// that another catalog model shares is retained as shared. A found target's removal deletes the
/// stored parts no other model needs, and completes even when other models need all of them.
/// When an external copy of the target remains after Magnitude's is removed, it serves the catalog
/// model from where it is.
fn removal_plan(
    ids: &BTreeSet<ModelPackageId>,
    target_ids: &BTreeSet<ModelPackageId>,
    managed_ids: &BTreeSet<ModelPackageId>,
    shared_ids: &BTreeSet<ModelPackageId>,
) -> CatalogRemovalPlan {
    let managed_targets = target_ids
        .intersection(managed_ids)
        .cloned()
        .collect::<BTreeSet<_>>();
    CatalogRemovalPlan {
        package_ids: ids
            .intersection(managed_ids)
            .filter(|package_id| !shared_ids.contains(*package_id))
            .cloned()
            .collect(),
        installed: !target_ids.is_empty(),
        externally_owned: managed_targets.is_empty(),
        stores_packages: !managed_ids.is_disjoint(ids),
        kept_components: Vec::new(),
        shared: !managed_targets.is_disjoint(shared_ids),
    }
}

/// Whether a location other than Magnitude's store holds the bytes of Magnitude's copy of
/// `package_id`: the same file by its SHA-256, or by its header digest and size.
fn held_outside_store<'a>(
    records: impl Iterator<Item = &'a InstalledPackageRecord> + Clone,
    package_id: &ModelPackageId,
) -> bool {
    let content = records
        .clone()
        .find(|record| {
            record.installed.package.id == *package_id
                && record.installed.origin == ModelPackageInstallationOrigin::Magnitude
        })
        .and_then(|own| {
            DrafterContent::of_package(&own.installed.package).map(|content| DrafterContent {
                header_digest: own
                    .primary
                    .as_ref()
                    .map(|evidence| evidence.header_digest.clone()),
                ..content
            })
        });
    content.is_some_and(|content| {
        records
            .filter(|record| record.installed.origin != ModelPackageInstallationOrigin::Magnitude)
            .any(|record| content.held_by(record))
    })
}

/// Unavailable catalog models cannot be installed or loaded; their files remain removable.
pub(crate) fn reject_unavailable(
    model_id: &ModelId,
    definition: &RecommendableModel,
) -> Result<(), InventoryError> {
    match &definition.support {
        CatalogSupport::Supported => Ok(()),
        CatalogSupport::Disabled { reason } => Err(InventoryError::ModelOperation {
            code: "catalog_model_disabled".to_owned(),
            message: format!("catalog model {model_id} is disabled ({reason})"),
            retryable: false,
        }),
        CatalogSupport::Deprecated { replacement, reason, .. } => Err(InventoryError::ModelOperation {
            code: "catalog_model_deprecated".to_owned(),
            message: format!("catalog model {model_id} is deprecated ({reason}); use {replacement}"),
            retryable: false,
        }),
    }
}

impl ModelDomainResolver {
    pub fn serving_configuration(
        &self,
        model_id: &ModelId,
    ) -> Result<ModelServingConfiguration, InventoryError> {
        match model_id.parsed() {
            ParsedModelId::Catalog { .. } => {
                reject_unavailable(model_id, self.catalog_definition(model_id)?)?;
                self.effective_catalog_configuration(model_id)
            }
            ParsedModelId::HuggingFace { .. } | ParsedModelId::Local { .. } => {
                self.custom_serving_configuration(model_id)
            }
        }
    }

    /// The model a request names: an `hf:` ID a catalog model lists as an alias (the ID its exact
    /// target had as a found model before it was attributed) is that catalog model; every other ID
    /// is itself.
    pub fn canonical_model_id(&self, model_id: ModelId) -> Result<ModelId, InventoryError> {
        if !matches!(model_id.parsed(), ParsedModelId::HuggingFace { .. }) {
            return Ok(model_id);
        }
        let records = self
            .inventory
            .installed_packages
            .read()
            .map_err(|_| InventoryError::Internal("installed package lock poisoned".to_owned()))?;
        let aliased = catalog_aliases(records.records.values().map(|record| &record.installed), None)
            .into_iter()
            .find(|(_, aliases)| aliases.contains(&model_id))
            .map(|((base, variant), _)| ModelId::catalog(&base, &variant));
        Ok(aliased.unwrap_or(model_id))
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
                magnitude_service_contracts::models::CatalogModelSelection::Desired => {
                    let configuration = self.catalog_definition(model_id)?.configuration.clone();
                    Ok(if self.catalog_drafter_fit(model_id).withheld {
                        plain_configuration(configuration)
                    } else {
                        configuration
                    })
                }
                magnitude_service_contracts::models::CatalogModelSelection::Effective => {
                    self.effective_catalog_configuration(model_id)
                }
            },
            ModelAssessmentSubject::Custom { model_id } => self.serving_configuration(model_id),
        }
    }

    fn effective_catalog_configuration(
        &self,
        model_id: &ModelId,
    ) -> Result<ModelServingConfiguration, InventoryError> {
        let definition = self.catalog_definition(model_id)?;
        let installed = self.inventory.catalog_packages_response()?;
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
        let resolution = catalog_resolution(definition, &present, &affiliations, &BTreeSet::new())
            .serving(definition, &present, self.catalog_drafter_fit(model_id));
        match resolution.state {
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

    /// Monotonic over the inventory and the catalog's own generation.
    pub fn revision(&self) -> Result<u64, InventoryError> {
        Ok(self.inventory.installed_packages_response()?.revision
            + self.catalog_changes.generation.load(Ordering::Acquire))
    }

    /// Republishes the catalog list after which catalog models serve without their drafter changed.
    pub fn catalog_drafter_fit_changed(&self) {
        self.catalog_changes.generation.fetch_add(1, Ordering::AcqRel);
        self.publish_catalog_change();
    }

    pub(crate) fn publish_catalog_change(&self) {
        if let Ok(revision) = self.revision() {
            let _ = self
                .catalog_changes
                .changes
                .send(ModelDomainInvalidation { revision });
        }
    }

    pub fn catalog_snapshot(&self) -> Result<CatalogModelsResponse, InventoryError> {
        let installed = self.inventory.catalog_packages_response()?;
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
        let records = self
            .inventory
            .installed_packages
            .read()
            .map_err(|_| InventoryError::Internal("installed package lock poisoned".to_owned()))?;
        let aliases = catalog_aliases(
            records.records.values().map(|record| &record.installed),
            Some(&self.catalog.models),
        );
        // A catalog drafter's bytes found in any location (a folder's file has its own identity)
        // are on this computer, as they are for the found models that use that drafter.
        let found_drafter_files = self
            .catalog
            .models
            .iter()
            .filter_map(catalog_draft)
            .flat_map(|draft| &draft.files)
            .filter(|file| {
                let content = self
                    .inventory
                    .catalog_file_content(&file.sha256, file.size_bytes);
                records.records.values().any(|record| content.held_by(record))
            })
            .map(|file| file.id.clone());
        let local_files = records
            .records
            .values()
            .flat_map(|record| &record.installed.package.files)
            .map(|file| file.id.clone())
            .chain(found_drafter_files)
            .collect::<BTreeSet<_>>();
        let occurrence_origins = records
            .records
            .values()
            .filter(|record| {
                !self
                    .inventory
                    .hidden_occurrence(record.installed.origin, &record.installed.package)
            })
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
        let mut occurrence_origins = occurrence_origins;
        // A composed target's weights are its found copy's; Magnitude's projector beside them doesn't
        // make Magnitude own the model (see `installation`).
        for (composed, weights, _) in self.inventory.composed_catalog_targets(&records) {
            occurrence_origins
                .entry(composed.package.id)
                .or_default()
                .include(weights.installed.origin);
        }
        Ok(CatalogModelsResponse {
            revision: self.revision()?,
            reconciliation_complete: installed.reconciliation_complete,
            models: self
                .catalog
                .models
                .iter()
                .map(|definition| {
                    let mut model = catalog_model(
                        definition,
                        &present,
                        &affiliations,
                        &occurrence_origins,
                        &local_files,
                        self.catalog_drafter_fit(&catalog_id(definition)),
                    );
                    model.aliases = aliases
                        .get(&(definition.model_id.clone(), definition.variant_id.clone()))
                        .map(|aliases| aliases.iter().cloned().collect())
                        .unwrap_or_default();
                    model
                })
                .collect(),
        })
    }

    /// The download that fetches only `component` of catalog model `id`, or `None` when the
    /// component is not missing. Only a component a found exact target lacks can be fetched alone.
    pub(crate) fn catalog_component_download(
        &self,
        id: &ModelId,
        component: CatalogComponent,
    ) -> Result<Option<ServableModelBundle>, InventoryError> {
        let definition = self.catalog_definition(id)?;
        let model = self
            .catalog_snapshot()?
            .models
            .into_iter()
            .find(|model| model.id == *id)
            .ok_or_else(|| InventoryError::NotFound(id.to_string()))?;
        let CatalogModelState::Installed {
            missing_components, ..
        } = model.local_state
        else {
            return Err(InventoryError::ModelOperation {
                code: "catalog_model_not_installed".to_owned(),
                message: format!("catalog model {id} is not installed"),
                retryable: false,
            });
        };
        if !missing_components
            .iter()
            .any(|missing| missing.component == component)
        {
            return Ok(None);
        }
        let package = match component {
            CatalogComponent::Projector => catalog_target(definition),
            CatalogComponent::Drafter => catalog_draft(definition)
                .ok_or_else(|| InventoryError::Internal(format!("catalog model {id} has no separate drafter")))?,
        };
        Ok(Some(ServableModelBundle::Standalone {
            package: package.clone(),
        }))
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
        let installed = self.inventory.catalog_packages_response()?;
        let affiliations = self
            .inventory
            .catalog_affiliations
            .read()
            .map_err(|_| InventoryError::Internal("catalog affiliation lock poisoned".to_owned()))?
            .entries()
            .cloned()
            .collect::<Vec<_>>();
        let attributed = |entry: &InstalledModelPackage| {
            matches!(
                &entry.catalog_attribution,
                InstalledCatalogAttribution::Attributed { model_id, variant_id }
                    if model_id == &definition.model_id && variant_id == &definition.variant_id
            )
        };
        let ids = installed
            .packages
            .iter()
            .filter(|entry| {
                attributed(entry)
                    || catalog_store_packages(definition)
                        .any(|(package, _)| package.id == entry.package.id)
                    || affiliations.iter().any(|affiliation| {
                        affiliation.model_id == definition.model_id
                            && affiliation.variant_id == definition.variant_id
                            && affiliation.package_id == entry.package.id
                    })
            })
            .map(|entry| entry.package.id.clone())
            .collect::<BTreeSet<_>>();
        let attributed_targets = installed
            .packages
            .iter()
            .filter(|entry| attributed(entry))
            .map(|entry| entry.package.id.clone())
            .collect::<BTreeSet<_>>();
        let occurrences =
            self.inventory.installed_packages.read().map_err(|_| {
                InventoryError::Internal("installed package lock poisoned".to_owned())
            })?;
        let managed_ids = occurrences
            .records
            .values()
            .filter(|record| {
                ids.contains(&record.installed.package.id)
                    && record.installed.origin == ModelPackageInstallationOrigin::Magnitude
            })
            .map(|record| record.installed.package.id.clone())
            .collect::<BTreeSet<_>>();
        // A package is shared only with another catalog model that is installed: one whose target
        // is here and whose bundle (projector package included) or affiliations hold the package.
        // Variants that share a drafter or projector but aren't installed never keep it.
        let installed_elsewhere = self
            .catalog
            .models
            .iter()
            .filter(|candidate| catalog_id(candidate) != *id)
            .filter(|candidate| {
                installed.packages.iter().any(|entry| {
                    entry.package.id == catalog_target(candidate).id
                        || matches!(
                            &entry.catalog_attribution,
                            InstalledCatalogAttribution::Attributed { model_id, variant_id }
                                if model_id == &candidate.model_id && variant_id == &candidate.variant_id
                        )
                })
            })
            .collect::<Vec<_>>();
        let shared_ids = ids
            .iter()
            .filter(|package_id| {
                installed_elsewhere.iter().any(|candidate| {
                    catalog_store_packages(candidate).any(|(package, _)| package.id == **package_id)
                        || affiliations.iter().any(|affiliation| {
                            affiliation.package_id == **package_id
                                && affiliation.model_id == candidate.model_id
                                && affiliation.variant_id == candidate.variant_id
                        })
                })
            })
            .cloned()
            .collect::<BTreeSet<_>>();
        let target_ids = ids
            .iter()
            .filter(|package_id| {
                **package_id == catalog_target(definition).id
                    || attributed_targets.contains(*package_id)
                    || affiliations.iter().any(|affiliation| {
                        affiliation.model_id == definition.model_id
                            && affiliation.variant_id == definition.variant_id
                            && affiliation.package_id == **package_id
                            && affiliation.role == CatalogPackageRole::Target
                    })
            })
            .cloned()
            .collect::<BTreeSet<_>>();
        drop(occurrences);
        let mut plan = removal_plan(&ids, &target_ids, &managed_ids, &shared_ids);
        // A drafter another model uses stays; pruning removes it once no model does. While a
        // model's automatic drafter is still being worked out, this model's drafter stays too when
        // Remove deletes the target anyway. A found target's stored parts are all that Remove
        // deletes, so only a drafter another model serves with keeps one of them.
        let custom = self.custom_drafter_use()?;
        let mut in_use = custom.used;
        if custom.pending && !plan.externally_owned {
            in_use.extend(crate::custom_models::separate_draft(
                &definition.configuration.bundle,
            ));
        }
        // A model serves with any file holding a drafter's bytes, so Magnitude's copy is needed
        // only while no other location holds them.
        let occurrences =
            self.inventory.installed_packages.read().map_err(|_| {
                InventoryError::Internal("installed package lock poisoned".to_owned())
            })?;
        in_use.retain(|package_id| !held_outside_store(occurrences.records.values(), package_id));
        plan.package_ids.retain(|package_id| !in_use.contains(package_id));
        if plan.externally_owned {
            plan.kept_components = [
                (CatalogComponent::Projector, catalog_projector_package(catalog_target(definition))),
                (CatalogComponent::Drafter, catalog_draft(definition).cloned()),
            ]
            .into_iter()
            .filter_map(|(component, package)| Some((component, package?.id)))
            .filter(|(_, id)| managed_ids.contains(id) && !plan.package_ids.contains(id))
            .map(|(component, _)| component)
            .collect();
        }
        // The store lists a target held with its projector and, from the same snapshot file, that
        // projector as a package of its own: deleting the target deletes the file, so the projector
        // package goes with it rather than unlinking the target's file before the target's turn.
        plan.package_ids = occurrences.without_contained_managed_packages(&plan.package_ids);
        drop(occurrences);
        Ok(plan)
    }

    pub(crate) fn catalog_cleanup_package_ids(
        &self,
        id: &ModelId,
    ) -> Result<Vec<ModelPackageId>, InventoryError> {
        let definition = self.catalog_definition(id)?;
        let installed = self.inventory.catalog_packages_response()?;
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
        let resolution = catalog_resolution(definition, &present, &affiliations, &BTreeSet::new());
        if !resolution.missing_desired_package_ids.is_empty() {
            return Ok(Vec::new());
        }
        // A package holding a current catalog package's weights or drafter is that package's
        // material (its weights found on their own, say), never a superseded revision.
        let current_files = self
            .catalog
            .models
            .iter()
            .flat_map(catalog_packages)
            .flat_map(|(package, _)| &package.files)
            .filter(|file| file.role != ModelFileRole::Projector)
            .map(|file| &file.id)
            .collect::<BTreeSet<_>>();
        let superseded = resolution
            .superseded_package_ids
            .into_iter()
            .filter(|package_id| {
                present.get(package_id).is_some_and(|entry| {
                    entry.origin == ModelPackageInstallationOrigin::Magnitude
                        && !entry
                            .package
                            .files
                            .iter()
                            .any(|file| current_files.contains(&file.id))
                })
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
            .collect::<BTreeSet<_>>();
        let records =
            self.inventory.installed_packages.read().map_err(|_| {
                InventoryError::Internal("installed package lock poisoned".to_owned())
            })?;
        Ok(superseded
            .iter()
            .filter(|package_id| !records.shares_managed_files(package_id, &superseded))
            .cloned()
            .collect())
    }
}

#[derive(Clone)]
pub struct ManagedCatalogModels {
    resolver: Arc<ModelDomainResolver>,
    installations: Arc<ManagedCatalogInstallations>,
    maintenance_generation: Arc<AtomicU64>,
    maintenance_running: Arc<AtomicBool>,
    inventory_observer: Arc<dyn InstalledPackagesObserver>,
}

impl ManagedCatalogModels {
    pub(crate) fn new(
        resolver: Arc<ModelDomainResolver>,
        installations: Arc<ManagedCatalogInstallations>,
        make_inventory_observer: impl FnOnce(Weak<Self>) -> Arc<dyn InstalledPackagesObserver>,
    ) -> Arc<Self> {
        Arc::new_cyclic(|catalog| Self {
            resolver,
            installations,
            maintenance_generation: Arc::new(AtomicU64::new(0)),
            maintenance_running: Arc::new(AtomicBool::new(false)),
            inventory_observer: make_inventory_observer(catalog.clone()),
        })
    }

    pub(crate) fn installed_packages_changed(&self, _revision: u64) {
        self.resolver.publish_catalog_change();
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

    fn install_catalog_component(
        &self,
        id: &ModelId,
        component: CatalogComponent,
    ) -> BoxFuture<'_, Result<CatalogInstallationAdmission, InventoryError>> {
        let id = id.clone();
        Box::pin(async move { self.installations.install_component(&id, component).await })
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
            self.resolver.catalog_changes.changes.subscribe(),
        )
    }
}

fn catalog_model(
    definition: &RecommendableModel,
    present: &BTreeMap<ModelPackageId, &InstalledModelPackage>,
    affiliations: &[magnitude_service_contracts::models::CatalogPackageAffiliation],
    occurrence_origins: &BTreeMap<ModelPackageId, InstallationOrigins>,
    local_files: &BTreeSet<ModelFileId>,
    fit: CatalogDrafterFit,
) -> CatalogModel {
    let resolution = catalog_resolution(definition, present, affiliations, local_files)
        .serving(definition, present, fit);
    CatalogModel {
        id: catalog_id(definition),
        desired: ready_model(
            &definition.configuration.bundle,
            definition.configuration.profile.clone(),
        ),
        display_name: definition.display_name.clone(),
        variant_label: definition.variant_label.clone(),
        description: definition.description.clone(),
        release_date: definition.release_date.clone(),
        license: definition.license.clone(),
        source_urls: bundle_packages(&definition.configuration.bundle)
            .filter_map(|package| match &package.source {
                magnitude_service_contracts::models::ModelPackageSource::HuggingFace { repository, .. } => {
                    Some(format!("https://huggingface.co/{repository}"))
                }
                magnitude_service_contracts::models::ModelPackageSource::Local { .. } => None,
            })
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect(),
        parameterization: definition.parameterization.clone(),
        intelligence: definition.intelligence.clone(),
        support: definition.support.clone(),
        fidelity_rank: definition.fidelity_rank,
        quantization_aware: definition.quantization_aware,
        aliases: Vec::new(),
        local_state: match resolution.state {
            CatalogMaterialState::NotInstalled => CatalogModelState::NotInstalled,
            CatalogMaterialState::Installed {
                selected_target,
                effective,
                ..
            } => {
                let installation = installation(
                    &resolution.installed_packages,
                    selected_target,
                    occurrence_origins,
                );
                let stored_components = stored_components(definition, &installation, present);
                CatalogModelState::Installed {
                    effective,
                    installation,
                    stored_components,
                    update_state: if resolution.required_download_bytes == 0 {
                        CatalogModelUpdate::Current
                    } else {
                        CatalogModelUpdate::Available {
                            required_download_bytes: resolution.required_download_bytes,
                            parts: resolution.update_parts,
                        }
                    },
                    missing_components: resolution.missing_components,
                    drafter_withheld: resolution.drafter_withheld,
                }
            }
        },
    }
}

/// The components Magnitude stores for a model whose weights are only in another app's location:
/// its target's projector package and its drafter, each when Magnitude's store holds it.
fn stored_components(
    definition: &RecommendableModel,
    installation: &ModelInstallation,
    present: &BTreeMap<ModelPackageId, &InstalledModelPackage>,
) -> Vec<CatalogComponent> {
    let external = matches!(
        installation,
        ModelInstallation::Resolved { ownership: ModelInstallationOwnership::External, .. }
            | ModelInstallation::Unresolved { ownership: ModelInstallationOwnership::External, .. }
    );
    if !external {
        return Vec::new();
    }
    let stored = |package: Option<ModelPackage>| {
        package.is_some_and(|package| {
            present
                .get(&package.id)
                .is_some_and(|entry| entry.origin == ModelPackageInstallationOrigin::Magnitude)
        })
    };
    [
        (CatalogComponent::Projector, catalog_projector_package(catalog_target(definition))),
        (CatalogComponent::Drafter, catalog_draft(definition).cloned()),
    ]
    .into_iter()
    .filter(|(_, package)| stored(package.clone()))
    .map(|(component, _)| component)
    .collect()
}

/// `required_download_bytes` is the update: what a newer catalog revision needs fetched.
/// `missing_components` are what a found exact target lacks, which is not an update.
struct CatalogResolution<'a> {
    installed_packages: Vec<&'a InstalledModelPackage>,
    missing_desired_package_ids: Vec<ModelPackageId>,
    superseded_package_ids: Vec<ModelPackageId>,
    required_download_bytes: u64,
    update_parts: Vec<CatalogUpdatePart>,
    missing_components: Vec<CatalogMissingComponent>,
    drafter_withheld: bool,
    state: CatalogMaterialState<'a>,
}

impl<'a> CatalogResolution<'a> {
    /// The model as it serves on this computer: plain when it fits only without its catalog
    /// drafter, else with its built-in heads while the catalog drafter is not here.
    fn serving(
        self,
        definition: &RecommendableModel,
        present: &BTreeMap<ModelPackageId, &InstalledModelPackage>,
        fit: CatalogDrafterFit,
    ) -> Self {
        if fit.withheld {
            self.without_drafter(present)
        } else if fit.built_in_heads {
            self.with_built_in_heads(definition, present)
        } else {
            self
        }
    }

    /// The current target served with its built-in heads while the catalog's separate drafter is
    /// not on this computer; the drafter stays a missing component to add.
    fn with_built_in_heads(
        mut self,
        definition: &RecommendableModel,
        present: &BTreeMap<ModelPackageId, &InstalledModelPackage>,
    ) -> Self {
        let catalog_target = catalog_target(definition);
        let draft_missing =
            catalog_draft(definition).is_some_and(|draft| !present.contains_key(&draft.id));
        if let CatalogMaterialState::Installed {
            configuration: Some(configuration),
            effective,
            ..
        } = &mut self.state
            && draft_missing
            && let ServableModelBundle::Standalone { package } = &configuration.bundle
            && (package.id == catalog_target.id
                || crate::package_service::exact_target_weights(catalog_target, package))
        {
            *configuration = ModelServingConfiguration {
                bundle: ServableModelBundle::SpeculativeDecoding {
                    target: package.clone(),
                    draft_source: SpeculativeDraftSource::Embedded,
                    method: SpeculativeMethod::Mtp,
                },
                profile: configuration.profile.clone(),
            };
            *effective = effective_configuration_model(configuration, present);
        }
        self
    }

    /// The model served without its catalog drafter, which it fits only without, whether or not
    /// that drafter is here; the drafter is then no missing component either.
    fn without_drafter(mut self, present: &BTreeMap<ModelPackageId, &InstalledModelPackage>) -> Self {
        if let CatalogMaterialState::Installed {
            configuration,
            effective,
            ..
        } = &mut self.state
        {
            if let Some(configuration) = configuration
                && matches!(configuration.bundle, ServableModelBundle::SpeculativeDecoding { .. })
            {
                *configuration = plain_configuration(configuration.clone());
                *effective = effective_configuration_model(configuration, present);
            }
            self.drafter_withheld = true;
        }
        self.missing_components
            .retain(|missing| missing.component != CatalogComponent::Drafter);
        self
    }
}

/// A configuration served plain: its target alone.
fn plain_configuration(configuration: ModelServingConfiguration) -> ModelServingConfiguration {
    let package = match configuration.bundle {
        ServableModelBundle::Standalone { package }
        | ServableModelBundle::SpeculativeDecoding {
            target: package, ..
        } => package,
    };
    ModelServingConfiguration {
        bundle: ServableModelBundle::Standalone { package },
        profile: configuration.profile,
    }
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
            ModelPackageInstallationOrigin::HuggingFaceCache
            | ModelPackageInstallationOrigin::LmStudio
            | ModelPackageInstallationOrigin::LlamaCpp
            | ModelPackageInstallationOrigin::Ollama
            | ModelPackageInstallationOrigin::Folder
            | ModelPackageInstallationOrigin::Imported => self.external = true,
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

/// `local_files` are files on this computer in any location, a hidden found copy's included, which
/// an installation copies rather than downloads.
///
/// The current target, the catalog's target package or a found exact copy of its weights (in
/// another app's location), serves with every catalog component on this computer: the catalog
/// drafter when it is here, without a projector it lacks. What it lacks is reported as missing
/// components, never as an update; update means an older catalog revision of this model is here.
fn catalog_resolution<'a>(
    definition: &RecommendableModel,
    present: &BTreeMap<ModelPackageId, &'a InstalledModelPackage>,
    affiliations: &[magnitude_service_contracts::models::CatalogPackageAffiliation],
    local_files: &BTreeSet<ModelFileId>,
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
    // A missing package's files already present elsewhere (its exact target weights found in a
    // Hugging Face cache, hidden or not) are copied locally, not downloaded.
    let present_files = present
        .values()
        .flat_map(|entry| &entry.package.files)
        .map(|file| &file.id)
        .chain(local_files)
        .collect::<BTreeSet<_>>();
    let download_bytes = |package: &ModelPackage| -> u64 {
        package
            .files
            .iter()
            .filter(|file| !present_files.contains(&file.id))
            .map(|file| file.size_bytes)
            .sum()
    };
    let missing_bytes = desired_packages
        .iter()
        .filter(|(package, _)| !present.contains_key(&package.id))
        .map(|(package, _)| download_bytes(package))
        .sum::<u64>();
    // A missing target whose weights are all here differs only in its projector: an image input update.
    let missing_parts = desired_packages
        .iter()
        .filter(|(package, _)| !present.contains_key(&package.id) && download_bytes(package) > 0)
        .map(|(package, role)| match role {
            CatalogPackageRole::Dependency => CatalogUpdatePart::Drafter,
            CatalogPackageRole::Target
                if package
                    .files
                    .iter()
                    .filter(|file| file.role != ModelFileRole::Projector)
                    .all(|file| present_files.contains(&file.id)) =>
            {
                CatalogUpdatePart::Projector
            }
            CatalogPackageRole::Target => CatalogUpdatePart::Model,
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let projector_package = catalog_projector_package(catalog_target(definition));
    let superseded_package_ids = affiliations
        .iter()
        .filter(|affiliation| {
            affiliation.model_id == definition.model_id
                && affiliation.variant_id == definition.variant_id
                && !desired_ids.contains(&affiliation.package_id)
                && projector_package
                    .as_ref()
                    .is_none_or(|projector| projector.id != affiliation.package_id)
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
    // A found copy and a projector package that a composed target holds count once, as it.
    let installed_packages = installed_packages
        .iter()
        .filter(|entry| {
            !installed_packages.iter().any(|other| {
                other.package.id != entry.package.id
                    && other.package.files.len() > entry.package.files.len()
                    && entry
                        .package
                        .files
                        .iter()
                        .all(|file| other.package.files.iter().any(|held| held.id == file.id))
            })
        })
        .copied()
        .collect::<Vec<_>>();
    let catalog_target = catalog_target(definition);
    let targets = present
        .values()
        .filter(|entry| {
            entry.package.id == catalog_target.id
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
        .find(|entry| entry.package.id == catalog_target.id)
        .copied();
    // The current target: the catalog's own target package, else a found copy of its exact weights.
    let current_target = desired_target.or_else(|| {
        targets
            .iter()
            .find(|entry| {
                crate::package_service::exact_target_weights(catalog_target, &entry.package)
            })
            .copied()
    });
    let selected_target = current_target.or(match targets.as_slice() {
        [target] => Some(*target),
        _ => None,
    });
    let mut required_download_bytes = missing_bytes;
    let mut update_parts = missing_parts;
    let mut missing_components = Vec::new();
    let state = if targets.is_empty() {
        CatalogMaterialState::NotInstalled
    } else if missing_desired_package_ids.is_empty() {
        let configuration = definition.configuration.clone();
        CatalogMaterialState::Installed {
            selected_target: desired_target,
            effective: effective_configuration_model(&configuration, present),
            configuration: Some(configuration),
        }
    } else if let Some(target) = current_target {
        if superseded_package_ids.is_empty() {
            required_download_bytes = 0;
            update_parts = Vec::new();
            missing_components = current_target_missing_components(
                definition,
                &target.package,
                present,
                download_bytes,
            );
        }
        let configuration = ModelServingConfiguration {
            bundle: present_components_bundle(
                &definition.configuration.bundle,
                &target.package,
                present,
            ),
            profile: definition.configuration.profile.clone(),
        };
        CatalogMaterialState::Installed {
            selected_target: Some(target),
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
        update_parts,
        missing_components,
        drafter_withheld: false,
        state,
    }
}

/// The catalog bundle served from `target` with the catalog's drafter when it is on this computer.
fn present_components_bundle(
    bundle: &ServableModelBundle,
    target: &ModelPackage,
    present: &BTreeMap<ModelPackageId, &InstalledModelPackage>,
) -> ServableModelBundle {
    match bundle {
        ServableModelBundle::SpeculativeDecoding {
            draft_source: SpeculativeDraftSource::Separate { draft },
            ..
        } if !present.contains_key(&draft.id) => ServableModelBundle::Standalone {
            package: target.clone(),
        },
        ServableModelBundle::SpeculativeDecoding {
            draft_source,
            method,
            ..
        } => ServableModelBundle::SpeculativeDecoding {
            target: target.clone(),
            draft_source: draft_source.clone(),
            method: method.clone(),
        },
        ServableModelBundle::Standalone { .. } => ServableModelBundle::Standalone {
            package: target.clone(),
        },
    }
}

/// What the current target serves without: the catalog target's projector when the serving target
/// (a found copy of its weights) lacks it, and the catalog's separate drafter when it is not on this
/// computer. Each downloads only the bytes not already here.
fn current_target_missing_components(
    definition: &RecommendableModel,
    found: &ModelPackage,
    present: &BTreeMap<ModelPackageId, &InstalledModelPackage>,
    download_bytes: impl Fn(&ModelPackage) -> u64,
) -> Vec<CatalogMissingComponent> {
    let catalog_target = catalog_target(definition);
    let lacks_projector = catalog_target.files.iter().any(|file| {
        file.role == ModelFileRole::Projector
            && !found.files.iter().any(|found| found.id == file.id)
    });
    let projector = catalog_projector_package(catalog_target)
        .filter(|_| lacks_projector)
        .map(|projector| CatalogMissingComponent {
            component: CatalogComponent::Projector,
            download_bytes: download_bytes(&projector),
        });
    let drafter = catalog_draft(definition)
        .filter(|draft| !present.contains_key(&draft.id))
        .map(|draft| CatalogMissingComponent {
            component: CatalogComponent::Drafter,
            download_bytes: download_bytes(draft),
        });
    projector.into_iter().chain(drafter).collect()
}

fn catalog_draft(definition: &RecommendableModel) -> Option<&ModelPackage> {
    match &definition.configuration.bundle {
        ServableModelBundle::SpeculativeDecoding {
            draft_source: SpeculativeDraftSource::Separate { draft },
            ..
        } => Some(draft),
        ServableModelBundle::Standalone { .. }
        | ServableModelBundle::SpeculativeDecoding {
            draft_source: SpeculativeDraftSource::Embedded,
            ..
        } => None,
    }
}

/// The `hf:` IDs of external installations of each catalog variant's exact target: what each is
/// called as a found custom model. Attributed copies always count; with `catalog`, so do exact
/// copies that are hidden (and so not attributed), which the catalog lists so they can be unhidden
/// by the catalog ID.
fn catalog_aliases<'a>(
    installed: impl Iterator<Item = &'a InstalledModelPackage>,
    catalog: Option<&[RecommendableModel]>,
) -> BTreeMap<(CatalogBaseId, CatalogVariantId), BTreeSet<ModelId>> {
    let mut aliases = BTreeMap::<_, BTreeSet<ModelId>>::new();
    for entry in installed {
        if entry.origin == ModelPackageInstallationOrigin::Magnitude {
            continue;
        }
        let Some(alias) = crate::package_service::hub_model_id(&entry.package) else {
            continue;
        };
        let mut variants = Vec::new();
        if let InstalledCatalogAttribution::Attributed {
            model_id,
            variant_id,
        } = &entry.catalog_attribution
        {
            variants.push((model_id.clone(), variant_id.clone()));
        }
        variants.extend(catalog.into_iter().flatten().filter_map(|definition| {
            crate::package_service::exact_target_weights(catalog_target(definition), &entry.package)
                .then(|| (definition.model_id.clone(), definition.variant_id.clone()))
        }));
        for variant in variants {
            aliases.entry(variant).or_default().insert(alias.clone());
        }
    }
    aliases
}

pub(crate) fn catalog_id(definition: &RecommendableModel) -> ModelId {
    ModelId::catalog(&definition.model_id, &definition.variant_id)
}

/// Ownership is of the model's own weights: where the selected target's copies are. A catalog
/// drafter or projector in Magnitude's store beside a found copy of the weights doesn't make
/// Magnitude own the model, so Remove (Magnitude's copy of the weights) and Hide (weights found
/// only elsewhere) follow the target alone. Without a selected target, every installed package
/// counts.
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
    let owned = selected_target.map_or_else(|| installed.to_vec(), |target| vec![target]);
    let origins = owned
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
        (false, true) => ModelInstallationOwnership::External,
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

    use magnitude_service_contracts::models::{
        CatalogBaseId, CatalogPackageAffiliation, CatalogSupport,
        CatalogVariantId, ModelFile, ModelFileId, ModelFileRole, ModelPackage,
        ModelPackageProperties, ModelPackageSource, ModelParameterization, ModelReleaseDate,
        PackageValidation, ResolvedModelInstallation, ServingProfile, SpeculativeDraftSource,
        SpeculativeMethod,
    };

    use super::*;
    use crate::model_projection::resolved_installation;

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
            validation: PackageValidation::Valid,
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
            parameterization: ModelParameterization::Dense {
                total_parameters: 1_000_000,
            },
            intelligence: 1,
            support: CatalogSupport::Supported,
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
            &BTreeSet::from([target.clone(), shared_dependency.clone()]),
            &BTreeSet::from([shared_dependency]),
        );
        let _ = external_dependency;
        assert!(plan.installed);
        assert!(!plan.externally_owned);
        assert!(!plan.shared);
        assert_eq!(plan.package_ids, vec![target]);
    }

    #[test]
    fn removal_is_retained_when_the_target_itself_is_external_or_shared() {
        let target = ModelPackageId("target".to_owned());
        let ids = BTreeSet::from([target.clone()]);
        let external = removal_plan(&ids, &ids, &BTreeSet::new(), &BTreeSet::new());
        assert!(external.externally_owned);
        assert!(external.package_ids.is_empty());
        let shared = removal_plan(&ids, &ids, &ids, &BTreeSet::from([target]));
        assert!(shared.shared);
    }

    /// After upgrading from 0.2.9, the catalog's target is both in Magnitude's store and in a
    /// Hugging Face cache: removal deletes Magnitude's copy and keeps the cache's, which then
    /// serves the catalog model.
    #[test]
    fn removal_deletes_magnitudes_copy_when_an_external_copy_of_the_target_also_exists() {
        let managed = ModelPackageId("target".to_owned());
        let external = ModelPackageId("found-weights".to_owned());
        let ids = BTreeSet::from([managed.clone(), external.clone()]);
        let plan = removal_plan(&ids, &ids, &BTreeSet::from([managed.clone()]), &BTreeSet::new());
        assert!(plan.installed);
        assert!(!plan.externally_owned);
        assert!(!plan.shared);
        assert_eq!(plan.package_ids, vec![managed]);
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
        );
        assert_eq!(ready.metadata.storage_bytes, 25);
        assert_eq!(ready.speculative_method, Some(SpeculativeMethod::DFlash));
    }

    #[test]
    fn unavailable_catalog_models_are_rejected_with_their_reason_or_replacement() {
        let mut model = definition(ServableModelBundle::Standalone {
            package: package("target", 1),
        });
        let id = catalog_id(&model);
        assert!(reject_unavailable(&id, &model).is_ok());
        model.support = CatalogSupport::Disabled {
            reason: "not yet qualified".to_owned(),
        };
        let Err(InventoryError::ModelOperation { code, message, .. }) = reject_unavailable(&id, &model) else {
            panic!("a disabled model must be rejected");
        };
        assert_eq!(code, "catalog_model_disabled");
        assert!(message.contains("not yet qualified"));
        model.support = CatalogSupport::Deprecated {
            since: "2026-09-27".to_owned(),
            replacement: "replacement:gguf:q4".parse().expect("model ID"),
            reason: "unsupported architecture".to_owned(),
        };
        let Err(InventoryError::ModelOperation {
            code,
            message,
            retryable,
        }) = reject_unavailable(&id, &model)
        else {
            panic!("a deprecated model must be rejected");
        };
        assert_eq!(code, "catalog_model_deprecated");
        assert!(message.contains("replacement:gguf:q4"));
        assert!(!retryable);
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
        let resolution = catalog_resolution(&definition, &BTreeMap::new(), &[], &BTreeSet::new());
        assert!(matches!(
            resolution.state,
            CatalogMaterialState::NotInstalled
        ));
    }

    /// The catalog target found in a Hugging Face cache without its projector, as `hf download`
    /// of the one GGUF leaves it.
    fn found_exact_target_without_projector() -> (ModelPackage, ModelPackage) {
        let mut found = package("found", 3_000);
        found.source = ModelPackageSource::HuggingFace {
            repository: "unsloth/gemma-4-E2B-it-qat-GGUF".to_owned(),
            revision: "a".repeat(40),
        };
        let mut target = found.clone();
        target.id = ModelPackageId("target".to_owned());
        target.files.push(ModelFile {
            id: ModelFileId("file-projector".to_owned()),
            path: PathBuf::from("mmproj.gguf"),
            role: ModelFileRole::Projector,
            size_bytes: 600,
            tensor_storage_bytes: Some(600),
            sha256: "b".repeat(64),
        });
        (found, target)
    }

    fn attributed_external(
        found: ModelPackage,
        definition: &RecommendableModel,
    ) -> InstalledModelPackage {
        let mut external = installed(found, ModelPackageInstallationOrigin::HuggingFaceCache);
        external.catalog_attribution = InstalledCatalogAttribution::Attributed {
            model_id: definition.model_id.clone(),
            variant_id: definition.variant_id.clone(),
        };
        external
    }

    #[test]
    fn a_found_exact_target_serves_with_the_catalog_drafter_and_reports_its_missing_projector() {
        let (found, target) = found_exact_target_without_projector();
        let draft = package("draft", 900);
        let definition = definition(ServableModelBundle::SpeculativeDecoding {
            target,
            draft_source: SpeculativeDraftSource::Separate {
                draft: draft.clone(),
            },
            method: SpeculativeMethod::Mtp,
        });
        let external = attributed_external(found.clone(), &definition);
        let drafter = installed(draft.clone(), ModelPackageInstallationOrigin::Magnitude);
        let present = BTreeMap::from([
            (external.package.id.clone(), &external),
            (drafter.package.id.clone(), &drafter),
        ]);
        let local_files = present
            .values()
            .flat_map(|entry| &entry.package.files)
            .map(|file| file.id.clone())
            .collect::<BTreeSet<_>>();

        let resolution = catalog_resolution(&definition, &present, &[], &local_files);
        let CatalogMaterialState::Installed {
            configuration: Some(configuration),
            effective: EffectiveModel::Ready { .. },
            ..
        } = resolution.state
        else {
            panic!("the found exact target serves the catalog model");
        };
        assert_eq!(
            configuration.bundle,
            ServableModelBundle::SpeculativeDecoding {
                target: found,
                draft_source: SpeculativeDraftSource::Separate { draft },
                method: SpeculativeMethod::Mtp,
            },
            "the catalog drafter on this computer serves with the found weights at once"
        );

        let model = catalog_model(
            &definition,
            &present,
            &[],
            &BTreeMap::new(),
            &local_files,
            CatalogDrafterFit::default(),
        );
        let CatalogModelState::Installed {
            effective: EffectiveModel::Ready { model: ready },
            installation: ModelInstallation::Resolved { ownership, .. },
            update_state,
            missing_components,
            ..
        } = model.local_state
        else {
            panic!("the found exact target is the installed catalog model: {:?}", model.local_state);
        };
        assert_eq!(
            ownership,
            ModelInstallationOwnership::External,
            "the catalog drafter in Magnitude's store doesn't make Magnitude own the found weights"
        );
        assert_eq!(ready.speculative_method, Some(SpeculativeMethod::Mtp));
        assert_eq!(
            update_state,
            CatalogModelUpdate::Current,
            "a missing projector is not a newer catalog revision"
        );
        assert_eq!(
            missing_components,
            vec![CatalogMissingComponent {
                component: CatalogComponent::Projector,
                download_bytes: 600,
            }],
            "only the projector is fetched; the found weights are copied, never downloaded"
        );
    }

    /// A catalog model that fits only without its drafter serves plain and says so, rather than
    /// being refused; the drafter is then no missing component to add.
    #[test]
    fn a_catalog_model_that_fits_only_without_its_drafter_serves_plain_and_says_so() {
        let (found, target) = found_exact_target_without_projector();
        let draft = package("draft", 900);
        let definition = definition(ServableModelBundle::SpeculativeDecoding {
            target,
            draft_source: SpeculativeDraftSource::Separate {
                draft: draft.clone(),
            },
            method: SpeculativeMethod::DFlash,
        });
        let external = attributed_external(found.clone(), &definition);
        let drafter = installed(draft, ModelPackageInstallationOrigin::Magnitude);
        let with_drafter = BTreeMap::from([
            (external.package.id.clone(), &external),
            (drafter.package.id.clone(), &drafter),
        ]);
        let without_drafter = BTreeMap::from([(external.package.id.clone(), &external)]);
        for present in [&with_drafter, &without_drafter] {
            let model = catalog_model(
                &definition,
                present,
                &[],
                &BTreeMap::new(),
                &BTreeSet::new(),
                withheld(),
            );
            let CatalogModelState::Installed {
                effective: EffectiveModel::Ready { model: ready },
                missing_components,
                drafter_withheld,
                ..
            } = model.local_state
            else {
                panic!("the model is installed and ready: {:?}", model.local_state);
            };
            assert_eq!(ready.speculative_method, None, "it serves plain");
            assert!(drafter_withheld, "and says why");
            assert!(
                missing_components
                    .iter()
                    .all(|missing| missing.component != CatalogComponent::Drafter),
                "a drafter it won't use is never offered: {missing_components:?}"
            );
            let resolution = catalog_resolution(&definition, present, &[], &BTreeSet::new())
                .without_drafter(present);
            let CatalogMaterialState::Installed {
                configuration: Some(configuration),
                ..
            } = resolution.state
            else {
                panic!("installed");
            };
            assert_eq!(
                configuration.bundle,
                ServableModelBundle::Standalone {
                    package: found.clone()
                }
            );
        }
        let model = catalog_model(
            &definition,
            &with_drafter,
            &[],
            &BTreeMap::new(),
            &BTreeSet::new(),
            CatalogDrafterFit::default(),
        );
        assert!(matches!(
            model.local_state,
            CatalogModelState::Installed { drafter_withheld: false, .. }
        ));
    }

    #[test]
    fn a_found_exact_target_without_the_catalog_drafter_offers_it_alone() {
        let (found, target) = found_exact_target_without_projector();
        let definition = definition(ServableModelBundle::SpeculativeDecoding {
            target,
            draft_source: SpeculativeDraftSource::Separate {
                draft: package("draft", 900),
            },
            method: SpeculativeMethod::DFlash,
        });
        let external = attributed_external(found.clone(), &definition);
        let present = BTreeMap::from([(external.package.id.clone(), &external)]);
        let resolution = catalog_resolution(&definition, &present, &[], &BTreeSet::new());
        let CatalogMaterialState::Installed {
            configuration: Some(configuration),
            effective: EffectiveModel::Ready { .. },
            ..
        } = &resolution.state
        else {
            panic!("the found exact target serves the catalog model");
        };
        assert_eq!(
            configuration.bundle,
            ServableModelBundle::Standalone { package: found }
        );
        assert_eq!(resolution.required_download_bytes, 0);
        assert_eq!(
            resolution.missing_components,
            vec![
                CatalogMissingComponent {
                    component: CatalogComponent::Projector,
                    download_bytes: 600,
                },
                CatalogMissingComponent {
                    component: CatalogComponent::Drafter,
                    download_bytes: 900,
                },
            ]
        );
    }

    fn withheld() -> CatalogDrafterFit {
        CatalogDrafterFit {
            withheld: true,
            built_in_heads: false,
        }
    }

    /// The mac-m4 round 3c finding: a found exact catalog target with built-in heads serves with
    /// them while its catalog drafter is missing, still offers that drafter, and takes the drafter
    /// once it is here.
    #[test]
    fn a_found_exact_target_serves_its_built_in_heads_until_its_catalog_drafter_is_here() {
        let (found, target) = found_exact_target_without_projector();
        let draft = package("draft", 900);
        let definition = definition(ServableModelBundle::SpeculativeDecoding {
            target,
            draft_source: SpeculativeDraftSource::Separate {
                draft: draft.clone(),
            },
            method: SpeculativeMethod::DFlash,
        });
        let heads = CatalogDrafterFit {
            withheld: false,
            built_in_heads: true,
        };
        let external = attributed_external(found.clone(), &definition);
        let without_drafter = BTreeMap::from([(external.package.id.clone(), &external)]);
        let model = catalog_model(
            &definition,
            &without_drafter,
            &[],
            &BTreeMap::new(),
            &BTreeSet::new(),
            heads,
        );
        let CatalogModelState::Installed {
            effective: EffectiveModel::Ready { model: ready },
            missing_components,
            update_state: CatalogModelUpdate::Current,
            ..
        } = model.local_state
        else {
            panic!(
                "the found exact target is the installed catalog model: {:?}",
                model.local_state
            );
        };
        assert_eq!(
            ready.speculative_method,
            Some(SpeculativeMethod::Mtp),
            "its heads serve"
        );
        assert!(
            missing_components
                .iter()
                .any(|missing| missing.component == CatalogComponent::Drafter),
            "the catalog drafter is still offered: {missing_components:?}"
        );
        let resolution = catalog_resolution(&definition, &without_drafter, &[], &BTreeSet::new())
            .serving(&definition, &without_drafter, heads);
        let CatalogMaterialState::Installed {
            configuration: Some(configuration),
            ..
        } = resolution.state
        else {
            panic!("installed");
        };
        assert_eq!(
            configuration.bundle,
            ServableModelBundle::SpeculativeDecoding {
                target: found.clone(),
                draft_source: SpeculativeDraftSource::Embedded,
                method: SpeculativeMethod::Mtp,
            }
        );

        let drafter = installed(draft, ModelPackageInstallationOrigin::Magnitude);
        let with_drafter = BTreeMap::from([
            (external.package.id.clone(), &external),
            (drafter.package.id.clone(), &drafter),
        ]);
        let model = catalog_model(
            &definition,
            &with_drafter,
            &[],
            &BTreeMap::new(),
            &BTreeSet::new(),
            heads,
        );
        let CatalogModelState::Installed {
            effective: EffectiveModel::Ready { model: ready },
            ..
        } = model.local_state
        else {
            panic!("installed and ready: {:?}", model.local_state);
        };
        assert_eq!(
            ready.speculative_method,
            Some(SpeculativeMethod::DFlash),
            "the catalog drafter serves once here"
        );

        let plain = catalog_model(
            &definition,
            &without_drafter,
            &[],
            &BTreeMap::new(),
            &BTreeSet::new(),
            CatalogDrafterFit::default(),
        );
        assert!(
            matches!(
                plain.local_state,
                CatalogModelState::Installed {
                    effective: EffectiveModel::Ready { model: ref ready },
                    ..
                } if ready.speculative_method.is_none()
            ),
            "a target without heads serves plain"
        );
    }

    /// The Fedora round 3b sequence: the exact target in a Hugging Face cache with no drafter
    /// anywhere, then the projector alone (stored as its own package, the target composed of it and
    /// the found weights), then the drafter alone. No step is an update, and each step reports
    /// exactly what is still missing.
    #[test]
    fn a_found_exact_target_fetching_its_projector_then_its_drafter_is_never_an_update() {
        let (found, target) = found_exact_target_without_projector();
        let draft = package("draft", 900);
        let definition = definition(ServableModelBundle::SpeculativeDecoding {
            target: target.clone(),
            draft_source: SpeculativeDraftSource::Separate {
                draft: draft.clone(),
            },
            method: SpeculativeMethod::Mtp,
        });
        let external = attributed_external(found.clone(), &definition);
        // Fetching the projector stores only the projector package; the inventory then lists the
        // catalog's target composed of the found weights and that package, where the weights are.
        let stored_projector = installed(
            crate::inventory::catalog_projector_package(&target).unwrap(),
            ModelPackageInstallationOrigin::Magnitude,
        );
        let composed_target = attributed_external(target.clone(), &definition);
        let stored_draft = installed(draft.clone(), ModelPackageInstallationOrigin::Magnitude);
        let affiliations = [affiliation(&stored_projector.package.id, CatalogPackageRole::Dependency)];
        let step = |entries: &[&InstalledModelPackage]| {
            let present = entries
                .iter()
                .map(|entry| (entry.package.id.clone(), *entry))
                .collect::<BTreeMap<_, _>>();
            let local_files = entries
                .iter()
                .flat_map(|entry| &entry.package.files)
                .map(|file| file.id.clone())
                .collect::<BTreeSet<_>>();
            let resolution = catalog_resolution(&definition, &present, &affiliations, &local_files);
            let installed_bytes = resolution
                .installed_packages
                .iter()
                .flat_map(|entry| &entry.package.files)
                .map(|file| file.size_bytes)
                .sum::<u64>();
            let CatalogMaterialState::Installed {
                configuration: Some(configuration),
                effective: EffectiveModel::Ready { model },
                ..
            } = resolution.state
            else {
                panic!("the catalog model is installed and ready at every step");
            };
            (
                installed_bytes,
                configuration.bundle,
                model.speculative_method,
                resolution.required_download_bytes,
                resolution.missing_components,
            )
        };
        let projector = CatalogMissingComponent {
            component: CatalogComponent::Projector,
            download_bytes: 600,
        };
        let drafter = CatalogMissingComponent {
            component: CatalogComponent::Drafter,
            download_bytes: 900,
        };

        let (bytes, bundle, method, update, missing) = step(&[&external]);
        assert_eq!(bytes, 3_000);
        assert_eq!(bundle, ServableModelBundle::Standalone { package: found });
        assert_eq!((method, update), (None, 0));
        assert_eq!(missing, vec![projector, drafter.clone()]);

        let (bytes, bundle, method, update, missing) =
            step(&[&external, &stored_projector, &composed_target]);
        assert_eq!(bytes, 3_000 + 600, "the found weights and the projector count once");
        assert_eq!(
            bundle,
            ServableModelBundle::Standalone {
                package: target.clone()
            },
            "the catalog's target, composed of the found weights and the stored projector, serves"
        );
        assert_eq!((method, update), (None, 0), "a missing drafter is not an update");
        assert_eq!(missing, vec![drafter], "only the drafter is left to fetch");

        let (bytes, bundle, method, update, missing) =
            step(&[&external, &stored_projector, &composed_target, &stored_draft]);
        assert_eq!(bytes, 3_000 + 600 + 900);
        assert_eq!(bundle, definition.configuration.bundle);
        assert_eq!((method, update), (Some(SpeculativeMethod::Mtp), 0));
        assert!(missing.is_empty());
    }

    #[test]
    fn a_found_exact_target_beside_an_older_catalog_revision_is_an_update() {
        let (found, target) = found_exact_target_without_projector();
        let definition = definition(ServableModelBundle::SpeculativeDecoding {
            target,
            draft_source: SpeculativeDraftSource::Separate {
                draft: package("draft", 900),
            },
            method: SpeculativeMethod::DFlash,
        });
        let external = attributed_external(found, &definition);
        let prior_draft = installed(package("prior-draft", 800), ModelPackageInstallationOrigin::Magnitude);
        let present = BTreeMap::from([
            (external.package.id.clone(), &external),
            (prior_draft.package.id.clone(), &prior_draft),
        ]);
        let resolution = catalog_resolution(
            &definition,
            &present,
            &[affiliation(&prior_draft.package.id, CatalogPackageRole::Dependency)],
            &BTreeSet::new(),
        );
        assert_eq!(resolution.required_download_bytes, 600 + 900);
        assert_eq!(resolution.update_parts, [CatalogUpdatePart::Projector, CatalogUpdatePart::Drafter]);
        assert!(resolution.missing_components.is_empty());
    }

    #[test]
    fn an_attributed_external_target_keeps_the_hub_id_it_had_as_a_found_model_as_an_alias() {
        let definition = definition(ServableModelBundle::Standalone {
            package: package("target", 10),
        });
        let mut found = package("found", 10);
        found.source = ModelPackageSource::HuggingFace {
            repository: "unsloth/Qwen3.5-4B-MTP-GGUF".to_owned(),
            revision: "a".repeat(40),
        };
        found.files[0].path = PathBuf::from("Qwen3.5-4B-UD-Q4_K_XL.gguf");
        let mut external = installed(found.clone(), ModelPackageInstallationOrigin::HuggingFaceCache);
        external.catalog_attribution = InstalledCatalogAttribution::Attributed {
            model_id: definition.model_id.clone(),
            variant_id: definition.variant_id.clone(),
        };
        let mut managed = installed(found, ModelPackageInstallationOrigin::Magnitude);
        managed.catalog_attribution = external.catalog_attribution.clone();
        let unattributed = installed(package("other", 10), ModelPackageInstallationOrigin::HuggingFaceCache);

        let aliases = catalog_aliases([&external, &managed, &unattributed].into_iter(), None);
        assert_eq!(
            aliases
                .get(&(definition.model_id.clone(), definition.variant_id.clone()))
                .map(|aliases| aliases.iter().map(ToString::to_string).collect::<Vec<_>>()),
            Some(vec!["hf:unsloth/Qwen3.5-4B-MTP-GGUF/Qwen3.5-4B-UD-Q4_K_XL.gguf".to_owned()])
        );
        assert_eq!(aliases.len(), 1);
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
            &BTreeSet::new(),
            CatalogDrafterFit::default(),
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
                    ref parts,
                },
            ..
        } = model.local_state
        else {
            panic!("the unique prior target must remain a ready installed fallback")
        };
        assert_eq!(installed_bytes, 10);
        assert_eq!(required_download_bytes, 20);
        assert_eq!(parts, &[CatalogUpdatePart::Model]);
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
        let resolution = catalog_resolution(&definition, &present, &[], &BTreeSet::new());
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
        assert_eq!(
            resolution.required_download_bytes, 0,
            "a missing drafter with no earlier revision here is not an update"
        );
        assert_eq!(
            resolution.missing_components,
            vec![CatalogMissingComponent {
                component: CatalogComponent::Drafter,
                download_bytes: 5,
            }]
        );

        // A missing package whose files are on this computer, as a hidden found copy's are, is
        // copied, not downloaded.
        let held = BTreeSet::from([ModelFileId("file-draft".to_owned())]);
        assert_eq!(
            catalog_resolution(&definition, &present, &[], &held).missing_components,
            vec![CatalogMissingComponent {
                component: CatalogComponent::Drafter,
                download_bytes: 0,
            }]
        );
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
        draft.validation = PackageValidation::Invalid {
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
        let model = catalog_model(
            &definition,
            &present,
            &[],
            &BTreeMap::new(),
            &BTreeSet::new(),
            CatalogDrafterFit::default(),
        );
        let CatalogModelState::Installed {
            effective: EffectiveModel::Unavailable { failure },
            installation:
                ModelInstallation::Resolved {
                    installed_bytes,
                    ownership,
                    ..
                },
            update_state: CatalogModelUpdate::Current,
            ..
        } = model.local_state
        else {
            panic!("an invalid installed dependency must make the effective bundle unavailable")
        };
        assert_eq!(failure.code, "invalid_draft");
        assert_eq!(installed_bytes, 25);
        assert_eq!(
            ownership,
            ModelInstallationOwnership::Magnitude,
            "ownership follows the target, not a drafter found elsewhere"
        );
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
            &BTreeSet::new(),
            CatalogDrafterFit::default(),
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
            &BTreeSet::new(),
            CatalogDrafterFit::default(),
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
            &BTreeSet::new(),
            CatalogDrafterFit::default(),
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
