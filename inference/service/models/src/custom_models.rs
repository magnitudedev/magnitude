//! Custom models: every non-catalog model found in the configured locations or declared by ACN,
//! with one identity per content, an origin that decides who owns its files, and a serving
//! configuration that includes its attached drafter.
//!
//! Identity is `hf:<owner>/<repository>/<selector>` when any occurrence knows its Hugging Face
//! file, else `local:<header digest>`. Occurrences with the same header digest and primary size
//! are the same model wherever they are; the managed store, then the Hugging Face cache, LM
//! Studio, llama.cpp, Ollama and folders represent it in that order.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock, Weak};

use futures_util::future::BoxFuture;
use futures_util::stream::BoxStream;
use magnitude_service_contracts::models::{
    AttachedDrafter, AttachedDrafterState, CatalogPackageRemover, CatalogPackageRole, CustomModel,
    CustomModelCatalogAttribution, CustomModelDeclaration, CustomModelDeclarationOrigin,
    CustomModelFile, CustomModelOrigin, CustomModelState, DrafterCandidate, DrafterChoice,
    DrafterPruning, DrafterReference, EffectiveModel, FoundSource, HuggingFaceArtifactSelector,
    HuggingFaceRepositoryId, InstalledCatalogAttribution, InstalledModelPackage,
    KnownUnsupportedReason, LocalModelDigest, LocalModelRemoval, LocalModels,
    LocalModelsConfiguration, LocalModelsResponse, ModelDomainInvalidation, ModelFailure,
    ModelFileRole, ModelId, ModelInstallationOwnership, ModelPackage, ModelPackageId,
    ModelPackageInstallationOrigin, ModelServingConfiguration, PackageValidation,
    RecommendableModelCatalog, ResolvedModelInstallation, ServableModelBundle, ServingProfile,
    SpeculativeDraftSource, SpeculativeMethod, UnsupportedLocalModel, UnsupportedReason,
};
use magnitude_service_contracts::{
    ExternalOrigin, HubFile, InventoryError, ModelLocation, ModelSource,
};

use crate::candidate_files::RecognizedArchitectures;
use crate::catalog_models::catalog_id;
use crate::gguf::{GgufBaseModel, GgufRole};
use crate::inventory::{
    InstalledPackageRecord, InstalledPackageSnapshot, ManagedModelStore, UNSUPPORTED_FORMAT_CODE,
    UnreadableDirectory, hf_repo_dir, origin_precedence,
};
use crate::local_watch::{RootWatcher, watch_targets};
use crate::model_domains::{ModelDomainResolver, domain_changes};
use crate::model_projection::{effective_configuration_model, resolved_installation};
use crate::package_service::PrimaryEvidence;

const DEFAULT_EXTERNAL_CONTEXT: u32 = 100_000;

/// What the projection knows about one custom model, for the enrichment that finds the catalog
/// model it is a variant of and the drafter to recommend.
pub struct CustomModelFacts<'a> {
    pub model_id: &'a ModelId,
    pub origin: &'a CustomModelOrigin,
    pub display_name: &'a str,
    pub target: &'a ModelPackage,
    pub hub: Option<&'a HubFile>,
    pub ollama_name: Option<&'a str>,
    pub name: Option<&'a str>,
    pub architecture: Option<&'a str>,
    pub base_models: &'a [GgufBaseModel],
}

/// A custom model's lineage hint, license, its automatic drafter candidates (every candidate the
/// engine binds, in the order automatic choice takes them, the first being the recommended one),
/// and the catalog models it matches, whose drafters it may use.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CustomModelEnrichmentResult {
    pub variant_of: Option<ModelId>,
    pub drafters: Vec<DrafterCandidate>,
    pub catalog_drafters: Vec<ModelId>,
    pub license: Option<String>,
}

impl CustomModelEnrichmentResult {
    #[must_use]
    pub fn recommended_drafter(&self) -> Option<&DrafterCandidate> {
        self.drafters.first()
    }
}

/// Lineage and drafter recommendation for custom models. It is called while the local-models
/// list is projected, so it answers from what it already knows and does no I/O; when its
/// knowledge changes it calls [`ModelDomainResolver::custom_model_enrichment_changed`]. A model
/// whose drafter choice is `Auto` uses the first of its automatic drafters that is on this
/// computer, checked against the files present at each projection.
pub trait CustomModelEnrichment: Send + Sync + 'static {
    fn enrich(&self, model: &CustomModelFacts<'_>) -> CustomModelEnrichmentResult;
    /// Whether `enrich` answers from knowledge rather than from not having computed it yet.
    fn knows(&self, _model: &CustomModelFacts<'_>) -> bool {
        true
    }
    /// The bytes a drafter reference names, when known, so a copy of them anywhere on this
    /// computer is used as that drafter.
    fn drafter_content(&self, _reference: &DrafterReference) -> Option<DrafterContent> {
        None
    }
    /// How a catalog model serves without its catalog drafter. When this changes it calls
    /// [`ModelDomainResolver::catalog_drafter_fit_changed`].
    fn catalog_drafter_fit(&self, _model_id: &ModelId) -> CatalogDrafterFit {
        CatalogDrafterFit::default()
    }
}

/// How a catalog model serves without its catalog drafter, which depends on the release material
/// and this computer, never on the files here. `withheld`: it fits served plain and not with that
/// drafter, so it is served plain whether or not the drafter is here. `built_in_heads`: its target
/// has built-in heads the engine binds and that keep it fitting, which serve while the catalog
/// drafter is not on this computer.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CatalogDrafterFit {
    pub withheld: bool,
    pub built_in_heads: bool,
}

/// The bytes of a drafter's GGUF: the digest of its header and its size, and its Hub SHA-256 when
/// known. A file with these bytes in any location, another app's included, is that drafter, used
/// where it is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DrafterContent {
    pub header_digest: Option<String>,
    pub sha256: Option<String>,
    pub size_bytes: u64,
}

impl DrafterContent {
    /// The content of a package's single weights file, by the digest its file identity records.
    #[must_use]
    pub fn of_package(package: &ModelPackage) -> Option<Self> {
        let mut weights = package
            .files
            .iter()
            .filter(|file| matches!(file.role, ModelFileRole::Weights | ModelFileRole::Draft));
        let file = weights.next()?;
        if weights.next().is_some() {
            return None;
        }
        Some(Self {
            header_digest: None,
            sha256: Some(file.sha256.clone()),
            size_bytes: file.size_bytes,
        })
    }

    /// These bytes with whatever `other`, describing the same file, knows besides.
    #[must_use]
    pub fn merge(self, other: Option<Self>) -> Self {
        match other {
            Some(other) if other.size_bytes == self.size_bytes => Self {
                header_digest: self.header_digest.or(other.header_digest),
                sha256: self.sha256.or(other.sha256),
                size_bytes: self.size_bytes,
            },
            _ => self,
        }
    }

    /// Whether an installed package's single weights file holds these bytes.
    pub(crate) fn held_by(&self, record: &InstalledPackageRecord) -> bool {
        let Some(file) = Self::of_package(&record.installed.package) else {
            return false;
        };
        file.size_bytes == self.size_bytes
            && (self.header_digest.as_deref().is_some_and(|digest| {
                record
                    .primary
                    .as_ref()
                    .is_some_and(|evidence| evidence.header_digest == digest)
            }) || self.sha256.is_some() && self.sha256 == file.sha256)
    }
}

struct NoEnrichment;

impl CustomModelEnrichment for NoEnrichment {
    fn enrich(&self, _model: &CustomModelFacts<'_>) -> CustomModelEnrichmentResult {
        CustomModelEnrichmentResult::default()
    }
}

/// Local-models state the domain resolver shares with the local-models service: the enrichment
/// hook and a generation that advances whenever the projection changes without the inventory
/// changing (configuration and enrichment).
pub(crate) struct LocalModelsContext {
    enrichment: RwLock<Arc<dyn CustomModelEnrichment>>,
    generation: AtomicU64,
    configuration_generation: AtomicU64,
    changes: tokio::sync::broadcast::Sender<ModelDomainInvalidation>,
}

impl LocalModelsContext {
    pub(crate) fn new() -> Self {
        let (changes, _) = tokio::sync::broadcast::channel(32);
        Self {
            enrichment: RwLock::new(Arc::new(NoEnrichment)),
            generation: AtomicU64::new(0),
            configuration_generation: AtomicU64::new(0),
            changes,
        }
    }
}

impl ModelDomainResolver {
    /// Installs the custom-model enrichment and republishes the local-models list.
    pub fn set_custom_model_enrichment(&self, enrichment: Arc<dyn CustomModelEnrichment>) {
        if let Ok(mut current) = self.local.enrichment.write() {
            *current = enrichment;
        }
        self.custom_model_enrichment_changed();
    }

    /// How a catalog model serves without its catalog drafter.
    pub(crate) fn catalog_drafter_fit(&self, model_id: &ModelId) -> CatalogDrafterFit {
        self.local
            .enrichment
            .read()
            .map(|enrichment| enrichment.catalog_drafter_fit(model_id))
            .unwrap_or_default()
    }

    /// Republishes the local-models list after the enrichment's knowledge changed.
    pub fn custom_model_enrichment_changed(&self) {
        self.local_models_changed();
    }

    /// Republishes the local-models list after something it shows changed without the inventory.
    pub(crate) fn local_models_changed(&self) {
        self.local.generation.fetch_add(1, Ordering::AcqRel);
        self.publish_local_change();
    }

    fn publish_local_change(&self) {
        if let Ok(revision) = self.local_revision() {
            let _ = self
                .local
                .changes
                .send(ModelDomainInvalidation { revision });
        }
    }

    /// What a custom model's enrichment is computed from on this side: the installed packages
    /// (every file in a location or the store, drafters included) and the local-models
    /// configuration. It advances whenever either changes, and never for the enrichment's own
    /// results, so an enrichment recomputed at a newer revision follows every change to them.
    pub fn custom_model_inputs_revision(&self) -> Result<u64, InventoryError> {
        Ok(self.inventory.installed_packages_response()?.revision
            + self.local.configuration_generation.load(Ordering::Acquire))
    }

    /// Monotonic over both the inventory and the local generation.
    pub(crate) fn local_revision(&self) -> Result<u64, InventoryError> {
        Ok(self.inventory.installed_packages_response()?.revision
            + self.local.generation.load(Ordering::Acquire))
    }

    fn local_projection(&self) -> Result<LocalProjection, InventoryError> {
        let installed = self.inventory.installed_packages_response()?;
        let configuration = self.inventory.local_models_configuration()?;
        let enrichment = self
            .local
            .enrichment
            .read()
            .map_err(|_| InventoryError::Internal("enrichment lock poisoned".to_owned()))?
            .clone();
        let records =
            self.inventory.installed_packages.read().map_err(|_| {
                InventoryError::Internal("installed package lock poisoned".to_owned())
            })?;
        let catalog_drafters = self.catalog_drafter_revisions();
        let mut projection = project(
            &records,
            &configuration,
            &self.catalog,
            enrichment.as_ref(),
            &self.inventory.config.root,
            &self.inventory.config.architectures,
            &catalog_drafters,
        );
        self.upstream.annotate(
            projection
                .models
                .iter_mut()
                .chain(&mut projection.hidden)
                .map(|projected| &mut projected.model),
            &configuration.declarations,
        );
        projection.revision = installed.revision + self.local.generation.load(Ordering::Acquire);
        projection.reconciliation_complete = installed.reconciliation_complete;
        Ok(projection)
    }

    /// Every drafter package each catalog model has had: the packages recorded as its drafter
    /// dependency, earlier revisions included, by catalog model.
    fn catalog_drafter_revisions(&self) -> CatalogDrafterRevisions {
        let mut revisions = CatalogDrafterRevisions::new();
        if let Ok(affiliations) = self.inventory.catalog_affiliations.read() {
            for affiliation in affiliations
                .entries()
                .filter(|affiliation| affiliation.role == CatalogPackageRole::Dependency)
            {
                revisions
                    .entry(ModelId::catalog(
                        &affiliation.model_id,
                        &affiliation.variant_id,
                    ))
                    .or_default()
                    .insert(affiliation.package_id.clone());
            }
        }
        revisions
    }

    /// The separate drafter packages custom models use: the drafter each serves with, or downloads
    /// with its target. `None` while a model's automatic drafter isn't worked out yet, when it may
    /// be about to use any catalog drafter.
    pub(crate) fn custom_drafter_packages(
        &self,
    ) -> Result<Option<BTreeSet<ModelPackageId>>, InventoryError> {
        let projection = self.local_projection()?;
        let projected = || projection.models.iter().chain(&projection.hidden);
        if projected().any(|projected| projected.drafter_pending) {
            return Ok(None);
        }
        Ok(Some(
            projected()
                .filter_map(|projected| match &projected.configuration {
                    Ok(configuration) => separate_draft(&configuration.bundle),
                    Err(_) => projected.awaited_drafter.clone(),
                })
                .collect(),
        ))
    }

    /// The Magnitude-owned drafter packages no model uses: not the drafter of an installed
    /// catalog model, and not the drafter any custom model serves with.
    fn unused_drafter_packages(&self) -> Result<Vec<ModelPackageId>, InventoryError> {
        let custom = self.custom_drafter_packages()?;
        let installed = self.inventory.catalog_packages_response()?;
        let present = installed
            .packages
            .iter()
            .map(|entry| &entry.package.id)
            .collect::<BTreeSet<_>>();
        let mut used = self
            .catalog
            .models
            .iter()
            .filter(|definition| {
                custom.is_none() || present.contains(&crate::lineage::catalog_target(definition).id)
            })
            .filter_map(|definition| separate_draft(&definition.configuration.bundle))
            .collect::<BTreeSet<_>>();
        used.extend(custom.into_iter().flatten());
        // A drafter stays while any model here would use it: as its recommended drafter or its
        // catalog model's drafter (even with its drafter turned off), or as a drafter from its own
        // repository.
        let projection = self.local_projection()?;
        let projected = projection.models.iter().chain(&projection.hidden);
        used.extend(
            projected
                .clone()
                .flat_map(|model| model.candidate_drafters.iter().cloned()),
        );
        let repositories = projected
            .filter_map(|model| model.lineage_repository.as_deref())
            .map(str::to_lowercase)
            .collect::<BTreeSet<_>>();
        let records =
            self.inventory.installed_packages.read().map_err(|_| {
                InventoryError::Internal("installed package lock poisoned".to_owned())
            })?;
        let unused = records
            .records
            .values()
            .filter(|record| record.installed.origin == ModelPackageInstallationOrigin::Magnitude)
            .filter(|record| {
                record
                    .primary
                    .as_ref()
                    .is_some_and(|evidence| {
                        evidence.role(&self.inventory.config.architectures) == GgufRole::Drafter
                    })
            })
            .filter(|record| {
                !matches!(
                    &record.installed.package.source,
                    magnitude_service_contracts::models::ModelPackageSource::HuggingFace { repository, .. }
                        if repositories.contains(&repository.to_lowercase())
                )
            })
            .map(|record| record.installed.package.id.clone())
            .filter(|package_id| !used.contains(package_id))
            .collect::<BTreeSet<_>>();
        Ok(unused.into_iter().collect())
    }

    /// Why drafter pruning may not delete `package_id`, checked against the inventory right
    /// before the deletion and whatever the reference accounting decided. Pruning deletes only a
    /// package of Magnitude's that holds drafter files alone (never a target's weights, which a
    /// model with built-in heads also is), that no installed catalog model's bundle and no custom
    /// model uses, and that shares no file with another package. Whether a resident model has it
    /// loaded is the residency's admission, at the removal itself.
    pub(crate) fn drafter_prune_refusal(
        &self,
        package_id: &ModelPackageId,
    ) -> Result<Option<&'static str>, InventoryError> {
        let projection = self.local_projection()?;
        let installed = self.inventory.catalog_packages_response()?;
        let records =
            self.inventory.installed_packages.read().map_err(|_| {
                InventoryError::Internal("installed package lock poisoned".to_owned())
            })?;
        let managed = records
            .records
            .values()
            .filter(|record| {
                &record.installed.package.id == package_id
                    && record.installed.origin == ModelPackageInstallationOrigin::Magnitude
            })
            .collect::<Vec<_>>();
        let Some(package) = managed.first().map(|record| &record.installed.package) else {
            return Ok(Some("Magnitude holds no copy of it"));
        };
        let drafter_only = package
            .files
            .iter()
            .all(|file| matches!(file.role, ModelFileRole::Weights | ModelFileRole::Draft))
            && managed.iter().all(|record| {
                record.primary.as_ref().is_some_and(|evidence| {
                    evidence.role(&self.inventory.config.architectures) == GgufRole::Drafter
                })
            });
        if !drafter_only {
            return Ok(Some("it is not a drafter-only package"));
        }
        let files = package
            .files
            .iter()
            .map(|file| &file.id)
            .collect::<BTreeSet<_>>();
        let holds = |other: &ModelPackage| {
            &other.id == package_id || other.files.iter().any(|file| files.contains(&file.id))
        };
        if self
            .catalog
            .models
            .iter()
            .any(|definition| holds(crate::lineage::catalog_target(definition)))
        {
            return Ok(Some("it holds a catalog model's target"));
        }
        let present = installed
            .packages
            .iter()
            .map(|entry| &entry.package.id)
            .collect::<BTreeSet<_>>();
        if self
            .catalog
            .models
            .iter()
            .filter(|definition| present.contains(&crate::lineage::catalog_target(definition).id))
            .flat_map(crate::inventory::catalog_packages)
            .any(|(package, _)| holds(package))
        {
            return Ok(Some("an installed catalog model uses it"));
        }
        let custom_uses = projection
            .models
            .iter()
            .chain(&projection.hidden)
            .any(|projected| {
                projected.managed_packages.contains(package_id)
                    || projected.awaited_drafter.as_ref() == Some(package_id)
                    || projected.configuration.as_ref().is_ok_and(|configuration| {
                        crate::model_projection::bundle_packages(&configuration.bundle)
                            .any(|package| holds(package))
                    })
            });
        if custom_uses {
            return Ok(Some("a custom model uses it"));
        }
        if records.shares_managed_files(package_id, &BTreeSet::new()) {
            return Ok(Some("another package is served from its files"));
        }
        Ok(None)
    }

    /// The Hub repository a custom model's lineage and drafters come from, when its files name
    /// one: its Hugging Face identity's, or an Ollama `hf.co` pull's even when no repository file
    /// has its bytes.
    pub fn custom_model_lineage_repository(
        &self,
        model_id: &ModelId,
    ) -> Result<Option<String>, InventoryError> {
        let projection = self.local_projection()?;
        Ok(projection
            .models
            .iter()
            .chain(&projection.hidden)
            .find(|projected| &projected.model.model_id == model_id)
            .and_then(|projected| projected.lineage_repository.clone()))
    }

    /// The primary file of every place a custom model's bytes are found, each with the folder
    /// its "in this folder" drafters come from. Only this model's own files: another model whose
    /// file shares their header is not one of its places.
    pub fn custom_model_locations(
        &self,
        model_id: &ModelId,
    ) -> Result<Vec<PathBuf>, InventoryError> {
        let projection = self.local_projection()?;
        Ok(projection
            .models
            .into_iter()
            .chain(projection.hidden)
            .find(|projected| &projected.model.model_id == model_id)
            .map(|projected| projected.locations)
            .unwrap_or_default())
    }

    pub(crate) fn custom_serving_configuration(
        &self,
        model_id: &ModelId,
    ) -> Result<ModelServingConfiguration, InventoryError> {
        let projection = self.local_projection()?;
        let projected = projection
            .models
            .into_iter()
            .chain(projection.hidden)
            .find(|projected| projected.model.model_id == *model_id)
            .ok_or_else(|| InventoryError::NotFound(model_id.to_string()))?;
        projected
            .configuration
            .map_err(|failure| InventoryError::ModelOperation {
                code: failure.code,
                message: failure.message,
                retryable: failure.retryable,
            })
    }
}

#[derive(Default)]
struct LocalProjection {
    revision: u64,
    reconciliation_complete: bool,
    models: Vec<ProjectedModel>,
    hidden: Vec<ProjectedModel>,
    unsupported: Vec<UnsupportedLocalModel>,
}

impl LocalProjection {
    fn response(self) -> LocalModelsResponse {
        LocalModelsResponse {
            revision: self.revision,
            reconciliation_complete: self.reconciliation_complete,
            models: self.models.into_iter().map(|model| model.model).collect(),
            hidden: self.hidden.into_iter().map(|model| model.model).collect(),
            unsupported: self.unsupported,
        }
    }
}

struct ProjectedModel {
    model: CustomModel,
    locations: Vec<PathBuf>,
    lineage_repository: Option<String>,
    candidate_drafters: Vec<ModelPackageId>,
    configuration: Result<ModelServingConfiguration, ModelFailure>,
    managed_packages: Vec<ModelPackageId>,
    drafter_pending: bool,
    awaited_drafter: Option<ModelPackageId>,
}

#[derive(Clone)]
struct HubIdentity {
    file: HubFile,
    commit: Option<String>,
    current: bool,
}

/// One observed occurrence of a package, with what identity needs from it. `lineage_repository` is
/// the Hub repository a location names without giving the file its identity: an Ollama `hf.co`
/// pull's, which serves lineage and drafter matching only.
struct Occurrence<'a> {
    record: &'a InstalledPackageRecord,
    evidence: Option<&'a PrimaryEvidence>,
    hub: Option<HubIdentity>,
    lineage_repository: Option<String>,
    ollama_name: Option<&'a str>,
    base: PathBuf,
    primary: PathBuf,
    primary_size: u64,
    precedence: u8,
}

impl Occurrence<'_> {
    fn origin(&self) -> ModelPackageInstallationOrigin {
        self.record.installed.origin
    }

    /// The same bytes, wherever they are: the model's `local:` digest and its primary file's size.
    /// A model without a Hugging Face identity is identified by the same digest.
    fn content_key(&self) -> Option<(String, u64)> {
        self.evidence
            .and_then(|evidence| evidence.local_digest.clone())
            .map(|digest| (digest, self.primary_size))
    }

    fn found_source(&self) -> Option<FoundSource> {
        match self.origin() {
            ModelPackageInstallationOrigin::HuggingFaceCache => Some(FoundSource::HuggingFaceCache),
            ModelPackageInstallationOrigin::LmStudio => Some(FoundSource::LmStudio),
            ModelPackageInstallationOrigin::LlamaCpp => Some(FoundSource::LlamaCpp),
            ModelPackageInstallationOrigin::Ollama => Some(FoundSource::Ollama),
            ModelPackageInstallationOrigin::Folder => Some(FoundSource::Folder {
                path: self.base.clone(),
            }),
            ModelPackageInstallationOrigin::Magnitude
            | ModelPackageInstallationOrigin::Imported => None,
        }
    }

    fn usable(&self) -> Result<(), ModelFailure> {
        match &self.record.installed.validation {
            PackageValidation::Valid => Ok(()),
            PackageValidation::Invalid { failure } | PackageValidation::Unsupported { failure } => {
                Err(failure.clone())
            }
            PackageValidation::Pending => Err(ModelFailure {
                code: "model_inspection_pending".to_owned(),
                message: "the file has not been inspected yet".to_owned(),
                retryable: true,
            }),
        }
    }
}

fn record_base(managed_root: &Path, record: &InstalledPackageRecord) -> Option<PathBuf> {
    match (&record.model.location, &record.model.source) {
        (
            ModelLocation::MagnitudeCache { .. },
            ModelSource::HuggingFace {
                repository, commit, ..
            },
        ) => Some(
            managed_root
                .join("hub")
                .join(hf_repo_dir(repository))
                .join("snapshots")
                .join(commit),
        ),
        (ModelLocation::HuggingFaceCache { cache_root, .. }, _) => Some(cache_root.clone()),
        (ModelLocation::External { root, .. } | ModelLocation::Directory { root, .. }, _) => {
            Some(root.clone())
        }
        (ModelLocation::File { path, .. }, _) => path.parent().map(Path::to_path_buf),
        (ModelLocation::MagnitudeCache { .. }, ModelSource::Local { .. }) => None,
    }
}

fn hub_file(repository: &str, selector: &str) -> Option<HubFile> {
    HuggingFaceRepositoryId::new(repository).ok()?;
    HuggingFaceArtifactSelector::new(selector).ok()?;
    Some(HubFile {
        repository: repository.to_owned(),
        selector: selector.to_owned(),
    })
}

fn occurrence<'a>(
    managed_root: &Path,
    record: &'a InstalledPackageRecord,
) -> Option<Occurrence<'a>> {
    let base = record_base(managed_root, record)?;
    let primary_file = record
        .installed
        .package
        .files
        .iter()
        .filter(|file| file.role == ModelFileRole::Weights)
        .min_by(|left, right| left.path.cmp(&right.path))?;
    let selector = primary_file
        .path
        .components()
        .map(|component| component.as_os_str().to_str())
        .collect::<Option<Vec<_>>>()
        .map(|parts| parts.join("/"));
    let (hub, lineage_repository, ollama_name) =
        match (&record.model.location, &record.model.source) {
            (
                ModelLocation::MagnitudeCache { .. } | ModelLocation::HuggingFaceCache { .. },
                ModelSource::HuggingFace {
                    repository,
                    requested_revision,
                    commit,
                    ..
                },
            ) => (
                selector
                    .as_deref()
                    .and_then(|selector| hub_file(repository, selector))
                    .map(|file| HubIdentity {
                        file,
                        commit: Some(commit.clone()),
                        current: requested_revision == "main",
                    }),
                None,
                None,
            ),
            // An Ollama model keeps its `local:` identity: its `hf.co` repository, and the repository
            // file with its bytes once resolved, serve lineage and drafter matching only.
            (
                ModelLocation::External {
                    origin: ExternalOrigin::Ollama { name, hub },
                    ..
                },
                _,
            ) => (
                None,
                hub.as_ref()
                    .map(|file| file.repository.clone())
                    .or_else(|| ollama_hub_repository(name)),
                Some(name.as_str()),
            ),
            (ModelLocation::External { origin, .. }, _) => {
                let hub = match origin {
                    ExternalOrigin::LmStudio { hub } | ExternalOrigin::LlamaCpp { hub } => {
                        hub.as_ref()
                    }
                    ExternalOrigin::Ollama { .. }
                    | ExternalOrigin::Folder
                    | ExternalOrigin::Imported => None,
                };
                (
                    hub.map(|file| HubIdentity {
                        file: file.clone(),
                        commit: None,
                        current: false,
                    }),
                    None,
                    None,
                )
            }
            _ => (None, None, None),
        };
    Some(Occurrence {
        record,
        evidence: record.primary.as_ref(),
        hub,
        lineage_repository,
        ollama_name,
        primary: base.join(&primary_file.path),
        primary_size: primary_file.size_bytes,
        base,
        precedence: origin_precedence(record.installed.origin),
    })
}

/// The repository of an Ollama `hf.co/<owner>/<repository>:<tag>` pull, by its name.
fn ollama_hub_repository(name: &str) -> Option<String> {
    let rest = name
        .strip_prefix("hf.co/")
        .or_else(|| name.strip_prefix("huggingface.co/"))?;
    let repository = rest
        .split_once(':')
        .map_or(rest, |(repository, _)| repository);
    HuggingFaceRepositoryId::new(repository).ok()?;
    Some(repository.to_owned())
}

fn same_file(left: &Path, right: &Path) -> bool {
    left == right
        || matches!(
            (left.canonicalize(), right.canonicalize()),
            (Ok(left), Ok(right)) if left == right
        )
}

fn occurrence_rank(occurrence: &Occurrence<'_>) -> (u8, bool, std::cmp::Reverse<u64>) {
    (
        occurrence.precedence,
        !occurrence
            .hub
            .as_ref()
            .is_some_and(|identity| identity.current),
        std::cmp::Reverse(occurrence.record.model.updated_at),
    )
}

/// Occurrences of one content, best first.
struct ContentGroup<'a> {
    digest: String,
    occurrences: Vec<&'a Occurrence<'a>>,
}

impl<'a> ContentGroup<'a> {
    fn best(&self) -> &'a Occurrence<'a> {
        self.occurrences[0]
    }

    fn hub(&self) -> Option<&'a HubFile> {
        self.occurrences
            .iter()
            .find_map(|occurrence| occurrence.hub.as_ref().map(|identity| &identity.file))
    }

    /// The Hub repository lineage and drafter matching use: the identity's, else one a location
    /// names (an Ollama `hf.co` pull's).
    fn lineage_repository(&self) -> Option<String> {
        self.hub().map(|file| file.repository.clone()).or_else(|| {
            self.occurrences
                .iter()
                .find_map(|occurrence| occurrence.lineage_repository.clone())
        })
    }

    fn computed_id(&self) -> Option<ModelId> {
        match self.hub() {
            Some(file) => Some(ModelId::hugging_face(
                &HuggingFaceRepositoryId::new(file.repository.clone()).ok()?,
                &HuggingFaceArtifactSelector::new(file.selector.clone()).ok()?,
            )),
            None => Some(ModelId::local(
                &LocalModelDigest::new(self.digest.clone()).ok()?,
            )),
        }
    }

    fn matches(&self, declaration: &CustomModelDeclaration) -> bool {
        let Some(target) = declaration.target.first() else {
            return false;
        };
        match &declaration.origin {
            CustomModelDeclarationOrigin::Added { repository, commit } => {
                self.occurrences.iter().any(|occurrence| {
                    occurrence.hub.as_ref().is_some_and(|identity| {
                        identity.file.repository == repository.as_str()
                            && identity.file.selector == *target
                            && identity.commit.as_deref() == Some(commit.as_str())
                    })
                })
            }
            CustomModelDeclarationOrigin::Imported { .. } => self
                .occurrences
                .iter()
                .any(|occurrence| same_file(&occurrence.primary, Path::new(target))),
        }
    }

    /// The declaration that names this content: an added model's, else an imported file's while
    /// no location Magnitude scans holds the same bytes. Importing bytes already found elsewhere
    /// adds one more location of that model and changes neither its identity nor its origin.
    fn declaration<'d>(
        &self,
        declarations: &'d [CustomModelDeclaration],
    ) -> Option<&'d CustomModelDeclaration> {
        let matching = || {
            declarations
                .iter()
                .filter(|declaration| self.matches(declaration))
        };
        matching()
            .find(|declaration| {
                matches!(
                    declaration.origin,
                    CustomModelDeclarationOrigin::Added { .. }
                )
            })
            .or_else(|| {
                let found = self
                    .occurrences
                    .iter()
                    .any(|occurrence| occurrence.found_source().is_some());
                matching().find(|_| !found)
            })
    }
}

fn custom_profile(installed: &InstalledModelPackage) -> ServingProfile {
    ServingProfile {
        context_length: installed
            .package
            .properties
            .maximum_context_length
            .unwrap_or(DEFAULT_EXTERNAL_CONTEXT),
    }
}

fn unreadable_failure(occurrence: &Occurrence<'_>) -> String {
    match &occurrence.record.installed.validation {
        PackageValidation::Invalid { failure } | PackageValidation::Unsupported { failure } => {
            failure.message.clone()
        }
        PackageValidation::Valid | PackageValidation::Pending => {
            "the GGUF header could not be read".to_owned()
        }
    }
}

fn file_stem(path: &Path) -> String {
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("local model")
        .to_owned()
}

fn project(
    snapshot: &InstalledPackageSnapshot,
    configuration: &LocalModelsConfiguration,
    catalog: &RecommendableModelCatalog,
    enrichment: &dyn CustomModelEnrichment,
    managed_root: &Path,
    architectures: &RecognizedArchitectures,
    catalog_drafters: &CatalogDrafterRevisions,
) -> LocalProjection {
    let occurrences = snapshot
        .records
        .values()
        .filter_map(|record| occurrence(managed_root, record))
        .collect::<Vec<_>>();
    let present = snapshot.records.values().fold(
        BTreeMap::<ModelPackageId, &InstalledModelPackage>::new(),
        |mut present, record| {
            present
                .entry(record.installed.package.id.clone())
                .and_modify(|selected| {
                    if origin_precedence(record.installed.origin)
                        < origin_precedence(selected.origin)
                    {
                        *selected = &record.installed;
                    }
                })
                .or_insert(&record.installed);
            present
        },
    );
    let catalog_content = occurrences
        .iter()
        .filter(|occurrence| {
            matches!(
                occurrence.record.installed.catalog_attribution,
                InstalledCatalogAttribution::Attributed { .. }
            )
        })
        .filter_map(Occurrence::content_key)
        .collect::<BTreeSet<_>>();

    let mut projection = LocalProjection::default();
    let mut unsupported_paths = BTreeSet::new();
    let mut groups = BTreeMap::<(String, u64), Vec<&Occurrence<'_>>>::new();
    for occurrence in &occurrences {
        match (occurrence.evidence, occurrence.content_key()) {
            (Some(evidence), Some(key)) => {
                if matches!(evidence.role(architectures), GgufRole::Model { .. })
                    && !catalog_content.contains(&key)
                {
                    groups.entry(key).or_default().push(occurrence);
                }
            }
            _ => {
                if let Some(unsupported) =
                    unsupported_model(occurrence, configuration, &mut unsupported_paths)
                {
                    projection.unsupported.push(unsupported);
                }
            }
        }
    }
    for directory in &snapshot.unreadable_directories {
        if unsupported_paths.insert(directory.path.clone()) {
            projection.unsupported.push(unreadable_directory(directory));
        }
    }

    // One model per identity: the declared one, else the best-represented content.
    let mut identified =
        BTreeMap::<ModelId, (Option<&CustomModelDeclaration>, ContentGroup)>::new();
    for ((digest, _), mut members) in groups {
        members.sort_by_key(|occurrence| occurrence_rank(occurrence));
        let group = ContentGroup {
            digest,
            occurrences: members,
        };
        let declaration = group.declaration(&configuration.declarations);
        let Some(id) = declaration
            .map(|declaration| declaration.model_id.clone())
            .or_else(|| group.computed_id())
        else {
            continue;
        };
        let rank = |declared: bool, group: &ContentGroup| {
            (
                !declared,
                occurrence_rank(group.best()),
                group.digest.clone(),
            )
        };
        match identified.get(&id) {
            Some((existing_declaration, existing))
                if rank(existing_declaration.is_some(), existing)
                    <= rank(declaration.is_some(), &group) => {}
            _ => {
                identified.insert(id, (declaration, group));
            }
        }
    }

    let hidden = configuration.hidden.iter().collect::<BTreeSet<_>>();
    let context = ModelContext {
        configuration,
        catalog,
        catalog_drafters,
        enrichment,
        occurrences: &occurrences,
        present: &present,
    };
    for (model_id, (declaration, group)) in identified {
        let Some(projected) = context.model(model_id, declaration, &group) else {
            continue;
        };
        if hidden.contains(&projected.model.model_id) {
            projection.hidden.push(projected);
        } else {
            projection.models.push(projected);
        }
    }
    let listed = projection
        .models
        .iter()
        .chain(&projection.hidden)
        .map(|projected| projected.model.model_id.clone())
        .collect::<BTreeSet<_>>();
    for declaration in &configuration.declarations {
        if listed.contains(&declaration.model_id) {
            continue;
        }
        let Some(projected) = context.pending_added(declaration) else {
            continue;
        };
        if hidden.contains(&projected.model.model_id) {
            projection.hidden.push(projected);
        } else {
            projection.models.push(projected);
        }
    }
    projection
}

fn unsupported_model(
    occurrence: &Occurrence<'_>,
    configuration: &LocalModelsConfiguration,
    seen: &mut BTreeSet<PathBuf>,
) -> Option<UnsupportedLocalModel> {
    let declared =
        configuration
            .declarations
            .iter()
            .find_map(|declaration| match &declaration.origin {
                CustomModelDeclarationOrigin::Imported { path }
                    if declaration.target.first().is_some_and(|target| {
                        same_file(&occurrence.primary, Path::new(target))
                    }) =>
                {
                    Some(path.clone())
                }
                _ => None,
            });
    let origin = match (declared, occurrence.found_source()) {
        (Some(path), _) => CustomModelOrigin::Imported { path },
        (None, Some(source)) => CustomModelOrigin::Found { source },
        (None, None) => return None,
    };
    if !seen.insert(occurrence.primary.clone()) {
        return None;
    }
    let detail = unreadable_failure(occurrence);
    Some(UnsupportedLocalModel {
        display_name: occurrence
            .ollama_name
            .map(str::to_owned)
            .unwrap_or_else(|| file_stem(&occurrence.primary)),
        origin,
        path: occurrence.primary.clone(),
        reasons: vec![UnsupportedReason::Known(unreadable_reason(occurrence))],
        detail,
    })
}

/// A directory inside a location that Magnitude can't read, listed by its path within the location
/// so the person sees which part of it is missing.
fn unreadable_directory(directory: &UnreadableDirectory) -> UnsupportedLocalModel {
    let within = directory
        .path
        .strip_prefix(&directory.root)
        .ok()
        .filter(|relative| !relative.as_os_str().is_empty())
        .unwrap_or(&directory.path);
    UnsupportedLocalModel {
        display_name: within.display().to_string(),
        origin: CustomModelOrigin::Found {
            source: directory.source.clone(),
        },
        path: directory.path.clone(),
        reasons: vec![UnsupportedReason::Known(KnownUnsupportedReason::Unreadable)],
        detail: directory.detail.clone(),
    }
}

fn unreadable_reason(occurrence: &Occurrence<'_>) -> KnownUnsupportedReason {
    match &occurrence.record.installed.validation {
        PackageValidation::Unsupported { failure } if failure.code == UNSUPPORTED_FORMAT_CODE => {
            KnownUnsupportedReason::UnsupportedFormat
        }
        _ => KnownUnsupportedReason::Unreadable,
    }
}

/// What every model's projection consults: the configuration, the catalog, the enrichment, every
/// occurrence (drafters resolve among them) and the installed packages by identity.
struct ModelContext<'a> {
    configuration: &'a LocalModelsConfiguration,
    catalog: &'a RecommendableModelCatalog,
    catalog_drafters: &'a CatalogDrafterRevisions,
    enrichment: &'a dyn CustomModelEnrichment,
    occurrences: &'a [Occurrence<'a>],
    present: &'a BTreeMap<ModelPackageId, &'a InstalledModelPackage>,
}

impl ModelContext<'_> {
    fn model(
        &self,
        model_id: ModelId,
        declaration: Option<&CustomModelDeclaration>,
        group: &ContentGroup<'_>,
    ) -> Option<ProjectedModel> {
        let ModelContext {
            configuration,
            catalog,
            catalog_drafters,
            enrichment,
            occurrences,
            present,
        } = self;
        let representative = group.best();
        let origin = match declaration.map(|declaration| &declaration.origin) {
            Some(CustomModelDeclarationOrigin::Added { repository, commit }) => {
                CustomModelOrigin::Added {
                    repository: repository.clone(),
                    commit: commit.clone(),
                }
            }
            Some(CustomModelDeclarationOrigin::Imported { path }) => {
                CustomModelOrigin::Imported { path: path.clone() }
            }
            // Material only in the managed store without a declaration is no longer a model anyone
            // added; it is not published.
            None => CustomModelOrigin::Found {
                source: group
                    .occurrences
                    .iter()
                    .find_map(|occurrence| occurrence.found_source())?,
            },
        };
        let installed = &representative.record.installed;
        let evidence = representative.evidence?;
        let hub = group.hub();
        let (presented_name, variant_label) = crate::candidate_files::model_presentation(
            &hub.map_or_else(
                || representative.primary.clone(),
                |hub| PathBuf::from(&hub.selector),
            ),
            hub.map(|hub| hub.repository.as_str()),
            evidence.name.as_deref(),
            evidence.quantization.as_deref(),
        );
        let display_name = match (representative.ollama_name, hub) {
            (Some(name), None) => name.to_owned(),
            _ => presented_name,
        };
        let facts = CustomModelFacts {
            model_id: &model_id,
            origin: &origin,
            display_name: &display_name,
            target: &installed.package,
            hub: group.hub(),
            ollama_name: group
                .occurrences
                .iter()
                .find_map(|occurrence| occurrence.ollama_name),
            name: evidence.name.as_deref(),
            architecture: evidence.architecture.as_deref(),
            base_models: &evidence.base_models,
        };
        let enriched = enrichment.enrich(&facts);
        let enrichment_known = enrichment.knows(&facts);

        let choice = configuration
            .drafters
            .iter()
            .find(|attachment| attachment.model_id == model_id)
            .map_or(DrafterChoice::Auto, |attachment| attachment.drafter.clone());
        let resolve = |reference: DrafterReference, method: SpeculativeMethod| {
            let content = enrichment.drafter_content(&reference);
            resolve_drafter(
                reference,
                method,
                content,
                catalog,
                catalog_drafters,
                occurrences,
            )
        };
        let mut automatic = enriched.drafters.iter().filter_map(|candidate| {
            Some(resolve(
                candidate.reference.clone(),
                candidate.method.clone()?,
            ))
        });
        // The drafter packages this model would use whatever its choice, so a turned-off drafter
        // stays referenced while the model is here: its recommended drafter, and the drafter of
        // every catalog model it matches, even when a drafter beside it is recommended instead.
        let separate = |resolved: ResolvedDrafter| match resolved.source? {
            SpeculativeDraftSource::Separate { draft } => Some(draft.id),
            SpeculativeDraftSource::Embedded => None,
        };
        let mut candidate_drafters = enriched
            .recommended_drafter()
            .and_then(|candidate| {
                separate(resolve(candidate.reference.clone(), candidate.method.clone()?))
            })
            .into_iter()
            .collect::<Vec<_>>();
        candidate_drafters.extend(enriched.catalog_drafters.iter().filter_map(|model_id| {
            let method = catalog
                .models
                .iter()
                .find(|definition| catalog_id(definition) == *model_id)
                .and_then(|definition| match &definition.configuration.bundle {
                    ServableModelBundle::SpeculativeDecoding { method, .. } => Some(method.clone()),
                    ServableModelBundle::Standalone { .. } => None,
                })?;
            separate(resolve(
                DrafterReference::Catalog {
                    model_id: model_id.clone(),
                },
                method,
            ))
        }));
        let drafter = match &choice {
            DrafterChoice::None => None,
            // The first automatic candidate on this computer now, so a recommended drafter that
            // went missing gives way to the next one that binds; with none here, the recommended
            // one, missing, which clients offer to add.
            DrafterChoice::Auto => automatic.next().map(|recommended| {
                if recommended.source.is_some() {
                    return recommended;
                }
                automatic
                    .find(|next| next.source.is_some())
                    .unwrap_or(recommended)
            }),
            DrafterChoice::Reference { reference, method } => {
                Some(resolve(reference.clone(), method.clone()))
            }
        };

        let profile = custom_profile(installed);
        let speculative = drafter
            .as_ref()
            .and_then(|drafter| Some((drafter.source.clone()?, drafter.attached.method.clone())));
        let bundle = match speculative {
            Some((draft_source, method)) => ServableModelBundle::SpeculativeDecoding {
                target: installed.package.clone(),
                draft_source,
                method,
            },
            None => ServableModelBundle::Standalone {
                package: installed.package.clone(),
            },
        };
        let serving = ModelServingConfiguration {
            bundle,
            profile: profile.clone(),
        };
        let mut installation = resolved_installation(installed);
        // An added model whose same bytes are also in another location is partly Magnitude's.
        if matches!(origin, CustomModelOrigin::Added { .. })
            && group
                .occurrences
                .iter()
                .any(|occurrence| occurrence.origin() != ModelPackageInstallationOrigin::Magnitude)
            && group
                .occurrences
                .iter()
                .any(|occurrence| occurrence.origin() == ModelPackageInstallationOrigin::Magnitude)
        {
            let ResolvedModelInstallation::Resolved { ownership, .. } = &mut installation;
            *ownership = ModelInstallationOwnership::Mixed;
        }
        let (state, configuration) = match effective_configuration_model(&serving, present) {
            EffectiveModel::Ready { model } => (
                CustomModelState::Ready {
                    installation,
                    model,
                    catalog_attribution: match &installed.catalog_attribution {
                        InstalledCatalogAttribution::Failed { failure } => {
                            CustomModelCatalogAttribution::Failed {
                                failure: failure.clone(),
                            }
                        }
                        InstalledCatalogAttribution::NotCatalogTarget
                        | InstalledCatalogAttribution::Attributed { .. } => {
                            CustomModelCatalogAttribution::NotInCatalog
                        }
                    },
                },
                Ok(serving),
            ),
            EffectiveModel::Unavailable { failure } => (
                CustomModelState::Unavailable {
                    installation,
                    failure: failure.clone(),
                },
                Err(failure),
            ),
        };

        let file = |role: ModelFileRole| {
            installed
                .package
                .files
                .iter()
                .filter(move |file| file.role == role)
                .map(|file| CustomModelFile {
                    path: representative.base.join(&file.path),
                    size_bytes: file.size_bytes,
                })
        };
        let files = file(ModelFileRole::Weights).collect::<Vec<_>>();
        let projector = file(ModelFileRole::Projector).next();
        let size_bytes = files
            .iter()
            .chain(projector.iter())
            .map(|file| file.size_bytes)
            .sum();
        Some(ProjectedModel {
            locations: group
                .occurrences
                .iter()
                .map(|occurrence| occurrence.primary.clone())
                .collect(),
            lineage_repository: group.lineage_repository(),
            candidate_drafters,
            drafter_pending: matches!(choice, DrafterChoice::Auto) && !enrichment_known,
            awaited_drafter: None,
            managed_packages: group
                .occurrences
                .iter()
                .filter(|occurrence| {
                    occurrence.origin() == ModelPackageInstallationOrigin::Magnitude
                })
                .map(|occurrence| occurrence.record.installed.package.id.clone())
                .collect(),
            model: CustomModel {
                model_id,
                origin,
                display_name,
                variant_label,
                variant_of: enriched.variant_of.clone(),
                files,
                size_bytes,
                projector,
                drafter: drafter.map(|drafter| drafter.attached),
                // A model whose drafter is turned off is offered none.
                recommended_drafter: enriched
                    .recommended_drafter()
                    .filter(|_| choice != DrafterChoice::None)
                    .cloned(),
                license: enriched.license.or_else(|| evidence.license.clone()),
                profile,
                state,
                update_commit: None,
            },
            configuration,
        })
    }
}

impl ModelContext<'_> {
    /// An added model whose target is not in the store yet: its row exists from the declaration,
    /// so its download shows as installation progress, and it cannot be served until it lands.
    fn pending_added(&self, declaration: &CustomModelDeclaration) -> Option<ProjectedModel> {
        let CustomModelDeclarationOrigin::Added { repository, commit } = &declaration.origin else {
            return None;
        };
        let primary = declaration.target.first()?;
        let resolved = self
            .configuration
            .drafters
            .iter()
            .find(|attachment| attachment.model_id == declaration.model_id)
            .and_then(|attachment| match &attachment.drafter {
                DrafterChoice::Reference { reference, method } => Some(resolve_drafter(
                    reference.clone(),
                    method.clone(),
                    self.enrichment.drafter_content(reference),
                    self.catalog,
                    self.catalog_drafters,
                    self.occurrences,
                )),
                DrafterChoice::Auto | DrafterChoice::None => None,
            });
        // The drafter it downloads with its target is in use while the target is still arriving.
        let awaited_drafter = resolved
            .as_ref()
            .and_then(|resolved| match &resolved.source {
                Some(SpeculativeDraftSource::Separate { draft }) => Some(draft.id.clone()),
                _ => None,
            });
        let drafter = resolved.map(|resolved| resolved.attached);
        let (display_name, variant_label) = crate::candidate_files::model_presentation(
            Path::new(primary),
            Some(repository.as_str()),
            None,
            None,
        );
        Some(ProjectedModel {
            locations: Vec::new(),
            lineage_repository: Some(repository.as_str().to_owned()),
            candidate_drafters: Vec::new(),
            model: CustomModel {
                model_id: declaration.model_id.clone(),
                origin: CustomModelOrigin::Added {
                    repository: repository.clone(),
                    commit: commit.clone(),
                },
                display_name,
                variant_label,
                variant_of: None,
                files: declaration
                    .target
                    .iter()
                    .map(|path| CustomModelFile {
                        path: PathBuf::from(path),
                        size_bytes: 0,
                    })
                    .collect(),
                size_bytes: 0,
                projector: None,
                drafter,
                recommended_drafter: None,
                license: None,
                profile: ServingProfile {
                    context_length: DEFAULT_EXTERNAL_CONTEXT,
                },
                state: CustomModelState::NotInstalled,
                update_commit: None,
            },
            configuration: Err(ModelFailure {
                code: "model_not_installed".to_owned(),
                message: format!("{} has not finished downloading", declaration.model_id),
                retryable: true,
            }),
            managed_packages: Vec::new(),
            drafter_pending: false,
            awaited_drafter,
        })
    }
}

/// The separate drafter package a bundle serves with.
pub(crate) fn separate_draft(bundle: &ServableModelBundle) -> Option<ModelPackageId> {
    match bundle {
        ServableModelBundle::SpeculativeDecoding {
            draft_source: SpeculativeDraftSource::Separate { draft },
            ..
        } => Some(draft.id.clone()),
        _ => None,
    }
}

/// The drafter packages each catalog model has had, earlier revisions included.
type CatalogDrafterRevisions = BTreeMap<ModelId, BTreeSet<ModelPackageId>>;

struct ResolvedDrafter {
    attached: AttachedDrafter,
    source: Option<SpeculativeDraftSource>,
}

fn rejected(
    reference: DrafterReference,
    method: SpeculativeMethod,
    detail: String,
) -> ResolvedDrafter {
    ResolvedDrafter {
        attached: AttachedDrafter {
            reference,
            method,
            state: AttachedDrafterState::Rejected {
                reasons: vec![UnsupportedReason::Known(
                    KnownUnsupportedReason::TargetMismatch {
                        detail: detail.clone(),
                    },
                )],
                detail,
            },
            update_available: false,
        },
        source: None,
    }
}

/// Resolves a drafter reference against the files on this computer: the file it names, or any
/// file holding its bytes (`content`), wherever it is. A drafter that is not here yet leaves the
/// model serving without one until it is.
fn resolve_drafter(
    reference: DrafterReference,
    method: SpeculativeMethod,
    content: Option<DrafterContent>,
    catalog: &RecommendableModelCatalog,
    catalog_drafters: &CatalogDrafterRevisions,
    occurrences: &[Occurrence<'_>],
) -> ResolvedDrafter {
    let holding = |content: Option<&DrafterContent>, named: &dyn Fn(&Occurrence<'_>) -> bool| {
        occurrences
            .iter()
            .filter(|occurrence| {
                named(occurrence)
                    || content.is_some_and(|content| content.held_by(occurrence.record))
            })
            .min_by_key(|occurrence| occurrence.precedence)
    };
    let separate = |reference: DrafterReference,
                    method: SpeculativeMethod,
                    found: Option<&Occurrence<'_>>| match found {
        None => ResolvedDrafter {
            attached: AttachedDrafter {
                reference,
                method,
                state: AttachedDrafterState::NotInstalled,
                update_available: false,
            },
            source: None,
        },
        Some(occurrence) => match occurrence.usable() {
            Ok(()) => ResolvedDrafter {
                attached: AttachedDrafter {
                    reference,
                    method,
                    state: AttachedDrafterState::Ready,
                    update_available: false,
                },
                source: Some(SpeculativeDraftSource::Separate {
                    draft: occurrence.record.installed.package.clone(),
                }),
            },
            Err(failure) => ResolvedDrafter {
                attached: AttachedDrafter {
                    reference,
                    method,
                    state: AttachedDrafterState::Rejected {
                        reasons: vec![UnsupportedReason::Known(KnownUnsupportedReason::Unreadable)],
                        detail: failure.message,
                    },
                    update_available: false,
                },
                source: None,
            },
        },
    };
    // Built-in prediction heads are always present; whether the engine binds them is its drafter
    // verdict for the model's assessment, never decided here.
    let embedded = |reference: DrafterReference, method: SpeculativeMethod| ResolvedDrafter {
        attached: AttachedDrafter {
            reference,
            method,
            state: AttachedDrafterState::Ready,
            update_available: false,
        },
        source: Some(SpeculativeDraftSource::Embedded),
    };
    match &reference {
        DrafterReference::Embedded => embedded(reference, method),
        DrafterReference::Catalog { model_id } => {
            let Some(definition) = catalog
                .models
                .iter()
                .find(|definition| catalog_id(definition) == *model_id)
            else {
                return rejected(
                    reference.clone(),
                    method,
                    format!("{model_id} is not in the catalog"),
                );
            };
            match &definition.configuration.bundle {
                ServableModelBundle::SpeculativeDecoding {
                    draft_source: SpeculativeDraftSource::Separate { draft },
                    ..
                } => {
                    let content = DrafterContent::of_package(draft)
                        .map(|catalog_content| catalog_content.merge(content));
                    let current = holding(content.as_ref(), &|occurrence| {
                        occurrence.record.installed.package.id == draft.id
                    });
                    // A catalog drafter with a new revision keeps serving with the earlier
                    // revision of that catalog model's drafter until it is updated.
                    let earlier = || {
                        let revisions = catalog_drafters.get(model_id)?;
                        holding(None, &|occurrence| {
                            revisions.contains(&occurrence.record.installed.package.id)
                        })
                    };
                    match current {
                        Some(current) => separate(reference, method, Some(current)),
                        None => match earlier() {
                            Some(earlier) => {
                                let mut resolved = separate(reference, method, Some(earlier));
                                resolved.attached.update_available = true;
                                resolved
                            }
                            None => separate(reference, method, None),
                        },
                    }
                }
                ServableModelBundle::SpeculativeDecoding {
                    draft_source: SpeculativeDraftSource::Embedded,
                    ..
                } => embedded(reference, method),
                ServableModelBundle::Standalone { .. } => rejected(
                    reference.clone(),
                    method,
                    format!("{model_id} has no drafter"),
                ),
            }
        }
        DrafterReference::HuggingFace {
            repository,
            commit,
            path,
        } => {
            let found = holding(content.as_ref(), &|occurrence| {
                occurrence.hub.as_ref().is_some_and(|identity| {
                    identity.file.repository == repository.as_str()
                        && identity.file.selector == *path
                        && identity.commit.as_deref() == Some(commit.as_str())
                })
            });
            separate(reference, method, found)
        }
        DrafterReference::Local { path } => {
            let found = holding(None, &|occurrence| same_file(&occurrence.primary, path));
            separate(reference, method, found)
        }
    }
}

pub struct ManagedLocalModels {
    resolver: Arc<ModelDomainResolver>,
    remover: Arc<dyn CatalogPackageRemover>,
    watcher: Mutex<Option<RootWatcher>>,
    unwatched: Mutex<BTreeSet<PathBuf>>,
    readable: Mutex<BTreeMap<PathBuf, bool>>,
    this: Weak<Self>,
}

impl ManagedLocalModels {
    /// `remover` deletes pruned drafters with the residency's admission, which refuses a package
    /// a resident model has loaded.
    pub(crate) fn new(
        resolver: Arc<ModelDomainResolver>,
        remover: Arc<dyn CatalogPackageRemover>,
    ) -> Arc<Self> {
        let models = Arc::new_cyclic(|this| Self {
            resolver,
            remover,
            watcher: Mutex::new(None),
            unwatched: Mutex::new(BTreeSet::new()),
            readable: Mutex::new(BTreeMap::new()),
            this: this.clone(),
        });
        models.rewatch();
        models.observe_readable_roots();
        models.start_root_recheck();
        models
    }

    /// Records which enabled locations can be read now, the baseline `recheck_roots` compares to.
    fn observe_readable_roots(&self) {
        let Ok(configuration) = self.inventory().local_models_configuration() else {
            return;
        };
        if let Ok(mut readable) = self.readable.lock() {
            *readable = configuration
                .roots
                .iter()
                .filter(|root| root.enabled)
                .map(|root| (root.path.clone(), std::fs::read_dir(&root.path).is_ok()))
                .collect();
        }
    }

    /// Checks the enabled locations periodically: one that becomes readable, or appears, is scanned
    /// right away, and one whose watch could not start is rescanned every `UNWATCHED_RESCAN`.
    fn start_root_recheck(&self) {
        let Ok(runtime) = tokio::runtime::Handle::try_current() else {
            return;
        };
        let this = self.this.clone();
        runtime.spawn(async move {
            let mut interval = tokio::time::interval(crate::local_watch::ROOT_RECHECK);
            let mut last_unwatched_rescan = tokio::time::Instant::now();
            loop {
                interval.tick().await;
                let Some(models) = this.upgrade() else {
                    return;
                };
                let rescan_unwatched =
                    last_unwatched_rescan.elapsed() >= crate::local_watch::UNWATCHED_RESCAN;
                if rescan_unwatched {
                    last_unwatched_rescan = tokio::time::Instant::now();
                }
                models.recheck_roots(rescan_unwatched).await;
            }
        });
    }

    /// Rescans the enabled locations that became readable since the last check, and with
    /// `rescan_unwatched` the ones whose watch could not start.
    pub(crate) async fn recheck_roots(self: Arc<Self>, rescan_unwatched: bool) {
        let Ok(configuration) = self.inventory().local_models_configuration() else {
            return;
        };
        let mut affected = BTreeSet::new();
        if let Ok(mut readable) = self.readable.lock() {
            let enabled = configuration
                .roots
                .iter()
                .filter(|root| root.enabled)
                .map(|root| root.path.clone())
                .collect::<BTreeSet<_>>();
            readable.retain(|path, _| enabled.contains(path));
            for path in enabled {
                let now = std::fs::read_dir(&path).is_ok();
                if readable.insert(path.clone(), now) == Some(false) && now {
                    affected.insert(crate::inventory::canonical_or_self(&path));
                }
            }
        }
        if rescan_unwatched && let Ok(unwatched) = self.unwatched.lock() {
            affected.extend(unwatched.iter().cloned());
        }
        if !affected.is_empty() {
            self.rescan_after_change(affected).await;
        }
    }

    pub(crate) fn installed_packages_changed(&self, _revision: u64) {
        self.resolver.publish_local_change();
        self.resolver.upstream.wake();
    }

    fn inventory(&self) -> &ManagedModelStore {
        &self.resolver.inventory
    }

    /// Watches the enabled locations and declared files of the current configuration. A watch
    /// event rescans after the location has been quiet briefly, so a copy in progress settles
    /// into one reconciliation.
    fn rewatch(&self) {
        let Ok(configuration) = self.inventory().local_models_configuration() else {
            return;
        };
        let targets = watch_targets(&configuration);
        let Ok(mut watcher) = self.watcher.lock() else {
            return;
        };
        // A watch that could not start (an unreadable location, say) is retried on every rewatch.
        if watcher.as_ref().is_some_and(|watcher| {
            watcher.targets() == targets.as_slice() && watcher.unwatched().is_empty()
        }) {
            return;
        }
        let interests = targets
            .iter()
            .map(|target| target.interest.clone())
            .collect::<BTreeSet<_>>();
        let this = self.this.clone();
        *watcher = RootWatcher::start(targets, move |affected| {
            if let Some(models) = this.upgrade() {
                tokio::spawn(async move { models.rescan_after_change(affected).await });
            }
        });
        let unwatched = match watcher.as_ref() {
            Some(watcher) => watcher.unwatched().clone(),
            None => {
                tracing::warn!(
                    "model location watcher could not start; every location is rescanned periodically"
                );
                interests
            }
        };
        if let Ok(mut current) = self.unwatched.lock() {
            *current = unwatched;
        }
    }

    async fn rescan_after_change(self: Arc<Self>, affected: BTreeSet<PathBuf>) {
        if let Err(error) = self.inventory().rescan_model_roots(affected).await {
            tracing::warn!(%error, "model location rescan failed");
        }
        self.rewatch();
    }

    fn snapshot(&self) -> Result<LocalModelsResponse, InventoryError> {
        Ok(self.resolver.local_projection()?.response())
    }

    #[cfg(test)]
    pub(crate) fn watched(&self) -> Vec<crate::local_watch::WatchTarget> {
        self.watcher
            .lock()
            .ok()
            .and_then(|watcher| watcher.as_ref().map(|watcher| watcher.targets().to_vec()))
            .unwrap_or_default()
    }
}

impl LocalModels for ManagedLocalModels {
    fn configuration(&self) -> BoxFuture<'_, Result<LocalModelsConfiguration, InventoryError>> {
        Box::pin(async { self.inventory().local_models_configuration() })
    }

    fn configure(
        &self,
        configuration: LocalModelsConfiguration,
    ) -> BoxFuture<'_, Result<LocalModelsConfiguration, InventoryError>> {
        Box::pin(async move {
            if self
                .inventory()
                .configure_local_models(configuration.clone())
                .await?
            {
                self.resolver
                    .local
                    .configuration_generation
                    .fetch_add(1, Ordering::AcqRel);
                self.resolver
                    .local
                    .generation
                    .fetch_add(1, Ordering::AcqRel);
                self.resolver.publish_local_change();
                self.resolver.upstream.wake();
                self.rewatch();
            }
            Ok(configuration)
        })
    }

    fn list_local(&self) -> BoxFuture<'_, Result<LocalModelsResponse, InventoryError>> {
        Box::pin(async { self.snapshot() })
    }

    fn refresh_local(&self) -> BoxFuture<'_, Result<LocalModelsResponse, InventoryError>> {
        Box::pin(async move {
            self.inventory().rescan_model_inventory().await?;
            self.rewatch();
            self.snapshot()
        })
    }

    /// Deletes only the managed-store copy of an added model. Imported and found models are
    /// forgotten or hidden through the configuration; their files are never Magnitude's.
    fn remove_local(
        &self,
        model_id: &ModelId,
    ) -> BoxFuture<'_, Result<LocalModelRemoval, InventoryError>> {
        let model_id = model_id.clone();
        Box::pin(async move {
            use magnitude_service_contracts::models::InstalledModelPackages as _;
            let projection = self.resolver.local_projection()?;
            let projected = projection
                .models
                .iter()
                .chain(&projection.hidden)
                .find(|projected| projected.model.model_id == model_id)
                .ok_or_else(|| InventoryError::NotFound(model_id.to_string()))?;
            if !matches!(projected.model.origin, CustomModelOrigin::Added { .. }) {
                return Err(InventoryError::Unsupported(format!(
                    "{model_id} is not an added model; its files belong to their location"
                )));
            }
            let mut removed = false;
            let mut freed_bytes = 0_u64;
            for package_id in &projected.managed_packages {
                let removal = self.inventory().remove_installed(package_id).await?;
                removed |= removal.removed;
                freed_bytes = freed_bytes.saturating_add(removal.freed_bytes);
            }
            Ok(LocalModelRemoval {
                model_id,
                removed,
                freed_bytes,
            })
        })
    }

    fn prune_drafters(&self) -> BoxFuture<'_, Result<DrafterPruning, InventoryError>> {
        Box::pin(async move {
            self.inventory().ensure_installed_model_inventory().await?;
            let unused = self.resolver.unused_drafter_packages()?;
            let mut removed = Vec::new();
            let mut freed_bytes = 0_u64;
            for package_id in unused {
                if let Some(reason) = self.resolver.drafter_prune_refusal(&package_id)? {
                    tracing::warn!(package_id = %package_id.0, reason, "drafter pruning kept a package it may not delete");
                    continue;
                }
                match self.remover.remove_catalog_packages(vec![package_id.clone()]).await {
                    Ok(freed) => {
                        freed_bytes = freed_bytes.saturating_add(freed);
                        removed.push(package_id);
                    }
                    Err(InventoryError::Loaded(_)) => {}
                    Err(error) => return Err(error),
                }
            }
            Ok(DrafterPruning {
                removed,
                freed_bytes,
            })
        })
    }

    fn watch_local(&self) -> BoxStream<'static, ModelDomainInvalidation> {
        domain_changes(
            self.resolver.local_revision().unwrap_or_default(),
            self.resolver.local.changes.subscribe(),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::Duration;

    use magnitude_service_contracts::models::{
        CustomModelDeclaration, DrafterAttachment, DrafterCategory, DrafterLineage, DrafterVerdict,
        ModelAssessmentSubject, ModelRoot, ModelRootKind, RecommendableModel,
    };
    use magnitude_service_contracts::{ContentIdentity, ModelInventory as _};
    use sha2::{Digest, Sha256};

    use super::*;
    use crate::download::blob_key;
    use crate::inventory::InventoryConfig;
    use crate::test_gguf::{
        Meta, gguf_bytes, write_dflash_drafter, write_model, write_mtp_model, write_projector,
    };

    struct Fixture {
        _temporary: tempfile::TempDir,
        root: PathBuf,
        store: PathBuf,
        resolver: Arc<ModelDomainResolver>,
        local: Arc<ManagedLocalModels>,
    }

    impl Fixture {
        /// A fixture whose Hugging Face endpoint refuses every request.
        async fn new() -> Self {
            Self::with_hub("http://127.0.0.1:9".to_owned()).await
        }

        async fn with_hub(endpoint: String) -> Self {
            Self::open(tempfile::tempdir().unwrap(), endpoint, Vec::new(), BTreeMap::new()).await
        }

        /// `header_digests` are the release catalog's header digest of each catalog file, by its
        /// Hub SHA-256.
        async fn open(
            temporary: tempfile::TempDir,
            endpoint: String,
            catalog: Vec<RecommendableModel>,
            header_digests: BTreeMap<String, String>,
        ) -> Self {
            let root = temporary.path().canonicalize().unwrap();
            let store = root.join("store");
            let mut config =
                InventoryConfig::with_roots(store.clone(), root.join("cache")).unwrap();
            config.hub_endpoint = Some(endpoint);
            config.catalog_models = catalog.clone();
            config.catalog_header_digests = header_digests;
            config.architectures = crate::test_gguf::engine_architectures();
            let inventory = Arc::new(ManagedModelStore::open(config).await.unwrap());
            inventory.ensure_installed_model_inventory().await.unwrap();
            let resolver = ModelDomainResolver::new(
                inventory,
                RecommendableModelCatalog {
                    models: catalog,
                    diagnostics: Vec::new(),
                },
            );
            let local = ManagedLocalModels::new(
                resolver.clone(),
                Arc::new(StoreRemover(resolver.inventory.clone())),
            );
            Self {
                _temporary: temporary,
                root,
                store,
                resolver,
                local,
            }
        }

        async fn apply(&self, configuration: LocalModelsConfiguration) -> LocalModelsResponse {
            self.local.configure(configuration).await.unwrap();
            self.local.refresh_local().await.unwrap()
        }

        fn inventory(&self) -> &ManagedModelStore {
            &self.resolver.inventory
        }

        /// The same store opened again, as after a restart, with `catalog` as its catalog.
        async fn restart(self, catalog: Vec<RecommendableModel>) -> Self {
            self.restart_with_hub("http://127.0.0.1:9".to_owned(), catalog)
                .await
        }

        async fn restart_with_hub(
            self,
            endpoint: String,
            catalog: Vec<RecommendableModel>,
        ) -> Self {
            self.restart_with_catalog_headers(endpoint, catalog, BTreeMap::new())
                .await
        }

        async fn restart_with_catalog_headers(
            self,
            endpoint: String,
            catalog: Vec<RecommendableModel>,
            header_digests: BTreeMap<String, String>,
        ) -> Self {
            let Self {
                _temporary: temporary,
                resolver,
                local,
                ..
            } = self;
            drop(local);
            drop(resolver);
            Self::open(temporary, endpoint, catalog, header_digests).await
        }
    }

    fn root(kind: ModelRootKind, path: &Path) -> ModelRoot {
        ModelRoot {
            kind,
            path: path.to_path_buf(),
            enabled: true,
        }
    }

    fn roots(roots: Vec<ModelRoot>) -> LocalModelsConfiguration {
        LocalModelsConfiguration {
            roots,
            ..LocalModelsConfiguration::default()
        }
    }

    fn header_digest(path: &Path) -> String {
        crate::gguf::inspect(path).unwrap().header_digest
    }

    fn local_id(path: &Path) -> ModelId {
        ModelId::local(
            &LocalModelDigest::new(crate::gguf::local_model_digest_of(&[path]).unwrap()).unwrap(),
        )
    }

    fn ids(models: &[CustomModel]) -> Vec<String> {
        models
            .iter()
            .map(|model| model.model_id.to_string())
            .collect()
    }

    fn sha256(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    /// A Hugging Face hub-layout snapshot: blobs named by `blob_name`, linked from the snapshot.
    fn publish_snapshot(
        hub: &Path,
        repository: &str,
        commit: &str,
        files: &[(&str, Vec<u8>)],
        blob_name: impl Fn(&str) -> String,
    ) -> PathBuf {
        let repository_root = hub.join(hf_repo_dir(repository));
        let snapshot = repository_root.join("snapshots").join(commit);
        let blobs = repository_root.join("blobs");
        fs::create_dir_all(&blobs).unwrap();
        for (name, bytes) in files {
            let blob = blobs.join(blob_name(&sha256(bytes)));
            fs::write(&blob, bytes).unwrap();
            let link = snapshot.join(name);
            fs::create_dir_all(link.parent().unwrap()).unwrap();
            #[cfg(unix)]
            std::os::unix::fs::symlink(&blob, &link).unwrap();
            #[cfg(not(unix))]
            fs::hard_link(&blob, &link).unwrap();
        }
        fs::create_dir_all(repository_root.join("refs")).unwrap();
        fs::write(repository_root.join("refs/main"), commit).unwrap();
        snapshot
    }

    fn publish_managed(
        store: &Path,
        repository: &str,
        commit: &str,
        files: &[(&str, Vec<u8>)],
    ) -> PathBuf {
        publish_snapshot(&store.join("hub"), repository, commit, files, |digest| {
            blob_key(&ContentIdentity::Sha256 {
                value: digest.to_owned(),
            })
        })
    }

    fn publish_hf_cache(
        cache: &Path,
        repository: &str,
        commit: &str,
        files: &[(&str, Vec<u8>)],
    ) -> PathBuf {
        publish_snapshot(cache, repository, commit, files, str::to_owned)
    }

    fn model_bytes(name: &str) -> Vec<u8> {
        gguf_bytes(
            &[
                ("general.architecture", Meta::Str("llama")),
                ("general.name", Meta::Str(name)),
            ],
            &["token_embd.weight", "blk.0.attn_q.weight"],
        )
    }

    fn drafter_bytes(name: &str) -> Vec<u8> {
        gguf_bytes(
            &[
                ("general.architecture", Meta::Str("dflash")),
                ("general.name", Meta::Str(name)),
            ],
            &["blk.0.attn_q.weight"],
        )
    }

    #[tokio::test]
    async fn hugging_face_cache_models_use_their_hub_identity() {
        let fixture = Fixture::new().await;
        let cache = fixture.root.join("hf");
        publish_hf_cache(
            &cache,
            "owner/Model-GGUF",
            &"a".repeat(40),
            &[("Model-Q4_K_M.gguf", model_bytes("model"))],
        );

        let listed = fixture
            .apply(roots(vec![root(ModelRootKind::HuggingFaceCache, &cache)]))
            .await;

        assert_eq!(
            ids(&listed.models),
            ["hf:owner/Model-GGUF/Model-Q4_K_M.gguf"]
        );
        assert_eq!(
            listed.models[0].origin,
            CustomModelOrigin::Found {
                source: FoundSource::HuggingFaceCache
            }
        );
        assert!(matches!(
            listed.models[0].state,
            CustomModelState::Ready { .. }
        ));
    }

    /// The layout current `huggingface_hub` writes: the repository blob, named by the file's Hub
    /// hash, links into `<hub>/blobs/<xx>/<store hash>`, a store every repository shares.
    #[cfg(unix)]
    fn publish_shared_hf_cache(
        hub: &Path,
        repository: &str,
        commit: &str,
        files: &[(&str, Vec<u8>)],
    ) -> PathBuf {
        use std::os::unix::fs::symlink;

        let repository_root = hub.join(hf_repo_dir(repository));
        let snapshot = repository_root.join("snapshots").join(commit);
        let blobs = repository_root.join("blobs");
        fs::create_dir_all(&blobs).unwrap();
        for (name, bytes) in files {
            let hub_hash = sha256(bytes);
            let store_hash = sha256(hub_hash.as_bytes());
            let shard = hub.join("blobs").join(&store_hash[..2]);
            fs::create_dir_all(&shard).unwrap();
            fs::write(shard.join(&store_hash), bytes).unwrap();
            symlink(
                format!("../../blobs/{}/{store_hash}", &store_hash[..2]),
                blobs.join(&hub_hash),
            )
            .unwrap();
            let link = snapshot.join(name);
            fs::create_dir_all(link.parent().unwrap()).unwrap();
            symlink(format!("../../blobs/{hub_hash}"), &link).unwrap();
        }
        fs::create_dir_all(repository_root.join("refs")).unwrap();
        fs::write(repository_root.join("refs/main"), commit).unwrap();
        snapshot
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn shared_blob_hugging_face_caches_are_found_read_only_with_their_hub_identity() {
        let fixture = Fixture::new().await;
        let hub = fixture.root.join("hf/hub");
        let bytes = model_bytes("shared");
        let snapshot = publish_shared_hf_cache(
            &hub,
            "owner/Shared-GGUF",
            &"b".repeat(40),
            &[("Shared-Q4_K_M.gguf", bytes.clone())],
        );
        publish_hf_cache(
            &hub,
            "owner/Classic-GGUF",
            &"c".repeat(40),
            &[("Classic-Q4_K_M.gguf", model_bytes("classic"))],
        );

        let listed = fixture
            .apply(roots(vec![root(ModelRootKind::HuggingFaceCache, &hub)]))
            .await;

        assert_eq!(
            ids(&listed.models),
            [
                "hf:owner/Classic-GGUF/Classic-Q4_K_M.gguf",
                "hf:owner/Shared-GGUF/Shared-Q4_K_M.gguf"
            ]
        );
        let shared = listed
            .models
            .iter()
            .find(|model| model.model_id.as_str() == "hf:owner/Shared-GGUF/Shared-Q4_K_M.gguf")
            .unwrap();
        assert!(matches!(shared.state, CustomModelState::Ready { .. }));
        assert_eq!(shared.files[0].path, snapshot.join("Shared-Q4_K_M.gguf"));

        let inventory = fixture.inventory();
        let model = inventory
            .list()
            .await
            .unwrap()
            .into_iter()
            .find(|model| {
                matches!(
                    &model.source,
                    ModelSource::HuggingFace { repository, .. } if repository == "owner/Shared-GGUF"
                )
            })
            .expect("inventory publishes the shared-blob snapshot");
        assert_eq!(
            model.location.components()[0].content,
            ContentIdentity::Sha256 {
                value: sha256(&bytes)
            },
            "the repository blob's name, not the shared store's, identifies the content"
        );
        let resolved = inventory.resolve_ready(&model.id).await.unwrap();
        assert_eq!(
            resolved.components[0].path,
            snapshot.join("Shared-Q4_K_M.gguf")
        );
        assert!(!inventory.plan_delete(&model.id).await.unwrap().supported);
        assert!(inventory.delete(&model.id).await.is_err());
        assert!(snapshot.join("Shared-Q4_K_M.gguf").is_file());
    }

    #[tokio::test]
    async fn lm_studio_publisher_layout_gives_the_hub_identity_and_pairs_its_projector() {
        let fixture = Fixture::new().await;
        let lm_studio = fixture.root.join("lmstudio");
        let repository = lm_studio.join("lmstudio-community/Vision-GGUF");
        write_model(&repository.join("Vision-Q4_K_M.gguf"), "vision");
        write_projector(&repository.join("mmproj-Vision-F16.gguf"), "projector");
        write_model(&lm_studio.join("loose.gguf"), "loose");

        let listed = fixture
            .apply(roots(vec![root(ModelRootKind::LmStudio, &lm_studio)]))
            .await;

        let vision = listed
            .models
            .iter()
            .find(|model| {
                model.model_id.as_str() == "hf:lmstudio-community/Vision-GGUF/Vision-Q4_K_M.gguf"
            })
            .expect("publisher/repository layout gives the hub identity");
        assert_eq!(
            vision.origin,
            CustomModelOrigin::Found {
                source: FoundSource::LmStudio
            }
        );
        assert_eq!(
            vision.projector.as_ref().map(|file| file.path.clone()),
            Some(repository.join("mmproj-Vision-F16.gguf"))
        );
        assert_eq!(vision.files[0].path, repository.join("Vision-Q4_K_M.gguf"));
        assert!(
            ids(&listed.models).contains(&local_id(&lm_studio.join("loose.gguf")).to_string()),
            "a file outside the layout has no hub identity"
        );
    }

    #[tokio::test]
    async fn llama_cpp_names_parse_by_manifest_or_convention() {
        let fixture = Fixture::new().await;
        let llama = fixture.root.join("llama.cpp");
        write_model(
            &llama.join("ggml-org_gemma_3-1b_it_gemma-3-1b-it-Q4_K_M.gguf"),
            "manifest",
        );
        write_projector(
            &llama.join("ggml-org_gemma_3-1b_it_mmproj-model-f16.gguf"),
            "mmproj",
        );
        write_projector(&llama.join("other-projector.gguf"), "other");
        fs::write(
            llama.join("manifest=ggml-org=gemma_3-1b_it=latest.json"),
            serde_json::to_vec(&serde_json::json!({
                "ggufFile": { "rfilename": "gemma-3-1b-it-Q4_K_M.gguf" },
                "mmprojFile": { "rfilename": "mmproj-model-f16.gguf" }
            }))
            .unwrap(),
        )
        .unwrap();
        write_model(
            &llama.join("bartowski_Qwen3-8B-GGUF_Qwen3-8B-Q4_K_M.gguf"),
            "convention",
        );
        write_model(&llama.join("plain-name.gguf"), "plain");
        fs::write(
            llama.join("unsloth_x-GGUF_x.gguf.downloadInProgress"),
            b"partial",
        )
        .unwrap();

        let listed = fixture
            .apply(roots(vec![root(ModelRootKind::LlamaCpp, &llama)]))
            .await;

        let listed_ids = ids(&listed.models);
        assert!(
            listed_ids.contains(&"hf:ggml-org/gemma_3-1b_it/gemma-3-1b-it-Q4_K_M.gguf".to_owned())
        );
        assert!(listed_ids.contains(&"hf:bartowski/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf".to_owned()));
        assert!(listed_ids.contains(&local_id(&llama.join("plain-name.gguf")).to_string()));
        assert_eq!(listed.models.len(), 3);
        let gemma = listed
            .models
            .iter()
            .find(|model| model.model_id.as_str().contains("gemma"))
            .unwrap();
        assert_eq!(
            gemma.projector.as_ref().map(|file| file.path.clone()),
            Some(llama.join("ggml-org_gemma_3-1b_it_mmproj-model-f16.gguf")),
            "the manifest pairs the projector although the directory holds two"
        );
        assert_eq!(gemma.display_name, "gemma 3 1b it");
        assert_eq!(gemma.variant_label.as_deref(), Some("Q4_K_M"));
        let qwen = listed
            .models
            .iter()
            .find(|model| model.model_id.as_str().contains("Qwen3-8B"))
            .unwrap();
        assert_eq!(qwen.display_name, "Qwen3 8B");
        assert_eq!(qwen.variant_label.as_deref(), Some("Q4_K_M"));
    }

    /// Current llama.cpp writes `-hf` downloads in the Hugging Face hub layout: in `LLAMA_CACHE`
    /// itself or its `hub` folder they are llama.cpp's, keep their Hub identity and commit, and are
    /// never also listed as flat llama.cpp files. The same layout in the shared Hugging Face cache
    /// stays the Hugging Face cache's.
    #[tokio::test]
    async fn llama_cpp_hub_layout_downloads_are_llama_cpp_models_with_their_hub_identity() {
        let fixture = Fixture::new().await;
        let llama = fixture.root.join("llama.cpp");
        publish_hf_cache(
            &llama,
            "owner/Model-GGUF",
            &"a".repeat(40),
            &[("Model-Q4_K_M.gguf", model_bytes("in-cache"))],
        );
        publish_hf_cache(
            &llama.join("hub"),
            "owner/Other-GGUF",
            &"b".repeat(40),
            &[("Other-Q8_0.gguf", model_bytes("in-hub"))],
        );
        write_model(
            &llama.join("bartowski_Qwen3-8B-GGUF_Qwen3-8B-Q4_K_M.gguf"),
            "flat",
        );
        let shared = fixture.root.join("hf");
        publish_hf_cache(
            &shared,
            "owner/Shared-GGUF",
            &"c".repeat(40),
            &[("Shared-Q4_K_M.gguf", model_bytes("shared"))],
        );

        let listed = fixture
            .apply(roots(vec![
                root(ModelRootKind::LlamaCpp, &llama),
                root(ModelRootKind::HuggingFaceCache, &shared),
            ]))
            .await;

        let source = |id: &str| {
            listed
                .models
                .iter()
                .find(|model| model.model_id.as_str() == id)
                .map(|model| model.origin.clone())
        };
        let llama_cpp = Some(CustomModelOrigin::Found {
            source: FoundSource::LlamaCpp,
        });
        assert_eq!(listed.models.len(), 4, "{:?}", ids(&listed.models));
        assert_eq!(source("hf:owner/Model-GGUF/Model-Q4_K_M.gguf"), llama_cpp);
        assert_eq!(source("hf:owner/Other-GGUF/Other-Q8_0.gguf"), llama_cpp);
        assert_eq!(
            source("hf:bartowski/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"),
            llama_cpp
        );
        assert_eq!(
            source("hf:owner/Shared-GGUF/Shared-Q4_K_M.gguf"),
            Some(CustomModelOrigin::Found {
                source: FoundSource::HuggingFaceCache
            })
        );
    }

    fn write_ollama(
        root: &Path,
        registry: &str,
        namespace: &str,
        model: &str,
        tag: &str,
        layers: &[(&str, &[u8])],
    ) -> Vec<u8> {
        let blobs = root.join("blobs");
        fs::create_dir_all(&blobs).unwrap();
        let layers = layers
            .iter()
            .map(|(media_type, bytes)| {
                let digest = sha256(bytes);
                fs::write(blobs.join(format!("sha256-{digest}")), bytes).unwrap();
                serde_json::json!({
                    "mediaType": media_type,
                    "digest": format!("sha256:{digest}"),
                    "size": bytes.len()
                })
            })
            .collect::<Vec<_>>();
        let manifest = root
            .join("manifests")
            .join(registry)
            .join(namespace)
            .join(model)
            .join(tag);
        fs::create_dir_all(manifest.parent().unwrap()).unwrap();
        let document =
            serde_json::to_vec(&serde_json::json!({ "schemaVersion": 2, "layers": layers }))
                .unwrap();
        fs::write(manifest, &document).unwrap();
        document
    }

    /// Ollama's newer `manifests-v2` entry: a symlink to the manifest stored as a blob.
    #[cfg(unix)]
    fn write_ollama_v2(root: &Path, registry: &str, path: &str, document: &[u8]) {
        let blob = format!("sha256-{}", sha256(document));
        fs::write(root.join("blobs").join(&blob), document).unwrap();
        let manifest = root.join("manifests-v2").join(registry).join(path);
        fs::create_dir_all(manifest.parent().unwrap()).unwrap();
        std::os::unix::fs::symlink(Path::new("../../../../blobs").join(blob), manifest).unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn ollama_manifests_v2_are_read_and_a_pull_in_both_trees_is_one_model() {
        let fixture = Fixture::new().await;
        let ollama = fixture.root.join("ollama");
        let qwen = model_bytes("qwen library");
        let document = write_ollama(
            &ollama,
            "registry.ollama.ai",
            "library",
            "qwen3.5",
            "0.8b",
            &[("application/vnd.ollama.image.model", &qwen)],
        );
        write_ollama_v2(&ollama, "ollama.com", "library/qwen3.5/0.8b", &document);
        let pulled = model_bytes("hub pull v2");
        let hub_document = write_ollama(
            &ollama,
            "hf.co",
            "unsloth",
            "Qwen3.5-0.8B-GGUF",
            "Q4_K_M",
            &[("application/vnd.ollama.image.model", &pulled)],
        );
        fs::remove_dir_all(ollama.join("manifests/hf.co")).unwrap();
        write_ollama_v2(
            &ollama,
            "hf.co",
            "unsloth/Qwen3.5-0.8B-GGUF/Q4_K_M",
            &hub_document,
        );

        let listed = fixture
            .apply(roots(vec![root(ModelRootKind::Ollama, &ollama)]))
            .await;

        let mut names = listed
            .models
            .iter()
            .map(|model| model.display_name.as_str())
            .collect::<Vec<_>>();
        names.sort();
        assert_eq!(
            names,
            ["hf.co/unsloth/Qwen3.5-0.8B-GGUF:Q4_K_M", "qwen3.5:0.8b"]
        );
    }

    /// A store a newer Ollama wrote only `manifests-v2/` in, with no `manifests/` at all, is read.
    #[cfg(unix)]
    #[tokio::test]
    async fn an_ollama_store_with_only_manifests_v2_is_scanned() {
        let fixture = Fixture::new().await;
        let ollama = fixture.root.join("ollama");
        let qwen = model_bytes("qwen v2 only");
        let document = write_ollama(
            &ollama,
            "registry.ollama.ai",
            "library",
            "qwen3.5",
            "0.8b",
            &[("application/vnd.ollama.image.model", &qwen)],
        );
        fs::remove_dir_all(ollama.join("manifests")).unwrap();
        write_ollama_v2(&ollama, "ollama.com", "library/qwen3.5/0.8b", &document);

        let listed = fixture
            .apply(roots(vec![root(ModelRootKind::Ollama, &ollama)]))
            .await;
        assert_eq!(
            listed
                .models
                .iter()
                .map(|model| model.display_name.as_str())
                .collect::<Vec<_>>(),
            ["qwen3.5:0.8b"]
        );
    }

    /// A model Ollama 0.40 stores for its MLX runner, one safetensors blob per tensor, can't be
    /// read in place, so it is listed under "can't run" by its Ollama name with the format as its
    /// reason, beside the store's GGUF pulls.
    #[cfg(unix)]
    #[tokio::test]
    async fn an_ollama_model_stored_as_mlx_tensors_is_unsupported_by_its_format() {
        let fixture = Fixture::new().await;
        let ollama = fixture.root.join("ollama");
        crate::local_roots::write_ollama_mlx_pull(&ollama, "qwen3.5", "4b", "qwen");
        let llama = model_bytes("llama gguf");
        write_ollama(
            &ollama,
            "registry.ollama.ai",
            "library",
            "llama3.2",
            "1b",
            &[("application/vnd.ollama.image.model", &llama)],
        );

        let listed = fixture
            .apply(roots(vec![root(ModelRootKind::Ollama, &ollama)]))
            .await;

        assert_eq!(
            listed
                .models
                .iter()
                .map(|model| model.display_name.as_str())
                .collect::<Vec<_>>(),
            ["llama3.2:1b"]
        );
        assert_eq!(listed.unsupported.len(), 1);
        let unsupported = &listed.unsupported[0];
        assert_eq!(unsupported.display_name, "qwen3.5:4b");
        assert_eq!(
            unsupported.origin,
            CustomModelOrigin::Found {
                source: FoundSource::Ollama
            }
        );
        assert_eq!(
            unsupported.reasons,
            [UnsupportedReason::Known(
                KnownUnsupportedReason::UnsupportedFormat
            )]
        );
        assert!(unsupported.detail.contains("safetensors"));
    }

    /// A version added from a catalog model's repository that isn't the catalog's exact target is
    /// a custom model, even when its intrinsic identity (`general.name` and file type) is the
    /// catalog target's, as a Q4_K_M beside a catalog UD-Q4_K_XL is: once its files are in
    /// Magnitude's store it is installed under its own ID, across a restart, and the catalog
    /// model stays not installed.
    #[tokio::test]
    async fn an_added_non_target_version_from_a_catalog_repository_is_an_installed_custom_model() {
        use magnitude_service_contracts::models::{CatalogModelState, InstalledModelPackages as _};
        let quant = |description: &str| {
            gguf_bytes(
                &[
                    ("general.architecture", Meta::Str("llama")),
                    ("general.name", Meta::Str("Model")),
                    ("general.description", Meta::Str(description)),
                    ("general.file_type", Meta::U32(15)),
                ],
                &["token_embd.weight", "blk.0.attn_q.weight"],
            )
        };
        let repository = "owner/Model-GGUF";
        let commit = "a".repeat(40);

        let release = Fixture::new().await;
        publish_managed(
            &release.store,
            repository,
            &commit,
            &[("Model-UD-Q4_K_XL.gguf", quant("catalog target"))],
        );
        release.apply(LocalModelsConfiguration::default()).await;
        let target = release.inventory().list_installed().await.unwrap().packages[0]
            .package
            .clone();
        assert!(target.properties.intrinsic_quality_id.is_some());
        let catalog = vec![catalog_definition(target)];

        let fixture = Fixture::new().await;
        publish_managed(
            &fixture.store,
            repository,
            &commit,
            &[("Model-Q4_K_M.gguf", quant("another version"))],
        );
        let id: ModelId = "hf:owner/Model-GGUF/Model-Q4_K_M.gguf".parse().unwrap();
        let configuration = LocalModelsConfiguration {
            declarations: vec![CustomModelDeclaration {
                model_id: id.clone(),
                origin: CustomModelDeclarationOrigin::Added {
                    repository: HuggingFaceRepositoryId::new(repository).unwrap(),
                    commit: commit.clone(),
                },
                target: vec!["Model-Q4_K_M.gguf".to_owned()],
                projector: None,
            }],
            ..LocalModelsConfiguration::default()
        };
        let mut fixture = fixture.restart(catalog.clone()).await;
        for restarted in [false, true] {
            if restarted {
                fixture = fixture.restart(catalog.clone()).await;
            }
            let listed = fixture.apply(configuration.clone()).await;
            let model = listed
                .models
                .iter()
                .find(|model| model.model_id == id)
                .unwrap_or_else(|| panic!("the added model is listed (restarted: {restarted}): {listed:#?}"));
            assert!(
                matches!(model.state, CustomModelState::Ready { .. }),
                "the added model is installed (restarted: {restarted}): {:?}",
                model.state
            );
            assert!(
                matches!(
                    fixture.resolver.catalog_snapshot().unwrap().models[0].local_state,
                    CatalogModelState::NotInstalled
                ),
                "the catalog model is not installed (restarted: {restarted})"
            );
        }
    }

    /// An added model whose same bytes are also in another tool's location is partly Magnitude's:
    /// its ownership is Mixed, and Magnitude alone otherwise.
    #[tokio::test]
    async fn an_added_model_also_found_elsewhere_has_mixed_ownership() {
        use magnitude_service_contracts::models::ModelInstallationOwnership;
        let fixture = Fixture::new().await;
        let commit = "c".repeat(40);
        let repository = "owner/Added-GGUF";
        let files = [("Added-Q4_K_M.gguf", model_bytes("added and found"))];
        publish_managed(&fixture.store, repository, &commit, &files);
        let cache = fixture.root.join("hf");
        let id: ModelId = "hf:owner/Added-GGUF/Added-Q4_K_M.gguf".parse().unwrap();
        let configuration = LocalModelsConfiguration {
            roots: vec![root(ModelRootKind::HuggingFaceCache, &cache)],
            declarations: vec![CustomModelDeclaration {
                model_id: id.clone(),
                origin: CustomModelDeclarationOrigin::Added {
                    repository: HuggingFaceRepositoryId::new(repository).unwrap(),
                    commit: commit.clone(),
                },
                target: vec!["Added-Q4_K_M.gguf".to_owned()],
                projector: None,
            }],
            ..LocalModelsConfiguration::default()
        };
        let ownership = |listed: &LocalModelsResponse| {
            let model = listed
                .models
                .iter()
                .find(|model| model.model_id == id)
                .unwrap();
            match &model.state {
                CustomModelState::Ready { installation, .. }
                | CustomModelState::Unavailable { installation, .. } => {
                    let ResolvedModelInstallation::Resolved { ownership, .. } = installation;
                    *ownership
                }
                CustomModelState::NotInstalled => panic!("the added model is installed"),
            }
        };

        let alone = fixture.apply(configuration.clone()).await;
        assert_eq!(ownership(&alone), ModelInstallationOwnership::Magnitude);

        publish_hf_cache(&cache, repository, &commit, &files);
        let both = fixture.local.refresh_local().await.unwrap();
        assert_eq!(ids(&both.models), [id.to_string()]);
        assert_eq!(ownership(&both), ModelInstallationOwnership::Mixed);
    }

    #[tokio::test]
    async fn several_ollama_roots_are_each_scanned_and_a_pull_in_both_is_one_model() {
        let fixture = Fixture::new().await;
        let user = fixture.root.join("home/.ollama/models");
        let system = fixture.root.join("var/lib/ollama");
        let shared = model_bytes("pulled by both installs");
        for root in [&user, &system] {
            write_ollama(
                root,
                "registry.ollama.ai",
                "library",
                "qwen3.5",
                "0.8b",
                &[("application/vnd.ollama.image.model", &shared)],
            );
        }
        write_ollama(
            &user,
            "registry.ollama.ai",
            "library",
            "llama3",
            "8b",
            &[(
                "application/vnd.ollama.image.model",
                &model_bytes("user only"),
            )],
        );
        write_ollama(
            &system,
            "registry.ollama.ai",
            "library",
            "gemma3",
            "4b",
            &[(
                "application/vnd.ollama.image.model",
                &model_bytes("system only"),
            )],
        );

        let listed = fixture
            .apply(roots(vec![
                root(ModelRootKind::Ollama, &user),
                root(
                    ModelRootKind::Ollama,
                    &fixture.root.join("usr/share/ollama/.ollama/models"),
                ),
                root(ModelRootKind::Ollama, &system),
            ]))
            .await;

        let mut names = listed
            .models
            .iter()
            .map(|model| model.display_name.as_str())
            .collect::<Vec<_>>();
        names.sort();
        assert_eq!(names, ["gemma3:4b", "llama3:8b", "qwen3.5:0.8b"]);
    }

    #[tokio::test]
    async fn several_hugging_face_caches_are_each_scanned() {
        let fixture = Fixture::new().await;
        let configured = fixture.root.join("custom/hub");
        let default = fixture.root.join("home/.cache/huggingface/hub");
        let shared = model_bytes("in both caches");
        publish_hf_cache(
            &configured,
            "owner/First-GGUF",
            &"a".repeat(40),
            &[
                ("first.gguf", model_bytes("first")),
                ("shared.gguf", shared.clone()),
            ],
        );
        publish_hf_cache(
            &default,
            "owner/Second-GGUF",
            &"b".repeat(40),
            &[("second.gguf", model_bytes("second"))],
        );
        publish_hf_cache(
            &default,
            "owner/First-GGUF",
            &"a".repeat(40),
            &[("shared.gguf", shared)],
        );

        let listed = fixture
            .apply(roots(vec![
                root(ModelRootKind::HuggingFaceCache, &configured),
                root(ModelRootKind::HuggingFaceCache, &default),
            ]))
            .await;

        let mut listed_ids = ids(&listed.models);
        listed_ids.sort();
        assert_eq!(
            listed_ids,
            [
                "hf:owner/First-GGUF/first.gguf",
                "hf:owner/First-GGUF/shared.gguf",
                "hf:owner/Second-GGUF/second.gguf",
            ]
        );
    }

    #[tokio::test]
    async fn ollama_manifests_name_their_blobs() {
        let fixture = Fixture::new().await;
        let ollama = fixture.root.join("ollama");
        let library = model_bytes("llama library");
        let pulled = model_bytes("hub pull");
        let projector = gguf_bytes(
            &[("general.architecture", Meta::Str("clip"))],
            &["v.patch_embd.weight"],
        );
        write_ollama(
            &ollama,
            "registry.ollama.ai",
            "library",
            "llama3",
            "8b",
            &[
                ("application/vnd.ollama.image.model", &library),
                ("application/vnd.ollama.image.projector", &projector),
                ("application/vnd.ollama.image.template", b"{{ .Prompt }}"),
            ],
        );
        write_ollama(
            &ollama,
            "hf.co",
            "unsloth",
            "Qwen3-8B-GGUF",
            "Q4_K_M",
            &[("application/vnd.ollama.image.model", &pulled)],
        );
        // A pull in progress: the manifest names a blob that is not complete yet.
        let incomplete = model_bytes("incomplete");
        write_ollama(
            &ollama,
            "registry.ollama.ai",
            "library",
            "mistral",
            "latest",
            &[("application/vnd.ollama.image.model", &incomplete)],
        );
        let incomplete_blob = ollama.join(format!("blobs/sha256-{}", sha256(&incomplete)));
        fs::write(&incomplete_blob, &incomplete[..incomplete.len() / 2]).unwrap();

        let listed = fixture
            .apply(roots(vec![root(ModelRootKind::Ollama, &ollama)]))
            .await;

        assert_eq!(listed.models.len(), 2, "{:?}", ids(&listed.models));
        let library_model = listed
            .models
            .iter()
            .find(|model| model.display_name == "llama3:8b")
            .expect("library model named as Ollama names it");
        assert!(library_model.model_id.as_str().starts_with("local:"));
        assert_eq!(
            library_model.origin,
            CustomModelOrigin::Found {
                source: FoundSource::Ollama
            }
        );
        assert!(library_model.projector.is_some());
        let hub_pull = listed
            .models
            .iter()
            .find(|model| model.display_name == "hf.co/unsloth/Qwen3-8B-GGUF:Q4_K_M")
            .unwrap();
        assert_eq!(
            hub_pull.model_id,
            local_id(&ollama.join(format!("blobs/sha256-{}", sha256(&pulled)))),
            "an unresolved pull is identified by its header, never by Ollama's tag"
        );
        assert!(listed.unsupported.is_empty());
    }

    /// A Hub serving one repository's file listing at `main`.
    fn serve_repository(repository: &'static str, files: Vec<(&'static str, Vec<u8>)>) -> String {
        use std::io::{BufRead, BufReader, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let body = serde_json::to_vec(&serde_json::json!({
            "id": repository,
            "sha": "c".repeat(40),
            "tags": ["gguf"],
            "siblings": files.iter().map(|(path, bytes)| serde_json::json!({
                "rfilename": path,
                "size": bytes.len(),
                "lfs": { "sha256": sha256(bytes), "size": bytes.len() },
            })).collect::<Vec<_>>(),
        }))
        .unwrap();
        std::thread::spawn(move || {
            for connection in listener.incoming() {
                let Ok(mut connection) = connection else {
                    return;
                };
                let mut reader = BufReader::new(connection.try_clone().unwrap());
                let mut request_line = String::new();
                let _ = reader.read_line(&mut request_line);
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                        break;
                    }
                }
                let found =
                    request_line.contains(&format!("/api/models/{repository}/revision/main"));
                let (status, body) = if found {
                    ("200 OK", body.as_slice())
                } else {
                    ("404 Not Found", &b""[..])
                };
                let _ = write!(
                    connection,
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = connection.write_all(body);
            }
        });
        endpoint
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn an_ollama_hub_pull_keeps_its_local_id_and_takes_its_lineage_from_the_repository() {
        let pulled = model_bytes("hub pull");
        let endpoint = serve_repository(
            "unsloth/Qwen3-8B-GGUF",
            vec![
                ("Qwen3-8B-Q4_K_M.gguf", pulled.clone()),
                ("Qwen3-8B-Q8_0.gguf", model_bytes("other")),
            ],
        );
        let fixture = Fixture::with_hub(endpoint).await;
        let ollama = fixture.root.join("ollama");
        write_ollama(
            &ollama,
            "hf.co",
            "unsloth",
            "Qwen3-8B-GGUF",
            "Q4_K_M",
            &[("application/vnd.ollama.image.model", &pulled)],
        );
        let blob = ollama.join(format!("blobs/sha256-{}", sha256(&pulled)));
        let local = local_id(&blob).to_string();
        let listed = fixture
            .apply(roots(vec![root(ModelRootKind::Ollama, &ollama)]))
            .await;
        assert_eq!(
            ids(&listed.models),
            [local.clone()],
            "an Ollama model keeps its local: ID"
        );
        assert_eq!(
            listed.models[0].display_name,
            "hf.co/unsloth/Qwen3-8B-GGUF:Q4_K_M"
        );
        let model_id: ModelId = local.parse().unwrap();
        // Resolving the pull's file in its repository serves lineage only; the ID never changes.
        for _ in 0..20 {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            let rescanned = fixture.local.refresh_local().await.unwrap();
            assert_eq!(ids(&rescanned.models), [local.clone()]);
        }
        assert_eq!(
            fixture
                .resolver
                .custom_model_lineage_repository(&model_id)
                .unwrap()
                .as_deref(),
            Some("unsloth/Qwen3-8B-GGUF")
        );

        let hub = fixture.root.join("hf/hub");
        publish_hf_cache(
            &hub,
            "unsloth/Qwen3-8B-GGUF",
            &"c".repeat(40),
            &[("Qwen3-8B-Q4_K_M.gguf", pulled)],
        );
        let both = fixture
            .apply(roots(vec![
                root(ModelRootKind::Ollama, &ollama),
                root(ModelRootKind::HuggingFaceCache, &hub),
            ]))
            .await;
        assert_eq!(
            both.models.len(),
            1,
            "the same bytes in two locations are one model: {:?}",
            ids(&both.models)
        );
    }

    /// An `hf.co` pull whose repository can't be read (offline, or no file has its bytes) still
    /// takes its lineage from the repository its name gives.
    #[tokio::test]
    async fn an_unresolved_ollama_hub_pull_takes_its_lineage_from_its_name() {
        let fixture = Fixture::new().await;
        let ollama = fixture.root.join("ollama");
        let pulled = model_bytes("unresolved pull");
        write_ollama(
            &ollama,
            "hf.co",
            "unsloth",
            "Qwen3-8B-GGUF",
            "Q4_K_M",
            &[("application/vnd.ollama.image.model", &pulled)],
        );
        let listed = fixture
            .apply(roots(vec![root(ModelRootKind::Ollama, &ollama)]))
            .await;
        let model_id = listed.models[0].model_id.clone();
        assert!(model_id.as_str().starts_with("local:"));
        assert_eq!(
            fixture
                .resolver
                .custom_model_lineage_repository(&model_id)
                .unwrap()
                .as_deref(),
            Some("unsloth/Qwen3-8B-GGUF")
        );
        assert_eq!(ollama_hub_repository("qwen3:8b"), None);
        assert_eq!(ollama_hub_repository("someone/tool:latest"), None);
    }

    #[tokio::test]
    async fn folders_are_scanned_recursively_and_skip_partial_downloads() {
        let fixture = Fixture::new().await;
        let folder = fixture.root.join("models");
        write_model(&folder.join("a/b/deep.gguf"), "deep");
        fs::write(folder.join("copying.gguf.part"), b"partial").unwrap();
        fs::write(folder.join("model.gguf.incomplete"), b"partial").unwrap();

        let listed = fixture
            .apply(roots(vec![root(ModelRootKind::Folder, &folder)]))
            .await;

        assert_eq!(
            ids(&listed.models),
            [local_id(&folder.join("a/b/deep.gguf")).to_string()]
        );
        assert_eq!(
            listed.models[0].origin,
            CustomModelOrigin::Found {
                source: FoundSource::Folder {
                    path: folder.clone()
                }
            }
        );
        assert!(listed.unsupported.is_empty());
    }

    #[tokio::test]
    async fn the_same_bytes_in_two_places_are_one_model_represented_by_precedence() {
        let fixture = Fixture::new().await;
        let cache = fixture.root.join("hf");
        let folder = fixture.root.join("models");
        let bytes = model_bytes("shared");
        publish_hf_cache(
            &cache,
            "owner/Shared-GGUF",
            &"b".repeat(40),
            &[("shared.gguf", bytes.clone())],
        );
        fs::create_dir_all(&folder).unwrap();
        fs::write(folder.join("copy-of-shared.gguf"), &bytes).unwrap();

        let listed = fixture
            .apply(roots(vec![
                root(ModelRootKind::Folder, &folder),
                root(ModelRootKind::HuggingFaceCache, &cache),
            ]))
            .await;

        assert_eq!(ids(&listed.models), ["hf:owner/Shared-GGUF/shared.gguf"]);
        assert_eq!(
            listed.models[0].origin,
            CustomModelOrigin::Found {
                source: FoundSource::HuggingFaceCache
            }
        );
    }

    #[tokio::test]
    async fn a_moved_file_keeps_its_identity() {
        let fixture = Fixture::new().await;
        let first = fixture.root.join("first");
        let second = fixture.root.join("second");
        write_model(&first.join("model.gguf"), "moving");
        fs::create_dir_all(&second).unwrap();
        let id = local_id(&first.join("model.gguf"));
        let configuration = roots(vec![
            root(ModelRootKind::Folder, &first),
            root(ModelRootKind::Folder, &second),
        ]);

        let before = fixture.apply(configuration).await;
        fs::rename(first.join("model.gguf"), second.join("renamed.gguf")).unwrap();
        let after = fixture.local.refresh_local().await.unwrap();

        assert_eq!(ids(&before.models), [id.to_string()]);
        assert_eq!(ids(&after.models), [id.to_string()]);
        assert_eq!(after.models[0].files[0].path, second.join("renamed.gguf"));
    }

    /// A folder of links to model files kept elsewhere lists each file once, by its link.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_folder_link_to_a_file_elsewhere_is_found_by_its_link() {
        let fixture = Fixture::new().await;
        let target = fixture.root.join("elsewhere/model.gguf");
        let folder = fixture.root.join("models");
        write_model(&target, "linked");
        fs::create_dir_all(folder.join("links")).unwrap();
        let link = folder.join("links/model-link.gguf");
        std::os::unix::fs::symlink(&target, &link).unwrap();

        let listed = fixture
            .apply(roots(vec![root(ModelRootKind::Folder, &folder)]))
            .await;

        assert_eq!(ids(&listed.models), [local_id(&target).to_string()]);
        assert_eq!(listed.models[0].files[0].path, link);
        assert!(listed.unsupported.is_empty());

        std::os::unix::fs::symlink(&target, folder.join("second-link.gguf")).unwrap();
        let both = fixture
            .apply(roots(vec![
                root(ModelRootKind::Folder, &folder),
                root(ModelRootKind::Folder, &fixture.root.join("elsewhere")),
            ]))
            .await;
        assert_eq!(ids(&both.models), [local_id(&target).to_string()]);
    }

    /// A linked directory is never descended: following one could scan any tree on the disk.
    #[cfg(unix)]
    #[tokio::test]
    async fn a_folder_does_not_descend_into_a_linked_directory() {
        let fixture = Fixture::new().await;
        let elsewhere = fixture.root.join("elsewhere");
        let folder = fixture.root.join("models");
        write_model(&elsewhere.join("model.gguf"), "beyond");
        fs::create_dir_all(&folder).unwrap();
        std::os::unix::fs::symlink(&elsewhere, folder.join("linked-directory")).unwrap();

        let listed = fixture
            .apply(roots(vec![root(ModelRootKind::Folder, &folder)]))
            .await;

        assert!(listed.models.is_empty());
        assert!(listed.unsupported.is_empty());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn a_link_within_a_folder_to_a_file_in_it_is_one_model() {
        let fixture = Fixture::new().await;
        let folder = fixture.root.join("models");
        let target = folder.join("real/model.gguf");
        write_model(&target, "inside");
        fs::create_dir_all(folder.join("links")).unwrap();
        std::os::unix::fs::symlink("../real/model.gguf", folder.join("links/alias.gguf")).unwrap();

        let listed = fixture
            .apply(roots(vec![root(ModelRootKind::Folder, &folder)]))
            .await;

        assert_eq!(ids(&listed.models), [local_id(&target).to_string()]);
        assert!(listed.unsupported.is_empty());
    }

    /// The `local:` identity covers more than the header: two files with the same header but
    /// different weights are two models.
    #[tokio::test]
    async fn the_same_header_with_different_weights_is_two_models() {
        let fixture = Fixture::new().await;
        let folder = fixture.root.join("models");
        fs::create_dir_all(&folder).unwrap();
        let bytes = model_bytes("same header");
        let mut retrained = bytes.clone();
        *retrained.last_mut().unwrap() ^= 0xff;
        fs::write(folder.join("original.gguf"), &bytes).unwrap();
        fs::write(folder.join("retrained.gguf"), &retrained).unwrap();
        assert_eq!(
            header_digest(&folder.join("original.gguf")),
            header_digest(&folder.join("retrained.gguf"))
        );

        let listed = fixture
            .apply(roots(vec![root(ModelRootKind::Folder, &folder)]))
            .await;
        let mut expected = vec![
            local_id(&folder.join("original.gguf")).to_string(),
            local_id(&folder.join("retrained.gguf")).to_string(),
        ];
        expected.sort();
        let mut listed_ids = ids(&listed.models);
        listed_ids.sort();
        assert_eq!(listed_ids, expected);
        assert_ne!(expected[0], expected[1]);
    }

    /// The content sample of a 512 GiB file reads its header and two MiB of its weights, never the
    /// rest.
    #[test]
    fn the_content_sample_of_a_huge_file_reads_a_few_megabytes() {
        struct Counting<R> {
            inner: R,
            read: u64,
        }
        impl<R: std::io::Read> std::io::Read for Counting<R> {
            fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
                let read = self.inner.read(buffer)?;
                self.read += read as u64;
                Ok(read)
            }
        }
        impl<R: std::io::Seek> std::io::Seek for Counting<R> {
            fn seek(&mut self, position: std::io::SeekFrom) -> std::io::Result<u64> {
                self.inner.seek(position)
            }
        }
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("huge.gguf");
        let header = model_bytes("huge");
        let file = fs::File::create(&path).unwrap();
        std::io::Write::write_all(&mut &file, &header).unwrap();
        let length = 512 * 1024 * 1024 * 1024;
        magnitude_service_utils::sparse_file::set_sparse_len(&file, length).unwrap();
        drop(file);
        let mut counting = Counting {
            inner: fs::File::open(&path).unwrap(),
            read: 0,
        };
        let header_bytes = crate::gguf::inspect(&path).unwrap().header_bytes;
        crate::gguf::content_sample_digest(&mut counting, header_bytes, length).unwrap();
        assert!(
            counting.read <= header_bytes + 2 * crate::gguf::CONTENT_SAMPLE_BYTES,
            "read {} bytes",
            counting.read
        );
    }

    #[tokio::test]
    async fn a_huge_file_is_identified_without_reading_its_weights() {
        let fixture = Fixture::new().await;
        let folder = fixture.root.join("models");
        fs::create_dir_all(&folder).unwrap();
        let path = folder.join("huge.gguf");
        let header = model_bytes("huge");
        let file = fs::File::create(&path).unwrap();
        std::io::Write::write_all(&mut &file, &header).unwrap();
        magnitude_service_utils::sparse_file::set_sparse_len(&file, 512 * 1024 * 1024 * 1024)
            .unwrap();
        drop(file);

        let listed = tokio::time::timeout(
            Duration::from_secs(30),
            fixture.apply(roots(vec![root(ModelRootKind::Folder, &folder)])),
        )
        .await
        .expect("a 512 GiB file is identified from its header and a sample of its weights");

        assert_eq!(ids(&listed.models), [local_id(&path).to_string()]);
        let CustomModelState::Ready { model, .. } = &listed.models[0].state else {
            panic!("ready model");
        };
        assert_eq!(model.metadata.storage_bytes, 512 * 1024 * 1024 * 1024);
    }

    #[tokio::test]
    async fn drafters_are_classified_by_header_and_mtp_named_targets_stay_visible() {
        let fixture = Fixture::new().await;
        let folder = fixture.root.join("models");
        write_mtp_model(&folder.join("Qwen3.6-35B-A3B-MTP-Q4_K_M.gguf"), "target");
        write_dflash_drafter(
            &folder.join("weights.gguf"),
            "drafter without a telling name",
        );

        let listed = fixture
            .apply(roots(vec![root(ModelRootKind::Folder, &folder)]))
            .await;

        assert_eq!(
            ids(&listed.models),
            [local_id(&folder.join("Qwen3.6-35B-A3B-MTP-Q4_K_M.gguf")).to_string()]
        );
    }

    fn catalog_definition(target: ModelPackage) -> RecommendableModel {
        RecommendableModel {
            model_id: magnitude_service_contracts::models::CatalogBaseId::new("model").unwrap(),
            variant_id: magnitude_service_contracts::models::CatalogVariantId::new("gguf:q4")
                .unwrap(),
            configuration: ModelServingConfiguration {
                bundle: ServableModelBundle::Standalone { package: target },
                profile: ServingProfile {
                    context_length: 4_096,
                },
            },
            display_name: "Model".to_owned(),
            variant_label: "Q4".to_owned(),
            description: String::new(),
            release_date: magnitude_service_contracts::models::ModelReleaseDate::new("2026-01-01")
                .unwrap(),
            license: "test".to_owned(),
            parameterization: magnitude_service_contracts::models::ModelParameterization::Dense {
                total_parameters: 1_000_000_000,
            },
            intelligence: 0,
            support: magnitude_service_contracts::models::CatalogSupport::Supported,
            fidelity_rank: 0,
            quantization_aware: false,
        }
    }

    /// A catalog model whose only copy is exactly its target in a Hugging Face cache (as after
    /// removing Magnitude's copy) is a found copy. Hiding it by the `hf:` ID it has as a found model
    /// ends its attribution, even though the catalog remembers the package: the catalog model is
    /// no longer installed, the copy is a hidden found model under its own ID, and the catalog ID
    /// no longer resolves to it, across a restart, until it is unhidden.
    #[tokio::test]
    async fn hiding_a_catalog_models_external_copy_makes_it_a_hidden_found_model() {
        use magnitude_service_contracts::models::{CatalogModelState, InstalledModelPackages as _};

        let fixture = Fixture::new().await;
        let cache = fixture.root.join("hf");
        publish_hf_cache(
            &cache,
            "owner/Model-GGUF",
            &"a".repeat(40),
            &[("Model-Q4_K_M.gguf", model_bytes("model"))],
        );
        let shown = roots(vec![root(ModelRootKind::HuggingFaceCache, &cache)]);
        fixture.apply(shown.clone()).await;
        let found = fixture.inventory().list_installed().await.unwrap().packages[0]
            .package
            .clone();
        let found_id = ModelId::hugging_face(
            &HuggingFaceRepositoryId::new("owner/Model-GGUF").unwrap(),
            &HuggingFaceArtifactSelector::new("Model-Q4_K_M.gguf").unwrap(),
        );
        let catalog = vec![catalog_definition(found)];
        let catalog_model_id = catalog_id(&catalog[0]);
        let hidden = LocalModelsConfiguration {
            hidden: vec![found_id.clone()],
            ..shown.clone()
        };

        let attributed = |fixture: &Fixture| {
            let catalog = fixture.resolver.catalog_snapshot().unwrap();
            assert_eq!(catalog.models[0].aliases, [found_id.clone()]);
            matches!(
                catalog.models[0].local_state,
                CatalogModelState::Installed { .. }
            )
        };

        // Shown, the copy is the catalog model's (and the catalog remembers its package).
        let mut fixture = fixture.restart(catalog.clone()).await;
        let listed = fixture.apply(shown.clone()).await;
        assert!(listed.models.is_empty() && listed.hidden.is_empty());
        assert!(attributed(&fixture));
        assert_eq!(
            fixture
                .resolver
                .canonical_model_id(found_id.clone())
                .unwrap(),
            catalog_model_id
        );

        for restarted in [false, true] {
            if restarted {
                fixture = fixture.restart(catalog.clone()).await;
            }
            let listed = fixture.apply(hidden.clone()).await;
            assert!(listed.models.is_empty(), "restarted: {restarted}");
            assert_eq!(
                ids(&listed.hidden),
                [found_id.to_string()],
                "restarted: {restarted}"
            );
            assert!(
                !attributed(&fixture),
                "the catalog model is not installed (restarted: {restarted})"
            );
            assert_eq!(
                fixture
                    .resolver
                    .canonical_model_id(found_id.clone())
                    .unwrap(),
                found_id
            );
            assert!(
                fixture
                    .resolver
                    .assessment_configuration(&ModelAssessmentSubject::Catalog {
                        model_id: catalog_model_id.clone(),
                        selection:
                            magnitude_service_contracts::models::CatalogModelSelection::Effective,
                    })
                    .is_err(),
                "the catalog ID no longer resolves to the hidden copy (restarted: {restarted})"
            );
        }

        let listed = fixture.apply(shown).await;
        assert!(listed.models.is_empty() && listed.hidden.is_empty());
        assert!(
            attributed(&fixture),
            "unhidden, the copy is the catalog model's again"
        );
        assert_eq!(
            fixture.resolver.canonical_model_id(found_id).unwrap(),
            catalog_model_id
        );
    }

    /// QA (linux-debian, round 3c): importing a link to a file holding a catalog target's bytes
    /// started a new model under an ID nothing listed, since those bytes are the catalog model's.
    /// The store names the catalog model whose installed target, in any location, has a file's
    /// bytes, so the import answers with that catalog model.
    #[tokio::test]
    async fn a_copy_of_an_installed_catalog_targets_bytes_is_that_catalog_model() {
        use magnitude_service_contracts::models::InstalledModelPackages as _;

        let fixture = Fixture::new().await;
        let cache = fixture.root.join("hf");
        let bytes = model_bytes("catalog target");
        publish_hf_cache(
            &cache,
            "owner/Model-GGUF",
            &"a".repeat(40),
            &[("Model-Q4_K_M.gguf", bytes.clone())],
        );
        let shown = roots(vec![root(ModelRootKind::HuggingFaceCache, &cache)]);
        fixture.apply(shown.clone()).await;
        let found = fixture.inventory().list_installed().await.unwrap().packages[0]
            .package
            .clone();
        let catalog = vec![catalog_definition(found)];
        let fixture = fixture.restart(catalog.clone()).await;
        fixture.apply(shown).await;

        let copy = fixture.root.join("models/R/renamed-q8.gguf");
        fs::create_dir_all(copy.parent().unwrap()).unwrap();
        fs::write(&copy, &bytes).unwrap();
        let link = fixture.root.join("stash/link-to-renamed.gguf");
        fs::create_dir_all(link.parent().unwrap()).unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(&copy, &link).unwrap();
        #[cfg(not(unix))]
        fs::copy(&copy, &link).unwrap();
        let digest = crate::gguf::local_model_digest_of(&[&link]).unwrap();
        assert_eq!(
            fixture
                .inventory()
                .catalog_model_with_content(&digest, bytes.len() as u64),
            Some(catalog_id(&catalog[0]))
        );
        assert_eq!(
            fixture
                .inventory()
                .catalog_model_with_content(&digest, bytes.len() as u64 + 1),
            None
        );
        let other = fixture.root.join("models/other.gguf");
        fs::write(&other, model_bytes("other")).unwrap();
        assert_eq!(
            fixture.inventory().catalog_model_with_content(
                &crate::gguf::local_model_digest_of(&[&other]).unwrap(),
                fs::metadata(&other).unwrap().len()
            ),
            None
        );
    }

    /// A Hub serving one repository's metadata and file metadata at `commit`, recording every
    /// request; it never serves file contents.
    fn serve_metadata_only(
        repository: &'static str,
        commit: String,
        files: Vec<(&'static str, Vec<u8>)>,
    ) -> (String, Arc<Mutex<Vec<String>>>) {
        use std::io::{BufRead, BufReader, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&requests);
        let body = serde_json::to_vec(&serde_json::json!({
            "id": repository,
            "sha": commit,
            "siblings": files.iter().map(|(path, bytes)| serde_json::json!({
                "rfilename": path,
                "size": bytes.len(),
                "lfs": { "sha256": sha256(bytes), "size": bytes.len() },
            })).collect::<Vec<_>>(),
        }))
        .unwrap();
        std::thread::spawn(move || {
            for connection in listener.incoming() {
                let Ok(mut connection) = connection else {
                    return;
                };
                let mut reader = BufReader::new(connection.try_clone().unwrap());
                let mut request_line = String::new();
                let _ = reader.read_line(&mut request_line);
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                        break;
                    }
                }
                recorded
                    .lock()
                    .unwrap()
                    .push(request_line.trim().to_owned());
                let file = files.iter().find(|(path, _)| {
                    request_line.contains(&format!("/{repository}/resolve/{commit}/{path}"))
                });
                let response = if request_line
                    .contains(&format!("/api/models/{repository}/revision/{commit}"))
                {
                    let mut response = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    )
                    .into_bytes();
                    response.extend_from_slice(&body);
                    response
                } else if let (true, Some((_, bytes))) = (request_line.starts_with("HEAD "), file) {
                    format!(
                        "HTTP/1.1 200 OK\r\nETag: \"{}\"\r\nX-Repo-Commit: {commit}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        sha256(bytes),
                        bytes.len()
                    )
                    .into_bytes()
                } else {
                    b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                        .to_vec()
                };
                let _ = connection.write_all(&response);
            }
        });
        (endpoint, requests)
    }

    /// A Hub serving each repository's metadata and file metadata at its commit and, for ranged
    /// GETs, its files' bytes, recording every request.
    fn serve_files(
        repositories: Vec<(&'static str, String, Vec<(&'static str, Vec<u8>)>)>,
    ) -> (String, Arc<Mutex<Vec<String>>>) {
        use std::io::{BufRead, BufReader, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&requests);
        std::thread::spawn(move || {
            for connection in listener.incoming() {
                let Ok(mut connection) = connection else {
                    return;
                };
                let mut reader = BufReader::new(connection.try_clone().unwrap());
                let mut request_line = String::new();
                let _ = reader.read_line(&mut request_line);
                let mut range = None;
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                        break;
                    }
                    if let Some((name, value)) = line.split_once(':')
                        && name.eq_ignore_ascii_case("range")
                    {
                        range = Some(value.trim().to_owned());
                    }
                }
                recorded.lock().unwrap().push(request_line.trim().to_owned());
                let mut response =
                    b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec();
                for (repository, commit, files) in &repositories {
                    if request_line.contains(&format!("/api/models/{repository}/revision/{commit}")) {
                        let body = serde_json::to_vec(&serde_json::json!({
                            "id": repository,
                            "sha": commit,
                            "siblings": files.iter().map(|(path, bytes)| serde_json::json!({
                                "rfilename": path,
                                "size": bytes.len(),
                                "lfs": { "sha256": sha256(bytes), "size": bytes.len() },
                            })).collect::<Vec<_>>(),
                        }))
                        .unwrap();
                        response = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            body.len()
                        )
                        .into_bytes();
                        response.extend_from_slice(&body);
                    }
                    let Some((_, bytes)) = files.iter().find(|(path, _)| {
                        request_line.contains(&format!("/{repository}/resolve/{commit}/{path}"))
                    }) else {
                        continue;
                    };
                    if request_line.starts_with("HEAD ") {
                        response = format!(
                            "HTTP/1.1 200 OK\r\nETag: \"{}\"\r\nX-Repo-Commit: {commit}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            sha256(bytes),
                            bytes.len()
                        )
                        .into_bytes();
                        continue;
                    }
                    let (start, end) = range
                        .as_deref()
                        .and_then(|range| range.trim_start_matches("bytes=").split_once('-'))
                        .map(|(start, end)| {
                            (
                                start.parse::<usize>().unwrap_or(0),
                                end.parse::<usize>().unwrap_or(bytes.len() - 1),
                            )
                        })
                        .unwrap_or((0, bytes.len() - 1));
                    let end = end.min(bytes.len() - 1);
                    let body = &bytes[start..=end];
                    response = format!(
                        "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: bytes {start}-{end}/{}\r\nConnection: close\r\n\r\n",
                        body.len(),
                        bytes.len()
                    )
                    .into_bytes();
                    response.extend_from_slice(body);
                }
                let _ = connection.write_all(&response);
            }
        });
        (endpoint, requests)
    }

    /// A bundle over its installed packages, as the server resolves one for serving.
    fn installed_bundle_input(
        bundle: &ServableModelBundle,
    ) -> magnitude_service_contracts::models::ModelBundleInput {
        use magnitude_service_contracts::models::{
            ModelBundleInput, ModelPackageOperand, SpeculativeDraftSourceInput,
        };
        let installed = |package: &ModelPackage| ModelPackageOperand::Installed {
            package_id: package.id.clone(),
        };
        match bundle {
            ServableModelBundle::Standalone { package } => ModelBundleInput::Standalone {
                package: installed(package),
            },
            ServableModelBundle::SpeculativeDecoding {
                target,
                draft_source,
                method,
            } => ModelBundleInput::SpeculativeDecoding {
                target: installed(target),
                draft_source: match draft_source {
                    SpeculativeDraftSource::Embedded => SpeculativeDraftSourceInput::Embedded,
                    SpeculativeDraftSource::Separate { draft } => {
                        SpeculativeDraftSourceInput::Separate {
                            draft: installed(draft),
                        }
                    }
                },
                method: method.clone(),
            },
        }
    }

    struct NoOptimizer;

    impl magnitude_service_contracts::models::CatalogModelOptimizer for NoOptimizer {
        fn optimize_catalog_model(
            &self,
            _model_id: ModelId,
            _progress: Box<dyn Fn(magnitude_service_contracts::models::CatalogOptimizationProgress) + Send + Sync>,
        ) -> futures_util::future::BoxFuture<'static, ()> {
            Box::pin(async {})
        }
    }

    struct StoreRemover(Arc<ManagedModelStore>);

    impl magnitude_service_contracts::models::CatalogPackageRemover for StoreRemover {
        fn remove_catalog_packages(
            &self,
            package_ids: Vec<ModelPackageId>,
        ) -> futures_util::future::BoxFuture<'_, Result<u64, InventoryError>> {
            use magnitude_service_contracts::models::InstalledModelPackages as _;
            Box::pin(async move {
                let mut freed = 0_u64;
                for package_id in package_ids {
                    freed += self.0.remove_installed(&package_id).await?.freed_bytes;
                }
                Ok(freed)
            })
        }
    }

    /// Every model file under Magnitude's store (not its integrity records), by size.
    fn stored_files(store: &Path) -> Vec<(PathBuf, u64)> {
        fn walk(directory: &Path, files: &mut Vec<(PathBuf, u64)>) {
            let Ok(entries) = fs::read_dir(directory) else {
                return;
            };
            for entry in entries.flatten() {
                let kind = entry.file_type().unwrap();
                if kind.is_dir() {
                    walk(&entry.path(), files);
                } else if kind.is_file()
                    && entry.path().extension().is_none_or(|extension| extension != "integrity")
                {
                    files.push((entry.path(), entry.metadata().unwrap().len()));
                }
            }
        }
        let mut files = Vec::new();
        walk(&store.join("hub"), &mut files);
        files
    }

    /// Linux CUDA round 4: the catalog's exact target weights in a Hugging Face cache, and the
    /// catalog drafter's exact file in a user folder (which gives it no Hub SHA-256). The drafter's
    /// bytes are on this computer for the catalog model as they are for found models: it offers the
    /// drafter with nothing to download, and fetching it makes Magnitude's own copy of the folder's
    /// file, never a link to it, without a request to the Hub.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_catalog_drafter_found_in_a_folder_is_never_downloaded_for_its_catalog_model() {
        use magnitude_service_contracts::models::{
            CatalogComponent, CatalogMissingComponent, CatalogModelState,
            InstalledModelPackages as _, SpeculativeDraftSource, SpeculativeMethod,
        };

        let fixture = Fixture::new().await;
        let cache = fixture.root.join("hf");
        let folder = fixture.root.join("models-own").join("pair");
        let commit = "a".repeat(40);
        let draft_commit = "b".repeat(40);
        let weights = model_bytes("model");
        let drafter = drafter_bytes("drafter");
        publish_hf_cache(
            &cache,
            "owner/Model-GGUF",
            &commit,
            &[("Model-Q4_K_M.gguf", weights.clone())],
        );
        publish_hf_cache(
            &cache,
            "owner/Model-DFlash-GGUF",
            &draft_commit,
            &[("Model-DFlash-Q8_0.gguf", drafter.clone())],
        );
        let cache_root = root(ModelRootKind::HuggingFaceCache, &cache);
        fixture.apply(roots(vec![cache_root.clone()])).await;
        let packages = fixture.inventory().list_installed().await.unwrap().packages;
        let package = |name: &str| {
            packages
                .iter()
                .map(|installed| installed.package.clone())
                .find(|package| package.files.iter().any(|file| file.path.ends_with(name)))
                .unwrap()
        };
        let (target, draft) = (package("Model-Q4_K_M.gguf"), package("Model-DFlash-Q8_0.gguf"));
        fs::remove_dir_all(cache.join(hf_repo_dir("owner/Model-DFlash-GGUF"))).unwrap();
        fs::create_dir_all(&folder).unwrap();
        fs::write(folder.join("Model-DFlash-Q8_0.gguf"), &drafter).unwrap();
        fs::write(folder.join("Other-Q5_K_M.gguf"), model_bytes("other")).unwrap();
        let header_digests = BTreeMap::from([(
            sha256(&drafter),
            header_digest(&folder.join("Model-DFlash-Q8_0.gguf")),
        )]);

        let mut definition = catalog_definition(target.clone());
        definition.configuration.bundle = ServableModelBundle::SpeculativeDecoding {
            target,
            draft_source: SpeculativeDraftSource::Separate { draft },
            method: SpeculativeMethod::DFlash,
        };
        let model_id = catalog_id(&definition);
        let (endpoint, requests) = serve_files(vec![
            (
                "owner/Model-GGUF",
                commit,
                vec![("Model-Q4_K_M.gguf", weights)],
            ),
            (
                "owner/Model-DFlash-GGUF",
                draft_commit,
                vec![("Model-DFlash-Q8_0.gguf", drafter.clone())],
            ),
        ]);
        let fixture = fixture
            .restart_with_catalog_headers(endpoint, vec![definition], header_digests)
            .await;
        fixture
            .apply(roots(vec![
                cache_root,
                root(ModelRootKind::Folder, &fixture.root.join("models-own")),
            ]))
            .await;
        let state = || fixture.resolver.catalog_snapshot().unwrap().models[0].local_state.clone();
        let missing = || match state() {
            CatalogModelState::Installed {
                missing_components, ..
            } => missing_components,
            CatalogModelState::NotInstalled => panic!("the found target installs the catalog model"),
        };
        assert_eq!(
            missing(),
            vec![CatalogMissingComponent {
                component: CatalogComponent::Drafter,
                download_bytes: 0,
            }],
            "the drafter's bytes are already on this computer"
        );

        let downloads = crate::ManagedModelDownloads::open(fixture.resolver.inventory.clone())
            .await
            .unwrap();
        let installations = crate::catalog_installations::ManagedCatalogInstallations::new(
            fixture.resolver.clone(),
            Arc::new(downloads),
            Arc::new(StoreRemover(fixture.resolver.inventory.clone())),
            Arc::new(NoOptimizer),
        );
        installations
            .install_component(&model_id, CatalogComponent::Drafter)
            .await
            .unwrap();
        for _ in 0..200 {
            if missing().is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert!(missing().is_empty(), "{:?}", missing());
        let fetched = requests
            .lock()
            .unwrap()
            .iter()
            .filter(|request| request.starts_with("GET ") && request.contains("/resolve/"))
            .cloned()
            .collect::<Vec<_>>();
        assert!(fetched.is_empty(), "nothing is downloaded: {fetched:?}");
        let stored = stored_files(&fixture.store);
        let (blob, _) = stored
            .iter()
            .find(|(_, size)| *size == drafter.len() as u64)
            .expect("Magnitude stores its own copy of the drafter");
        assert_eq!(fs::read(blob).unwrap(), drafter);
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt as _;
            let found = fs::metadata(folder.join("Model-DFlash-Q8_0.gguf")).unwrap();
            assert_eq!(found.nlink(), 1, "the folder's file is never hard linked");
            assert_ne!(fs::metadata(blob).unwrap().ino(), found.ino());
        }
        assert_eq!(fs::read(folder.join("Model-DFlash-Q8_0.gguf")).unwrap(), drafter);
    }

    /// The Debian round 3b sequence on a file system without clones: the catalog's exact target
    /// weights in a Hugging Face cache, its projector and drafter nowhere. Fetching the projector
    /// stores only the projector (the found weights are never copied), and the catalog model serves
    /// the found weights, the stored projector and, once fetched, the stored drafter as one bundle.
    /// Removing it removes only Magnitude's projector and drafter. When the found weights
    /// disappear, the model is not installed and its full download reuses the stored projector.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_found_catalog_target_stores_only_its_missing_components() {
        use magnitude_service_contracts::models::{
            CatalogComponent, CatalogInstallationRemoval, CatalogMissingComponent,
            CatalogModelState, CatalogModelUpdate, InstalledModelPackages as _, ModelFileRole,
            ModelPackageInstallationOrigin, SpeculativeDraftSource, SpeculativeMethod,
        };

        let fixture = Fixture::new().await;
        let cache = fixture.root.join("hf");
        let commit = "a".repeat(40);
        let draft_commit = "b".repeat(40);
        let weights = model_bytes("model");
        let projector = gguf_bytes(
            &[
                ("general.architecture", Meta::Str("clip")),
                ("general.name", Meta::Str("projector")),
            ],
            &["v.patch_embd.weight"],
        );
        let drafter = drafter_bytes("drafter");
        let snapshot = publish_hf_cache(
            &cache,
            "owner/Model-GGUF",
            &commit,
            &[
                ("Model-Q4_K_M.gguf", weights.clone()),
                ("mmproj-F16.gguf", projector.clone()),
            ],
        );
        publish_hf_cache(
            &cache,
            "owner/Model-DFlash-GGUF",
            &draft_commit,
            &[("Model-DFlash.gguf", drafter.clone())],
        );
        let shown = roots(vec![root(ModelRootKind::HuggingFaceCache, &cache)]);
        fixture.apply(shown.clone()).await;
        let packages = fixture.inventory().list_installed().await.unwrap().packages;
        let target = packages
            .iter()
            .map(|installed| installed.package.clone())
            .find(|package| package.files.iter().any(|file| file.role == ModelFileRole::Projector))
            .expect("the catalog target pairs its projector");
        let draft = packages
            .iter()
            .map(|installed| installed.package.clone())
            .find(|package| package.files.iter().any(|file| file.path.ends_with("Model-DFlash.gguf")))
            .expect("the drafter package");
        // The cache then holds only the target's weights, as `hf download` of the one GGUF leaves it.
        fs::remove_file(snapshot.join("mmproj-F16.gguf")).unwrap();
        fs::remove_dir_all(cache.join(hf_repo_dir("owner/Model-DFlash-GGUF"))).unwrap();
        let mut definition = catalog_definition(target.clone());
        definition.configuration.bundle = ServableModelBundle::SpeculativeDecoding {
            target: target.clone(),
            draft_source: SpeculativeDraftSource::Separate {
                draft: draft.clone(),
            },
            method: SpeculativeMethod::DFlash,
        };
        // Another variant of the same repository shares the projector and the drafter but isn't
        // installed, so it never keeps them from being removed with this model.
        let mut other_target = target.clone();
        let weights_id = other_target
            .files
            .iter()
            .find(|file| file.role == ModelFileRole::Weights)
            .map(|file| file.id.clone())
            .unwrap();
        let other_weights = magnitude_service_contracts::models::ModelFileId(format!("file_{}", "d".repeat(64)));
        for file in &mut other_target.files {
            if file.id == weights_id {
                file.id = other_weights.clone();
                file.sha256 = "d".repeat(64);
                file.path = PathBuf::from("Model-Q6_K.gguf");
            }
        }
        for relationship in &mut other_target.relationships {
            if let magnitude_service_contracts::models::ModelFileRelationship::ProjectorFor {
                weights_file_id,
                ..
            } = relationship
            {
                *weights_file_id = other_weights.clone();
            }
        }
        other_target.files.sort_by(|left, right| left.path.cmp(&right.path));
        other_target.id = crate::package_service::canonical_package_id(
            &other_target.files,
            &other_target.relationships,
        );
        let mut other = definition.clone();
        other.variant_id =
            magnitude_service_contracts::models::CatalogVariantId::new("gguf:q6").unwrap();
        other.configuration.bundle = ServableModelBundle::SpeculativeDecoding {
            target: other_target,
            draft_source: SpeculativeDraftSource::Separate {
                draft: draft.clone(),
            },
            method: SpeculativeMethod::DFlash,
        };
        let catalog = vec![definition, other];
        let model_id = catalog_id(&catalog[0]);
        let (endpoint, requests) = serve_files(vec![
            (
                "owner/Model-GGUF",
                commit.clone(),
                vec![
                    ("Model-Q4_K_M.gguf", weights.clone()),
                    ("mmproj-F16.gguf", projector.clone()),
                ],
            ),
            (
                "owner/Model-DFlash-GGUF",
                draft_commit,
                vec![("Model-DFlash.gguf", drafter.clone())],
            ),
        ]);
        let fixture = fixture.restart_with_hub(endpoint, catalog.clone()).await;
        fixture.apply(shown.clone()).await;
        let downloads = crate::ManagedModelDownloads::open(fixture.resolver.inventory.clone())
            .await
            .unwrap();
        let installations = crate::catalog_installations::ManagedCatalogInstallations::new(
            fixture.resolver.clone(),
            Arc::new(downloads),
            Arc::new(StoreRemover(fixture.resolver.inventory.clone())),
            Arc::new(NoOptimizer),
        );
        let state = || {
            fixture
                .resolver
                .catalog_snapshot()
                .unwrap()
                .models
                .into_iter()
                .find(|model| model.id == model_id)
                .unwrap()
                .local_state
        };
        let missing = |state: &CatalogModelState| match state {
            CatalogModelState::Installed {
                update_state,
                missing_components,
                ..
            } => {
                assert_eq!(*update_state, CatalogModelUpdate::Current, "{state:?}");
                missing_components.clone()
            }
            CatalogModelState::NotInstalled => panic!("the found target installs the catalog model"),
        };
        let settle = |expected: usize| async move {
            for _ in 0..200 {
                if missing(&state()).len() == expected {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            missing(&state())
        };
        let projector_component = CatalogMissingComponent {
            component: CatalogComponent::Projector,
            download_bytes: projector.len() as u64,
        };
        let drafter_component = CatalogMissingComponent {
            component: CatalogComponent::Drafter,
            download_bytes: drafter.len() as u64,
        };
        assert_eq!(missing(&state()), vec![projector_component, drafter_component.clone()]);

        installations
            .install_component(&model_id, CatalogComponent::Projector)
            .await
            .unwrap();
        assert_eq!(settle(1).await, vec![drafter_component]);
        let stored = stored_files(&fixture.store);
        assert_eq!(
            stored.iter().map(|(_, size)| size).sum::<u64>(),
            projector.len() as u64,
            "only the projector is stored, never a second copy of the target: {stored:?}"
        );
        assert!(stored.iter().all(|(path, _)| !path.ends_with("Model-Q4_K_M.gguf")));
        let fetched = |requests: &Arc<Mutex<Vec<String>>>| {
            requests
                .lock()
                .unwrap()
                .iter()
                .filter(|request| request.starts_with("GET ") && request.contains("/resolve/"))
                .cloned()
                .collect::<Vec<_>>()
        };
        assert!(
            fetched(&requests).iter().all(|request| request.contains("mmproj-F16.gguf")),
            "{:?}",
            fetched(&requests)
        );

        // The catalog's target is served composed of the found weights and the stored projector.
        async fn serving(
            fixture: &Fixture,
            model_id: &ModelId,
        ) -> (ServableModelBundle, magnitude_service_contracts::models::ResolvedServableModelBundle) {
            let configuration = fixture.resolver.serving_configuration(model_id).unwrap();
            let resolved = fixture
                .inventory()
                .resolve_bundle(installed_bundle_input(&configuration.bundle))
                .await
                .unwrap();
            (configuration.bundle, resolved)
        }
        let (bundle, resolved) = serving(&fixture, &model_id).await;
        assert_eq!(bundle, ServableModelBundle::Standalone { package: target.clone() });
        let paths = resolved
            .target_model
            .components
            .iter()
            .map(|component| (component.role.clone(), component.path.clone()))
            .collect::<Vec<_>>();
        assert!(paths.contains(&(
            magnitude_service_contracts::ComponentRole::Weights,
            snapshot.join("Model-Q4_K_M.gguf")
        )), "{paths:?}");
        assert!(paths.iter().any(|(role, path)| {
            *role == magnitude_service_contracts::ComponentRole::Projector && path.starts_with(&fixture.store)
        }), "{paths:?}");

        installations
            .install_component(&model_id, CatalogComponent::Drafter)
            .await
            .unwrap();
        assert!(settle(0).await.is_empty());
        let (bundle, resolved) = serving(&fixture, &model_id).await;
        assert_eq!(bundle, catalog[0].configuration.bundle, "found weights, stored projector and drafter");
        assert!(resolved.draft_model.is_some());
        assert_eq!(
            stored_files(&fixture.store).iter().map(|(_, size)| size).sum::<u64>(),
            (projector.len() + drafter.len()) as u64
        );

        let CatalogModelState::Installed {
            installation,
            stored_components,
            ..
        } = state()
        else {
            panic!("installed");
        };
        assert!(
            matches!(
                installation,
                magnitude_service_contracts::models::ModelInstallation::Resolved {
                    ownership: magnitude_service_contracts::models::ModelInstallationOwnership::External,
                    ..
                }
            ),
            "the model's weights are only in the Hugging Face cache"
        );
        assert_eq!(
            stored_components,
            vec![CatalogComponent::Projector, CatalogComponent::Drafter],
            "Magnitude's own parts, which Remove deletes"
        );

        // Removal removes Magnitude's projector and drafter; the found weights stay and still serve.
        let removed = installations.remove(&model_id).await.unwrap();
        assert!(matches!(removed, CatalogInstallationRemoval::Removed { .. }), "{removed:?}");
        fixture.inventory().ensure_installed_model_inventory().await.unwrap();
        assert!(stored_files(&fixture.store).is_empty());
        assert_eq!(fs::read(snapshot.join("Model-Q4_K_M.gguf")).unwrap(), weights);
        assert_eq!(missing(&state()).len(), 2);
        assert!(matches!(
            state(),
            CatalogModelState::Installed { stored_components, .. } if stored_components.is_empty()
        ));
        assert!(
            matches!(
                installations.remove(&model_id).await.unwrap(),
                CatalogInstallationRemoval::Retained { .. }
            ),
            "with nothing of Magnitude's left, the found model is hidden instead"
        );

        // With the projector stored again, the found weights disappearing leave the model not
        // installed, and its full download takes the stored projector instead of fetching it.
        installations
            .install_component(&model_id, CatalogComponent::Projector)
            .await
            .unwrap();
        assert_eq!(settle(1).await.len(), 1);
        fs::remove_dir_all(cache.join(hf_repo_dir("owner/Model-GGUF"))).unwrap();
        fixture.local.refresh_local().await.unwrap();
        assert_eq!(state(), CatalogModelState::NotInstalled);
        requests.lock().unwrap().clear();
        installations.install(&model_id).await.unwrap();
        for _ in 0..200 {
            if matches!(state(), CatalogModelState::Installed { missing_components, .. } if missing_components.is_empty()) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert!(matches!(
            state(),
            CatalogModelState::Installed { missing_components, .. } if missing_components.is_empty()
        ));
        assert!(
            fetched(&requests).iter().all(|request| !request.contains("mmproj-F16.gguf")),
            "the stored projector is reused: {:?}",
            fetched(&requests)
        );
        let target_owner = fixture
            .inventory()
            .list_installed()
            .await
            .unwrap()
            .packages
            .into_iter()
            .find(|installed| installed.package.id == target.id)
            .map(|installed| installed.origin);
        assert_eq!(target_owner, Some(ModelPackageInstallationOrigin::Magnitude));
    }

    /// Hiding changes only what is shown: installing a catalog model whose target is exactly a
    /// hidden found copy copies that copy's bytes into Magnitude's store instead of downloading
    /// them, needs no further download, and leaves the copy's files where they are.
    #[tokio::test(flavor = "multi_thread")]
    async fn installing_a_catalog_model_reuses_the_bytes_of_its_hidden_found_copy() {
        use magnitude_service_contracts::models::{
            CatalogModelState, InstalledModelPackages as _, ModelDownloads as _,
            ModelInstallationOwnership, StartModelDownloadRequest,
        };

        let fixture = Fixture::new().await;
        let cache = fixture.root.join("hf");
        let commit = "a".repeat(40);
        let bytes = model_bytes("model");
        let snapshot = publish_hf_cache(
            &cache,
            "owner/Model-GGUF",
            &commit,
            &[("Model-Q4_K_M.gguf", bytes.clone())],
        );
        let shown = roots(vec![root(ModelRootKind::HuggingFaceCache, &cache)]);
        fixture.apply(shown.clone()).await;
        let found = fixture.inventory().list_installed().await.unwrap().packages[0]
            .package
            .clone();
        let found_id = ModelId::hugging_face(
            &HuggingFaceRepositoryId::new("owner/Model-GGUF").unwrap(),
            &HuggingFaceArtifactSelector::new("Model-Q4_K_M.gguf").unwrap(),
        );
        let catalog = vec![catalog_definition(found)];
        let catalog_model_id = catalog_id(&catalog[0]);
        let (endpoint, requests) = serve_metadata_only(
            "owner/Model-GGUF",
            commit,
            vec![("Model-Q4_K_M.gguf", bytes.clone())],
        );
        let fixture = fixture.restart_with_hub(endpoint, catalog.clone()).await;
        let hidden = LocalModelsConfiguration {
            hidden: vec![found_id.clone()],
            ..shown
        };
        fixture.apply(hidden.clone()).await;
        assert!(matches!(
            fixture.resolver.catalog_snapshot().unwrap().models[0].local_state,
            CatalogModelState::NotInstalled
        ));

        let downloads = crate::ManagedModelDownloads::open(fixture.resolver.inventory.clone())
            .await
            .unwrap();
        let started = downloads
            .start(StartModelDownloadRequest {
                bundle: catalog[0].configuration.bundle.clone(),
                for_model: Some(catalog_model_id),
            })
            .await
            .unwrap();
        assert!(
            started.download.is_some(),
            "a hidden copy is not an installation of the catalog model"
        );
        let mut state = fixture.resolver.catalog_snapshot().unwrap().models[0]
            .local_state
            .clone();
        for _ in 0..200 {
            if matches!(state, CatalogModelState::Installed { .. }) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
            state = fixture.resolver.catalog_snapshot().unwrap().models[0]
                .local_state
                .clone();
        }
        let CatalogModelState::Installed {
            installation,
            update_state,
            ..
        } = state
        else {
            panic!("the catalog model is installed from the hidden copy's bytes: {state:?}");
        };
        assert!(matches!(
            installation,
            magnitude_service_contracts::models::ModelInstallation::Resolved {
                ownership: ModelInstallationOwnership::Magnitude,
                ..
            }
        ));
        assert_eq!(
            update_state,
            magnitude_service_contracts::models::CatalogModelUpdate::Current
        );
        let fetched = requests
            .lock()
            .unwrap()
            .iter()
            .filter(|request| request.starts_with("GET ") && request.contains("/resolve/"))
            .cloned()
            .collect::<Vec<_>>();
        assert!(
            fetched.is_empty(),
            "no file bytes are downloaded: {fetched:?}"
        );
        assert_eq!(fs::read(snapshot.join("Model-Q4_K_M.gguf")).unwrap(), bytes);
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt as _;
            let identity = |path: PathBuf| {
                let metadata = fs::metadata(path).unwrap();
                (metadata.dev(), metadata.ino(), metadata.nlink())
            };
            let installed = fixture
                .store
                .join("hub")
                .join("models--owner--Model-GGUF")
                .join("snapshots")
                .join("a".repeat(40))
                .join("Model-Q4_K_M.gguf");
            let (_, installed_inode, installed_links) = identity(installed);
            let (_, found_inode, found_links) = identity(snapshot.join("Model-Q4_K_M.gguf"));
            assert_ne!(
                installed_inode, found_inode,
                "Magnitude's blob is its own copy, never a link to the found file"
            );
            assert_eq!((installed_links, found_links), (1, 1));
        }
        // The copy holds the installed catalog model's content, so it is that model's and isn't
        // listed apart; its ID stays hidden, so removing Magnitude's copy hides it again.
        let listed = fixture.local.refresh_local().await.unwrap();
        assert!(listed.models.is_empty() && listed.hidden.is_empty());
        assert_eq!(
            fixture
                .inventory()
                .local_models_configuration()
                .unwrap()
                .hidden,
            [found_id]
        );
    }

    #[tokio::test]
    async fn hidden_models_are_listed_separately() {
        let fixture = Fixture::new().await;
        let folder = fixture.root.join("models");
        write_model(&folder.join("kept.gguf"), "kept");
        write_model(&folder.join("hidden.gguf"), "hidden");
        let hidden = local_id(&folder.join("hidden.gguf"));

        let listed = fixture
            .apply(LocalModelsConfiguration {
                roots: vec![root(ModelRootKind::Folder, &folder)],
                hidden: vec![hidden.clone()],
                ..LocalModelsConfiguration::default()
            })
            .await;

        assert_eq!(
            ids(&listed.models),
            [local_id(&folder.join("kept.gguf")).to_string()]
        );
        assert_eq!(ids(&listed.hidden), [hidden.to_string()]);
    }

    #[tokio::test]
    async fn unreadable_ggufs_are_listed_as_unsupported() {
        let fixture = Fixture::new().await;
        let folder = fixture.root.join("models");
        fs::create_dir_all(&folder).unwrap();
        fs::write(folder.join("broken.gguf"), b"GGUF but not really").unwrap();

        let listed = fixture
            .apply(roots(vec![root(ModelRootKind::Folder, &folder)]))
            .await;

        assert!(listed.models.is_empty());
        assert_eq!(listed.unsupported.len(), 1);
        assert_eq!(listed.unsupported[0].path, folder.join("broken.gguf"));
        assert_eq!(
            listed.unsupported[0].reasons,
            [UnsupportedReason::Known(KnownUnsupportedReason::Unreadable)]
        );
        assert_eq!(
            listed.unsupported[0].origin,
            CustomModelOrigin::Found {
                source: FoundSource::Folder {
                    path: folder.clone()
                }
            }
        );
    }

    #[tokio::test]
    async fn a_truncated_gguf_with_a_whole_header_is_unreadable() {
        let fixture = Fixture::new().await;
        let folder = fixture.root.join("models");
        fs::create_dir_all(&folder).unwrap();
        let whole = model_bytes("Whole");
        fs::write(folder.join("whole.gguf"), &whole).unwrap();
        fs::write(folder.join("truncated.gguf"), &whole[..whole.len() - 1]).unwrap();

        let listed = fixture
            .apply(roots(vec![root(ModelRootKind::Folder, &folder)]))
            .await;

        assert_eq!(
            ids(&listed.models),
            [local_id(&folder.join("whole.gguf")).to_string()]
        );
        assert_eq!(listed.unsupported.len(), 1);
        assert_eq!(listed.unsupported[0].path, folder.join("truncated.gguf"));
        assert_eq!(
            listed.unsupported[0].reasons,
            [UnsupportedReason::Known(KnownUnsupportedReason::Unreadable)]
        );
    }

    /// A subfolder Magnitude can't read hides nothing else in its folder: the folder's models, and
    /// its can't-read file, stay listed, and the subfolder is listed as can't read by its path in
    /// the folder until it becomes readable.
    #[tokio::test]
    async fn an_unreadable_subfolder_is_listed_as_unreadable_beside_the_rest_of_its_folder() {
        let fixture = Fixture::new().await;
        let folder = fixture.root.join("models");
        write_model(&folder.join("nested/kept.gguf"), "kept");
        fs::write(folder.join("broken.gguf"), b"GGUF but not really").unwrap();
        write_model(&folder.join("locked/inside.gguf"), "inside");
        let Some(locked) = crate::test_gguf::LockedDirectory::lock(&folder.join("locked")) else {
            return;
        };
        let Some(_empty) = crate::test_gguf::LockedDirectory::lock(&folder.join("empty-locked"))
        else {
            return;
        };

        let listed = fixture
            .apply(roots(vec![root(ModelRootKind::Folder, &folder)]))
            .await;

        assert_eq!(
            ids(&listed.models),
            [local_id(&folder.join("nested/kept.gguf")).to_string()]
        );
        let unsupported = listed
            .unsupported
            .iter()
            .map(|model| (model.display_name.as_str(), model.path.clone()))
            .collect::<Vec<_>>();
        assert_eq!(
            unsupported,
            [
                ("broken", folder.join("broken.gguf")),
                ("empty-locked", folder.join("empty-locked")),
                ("locked", folder.join("locked")),
            ]
        );
        for model in &listed.unsupported {
            assert_eq!(
                model.reasons,
                [UnsupportedReason::Known(KnownUnsupportedReason::Unreadable)]
            );
            assert_eq!(
                model.origin,
                CustomModelOrigin::Found {
                    source: FoundSource::Folder {
                        path: folder.clone()
                    }
                }
            );
        }

        drop(locked);
        fixture
            .inventory()
            .rescan_model_roots(BTreeSet::from([folder.clone()]))
            .await
            .unwrap();
        let listed = fixture.local.list_local().await.unwrap();
        assert_eq!(listed.models.len(), 2);
        assert_eq!(
            listed
                .unsupported
                .iter()
                .map(|model| model.display_name.as_str())
                .collect::<Vec<_>>(),
            ["broken", "empty-locked"]
        );
    }

    #[tokio::test]
    async fn an_unreadable_location_does_not_hide_the_others() {
        let fixture = Fixture::new().await;
        let folder = fixture.root.join("models");
        write_model(&folder.join("model.gguf"), "model");
        let file_as_root = fixture.root.join("not-a-directory");
        fs::write(&file_as_root, b"file").unwrap();
        let ollama = fixture.root.join("ollama");
        fs::create_dir_all(ollama.join("manifests/registry.ollama.ai/library/m")).unwrap();
        fs::write(
            ollama.join("manifests/registry.ollama.ai/library/m/latest"),
            b"not json",
        )
        .unwrap();

        let listed = fixture
            .apply(roots(vec![
                root(ModelRootKind::Folder, &folder),
                root(ModelRootKind::LmStudio, &file_as_root),
                root(ModelRootKind::Ollama, &ollama),
                root(ModelRootKind::LlamaCpp, &fixture.root.join("missing")),
            ]))
            .await;

        assert_eq!(listed.models.len(), 1);
    }

    #[tokio::test]
    async fn a_folder_is_refused_as_an_imported_model() {
        let fixture = Fixture::new().await;
        let folder = fixture.root.join("gguf-import");
        write_model(&folder.join("model.gguf"), "imported");
        let declaration = CustomModelDeclaration {
            model_id: local_id(&folder.join("model.gguf")),
            origin: CustomModelDeclarationOrigin::Imported {
                path: folder.clone(),
            },
            target: vec![folder.to_string_lossy().into_owned()],
            projector: None,
        };

        let refused = fixture
            .local
            .configure(LocalModelsConfiguration {
                declarations: vec![declaration],
                ..LocalModelsConfiguration::default()
            })
            .await;
        assert!(
            matches!(&refused, Err(InventoryError::InvalidRequest(message)) if message.contains("names the folder")),
            "{refused:?}"
        );
    }

    #[tokio::test]
    async fn imported_files_are_referenced_in_place_and_forgotten_without_deletion() {
        let fixture = Fixture::new().await;
        let elsewhere = fixture.root.join("Downloads");
        let file = elsewhere.join("imported.gguf");
        write_model(&file, "imported");
        let id = local_id(&file);
        let declaration = CustomModelDeclaration {
            model_id: id.clone(),
            origin: CustomModelDeclarationOrigin::Imported { path: file.clone() },
            target: vec![file.to_string_lossy().into_owned()],
            projector: None,
        };

        let listed = fixture
            .apply(LocalModelsConfiguration {
                declarations: vec![declaration],
                ..LocalModelsConfiguration::default()
            })
            .await;
        assert_eq!(ids(&listed.models), [id.to_string()]);
        assert_eq!(
            listed.models[0].origin,
            CustomModelOrigin::Imported { path: file.clone() }
        );
        assert!(matches!(
            fixture.local.remove_local(&id).await,
            Err(InventoryError::Unsupported(_))
        ));

        let forgotten = fixture.apply(LocalModelsConfiguration::default()).await;
        assert!(forgotten.models.is_empty());
        assert!(
            file.exists(),
            "forgetting an imported model never touches its file"
        );
    }

    #[tokio::test]
    async fn importing_a_copy_of_a_found_model_adds_a_location_and_never_replaces_it() {
        let fixture = Fixture::new().await;
        let lm_studio = fixture.root.join("lmstudio");
        let found = lm_studio.join("lmstudio-community/Shared-GGUF/Shared-Q4_K_M.gguf");
        write_model(&found, "shared");
        let copy = fixture.root.join("Downloads/Shared-copy.gguf");
        fs::create_dir_all(copy.parent().unwrap()).unwrap();
        fs::copy(&found, &copy).unwrap();
        let copy_id = local_id(&copy);
        let configuration = LocalModelsConfiguration {
            roots: vec![root(ModelRootKind::LmStudio, &lm_studio)],
            declarations: vec![CustomModelDeclaration {
                model_id: copy_id.clone(),
                origin: CustomModelDeclarationOrigin::Imported { path: copy.clone() },
                target: vec![copy.to_string_lossy().into_owned()],
                projector: None,
            }],
            ..LocalModelsConfiguration::default()
        };

        let listed = fixture.apply(configuration).await;
        assert_eq!(
            ids(&listed.models),
            ["hf:lmstudio-community/Shared-GGUF/Shared-Q4_K_M.gguf"],
            "the same bytes are one model, keeping the identity its found location gives it"
        );
        assert_eq!(
            listed.models[0].origin,
            CustomModelOrigin::Found {
                source: FoundSource::LmStudio
            }
        );
        assert_eq!(listed.models[0].files[0].path, found);

        fs::remove_file(&found).unwrap();
        let after = fixture.local.refresh_local().await.unwrap();
        assert_eq!(
            ids(&after.models),
            [copy_id.to_string()],
            "the imported copy keeps the model when the found file disappears"
        );
        assert_eq!(
            after.models[0].origin,
            CustomModelOrigin::Imported { path: copy.clone() }
        );
        assert_eq!(after.models[0].files[0].path, copy);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn an_imported_symlink_is_served_through_its_link_wherever_it_points() {
        let fixture = Fixture::new().await;
        let hub = fixture.root.join("hf/hub");
        let snapshot = publish_shared_hf_cache(
            &hub,
            "owner/Linked-GGUF",
            &"d".repeat(40),
            &[("Linked-Q4_K_M.gguf", model_bytes("linked"))],
        );
        let link = fixture.root.join("Downloads/linked.gguf");
        fs::create_dir_all(link.parent().unwrap()).unwrap();
        std::os::unix::fs::symlink(snapshot.join("Linked-Q4_K_M.gguf"), &link).unwrap();
        let id = local_id(&link);

        let listed = fixture
            .apply(LocalModelsConfiguration {
                declarations: vec![CustomModelDeclaration {
                    model_id: id.clone(),
                    origin: CustomModelDeclarationOrigin::Imported { path: link.clone() },
                    target: vec![link.to_string_lossy().into_owned()],
                    projector: None,
                }],
                ..LocalModelsConfiguration::default()
            })
            .await;

        assert_eq!(ids(&listed.models), [id.to_string()]);
        assert!(matches!(
            listed.models[0].state,
            CustomModelState::Ready { .. }
        ));
        assert_eq!(listed.models[0].files[0].path, link);
        let inventory = fixture.inventory();
        let model = inventory
            .list()
            .await
            .unwrap()
            .into_iter()
            .find(|model| {
                matches!(
                    model.location,
                    ModelLocation::External {
                        origin: ExternalOrigin::Imported,
                        ..
                    }
                )
            })
            .expect("the imported link is published");
        let resolved = inventory.resolve_ready(&model.id).await.unwrap();
        assert_eq!(resolved.components[0].path, link);
    }

    #[tokio::test]
    async fn added_models_survive_rescans_pair_declared_projectors_and_remove_only_their_copy() {
        let fixture = Fixture::new().await;
        let commit = "c".repeat(40);
        let repository = "owner/Added-GGUF";
        let bytes = model_bytes("added");
        let projector = |name: &str| {
            gguf_bytes(
                &[
                    ("general.architecture", Meta::Str("clip")),
                    ("general.name", Meta::Str(name)),
                ],
                &["v.patch_embd.weight"],
            )
        };
        publish_managed(
            &fixture.store,
            repository,
            &commit,
            &[
                ("Added-Q4_K_M.gguf", bytes.clone()),
                ("mmproj-F16.gguf", projector("f16")),
                ("mmproj-Q8_0.gguf", projector("q8")),
            ],
        );
        let cache = fixture.root.join("hf");
        let id: ModelId = "hf:owner/Added-GGUF/Added-Q4_K_M.gguf".parse().unwrap();
        let configuration = LocalModelsConfiguration {
            roots: vec![root(ModelRootKind::HuggingFaceCache, &cache)],
            declarations: vec![CustomModelDeclaration {
                model_id: id.clone(),
                origin: CustomModelDeclarationOrigin::Added {
                    repository: HuggingFaceRepositoryId::new(repository).unwrap(),
                    commit: commit.clone(),
                },
                target: vec!["Added-Q4_K_M.gguf".to_owned()],
                projector: Some("mmproj-Q8_0.gguf".to_owned()),
            }],
            ..LocalModelsConfiguration::default()
        };

        let listed = fixture.apply(configuration).await;
        let rescanned = fixture.local.refresh_local().await.unwrap();
        assert_eq!(
            listed,
            LocalModelsResponse {
                revision: listed.revision,
                ..rescanned.clone()
            }
        );
        assert_eq!(ids(&rescanned.models), [id.to_string()]);
        let added = &rescanned.models[0];
        assert!(matches!(added.origin, CustomModelOrigin::Added { .. }));
        assert_eq!(
            added
                .projector
                .as_ref()
                .and_then(|file| file.path.file_name())
                .and_then(|name| name.to_str()),
            Some("mmproj-Q8_0.gguf")
        );

        // The same bytes also sit in the user's Hugging Face cache.
        let external =
            publish_hf_cache(&cache, repository, &commit, &[("Added-Q4_K_M.gguf", bytes)]);
        fixture.local.refresh_local().await.unwrap();
        let removal = fixture.local.remove_local(&id).await.unwrap();
        assert!(removal.removed);
        assert!(removal.freed_bytes > 0);
        assert!(external.join("Added-Q4_K_M.gguf").exists());
        let after = fixture.local.refresh_local().await.unwrap();
        assert_eq!(ids(&after.models), [id.to_string()]);
        assert!(
            matches!(after.models[0].origin, CustomModelOrigin::Added { .. }),
            "the declaration still names it while ACN keeps it"
        );
        assert!(after.models[0].files[0].path.starts_with(&cache));
    }

    /// QA (mac-m4, round 3c): importing a file byte-identical to an added model said it was that
    /// model, yet listed a second entry under the file's own `local:` ID. Adding a version previews its
    /// headers first, and the preview's header-only files were inspected through the cache keyed
    /// by content, so the downloaded file with the same SHA-256 took the preview's content sample
    /// (read from zeros) and a `local:` digest no copy of its bytes has. A downloaded added model and
    /// a copy of it are one model.
    #[tokio::test]
    async fn a_previewed_then_downloaded_model_has_its_own_bytes_digest() {
        let commit = "c".repeat(40);
        let repository = "owner/Added-GGUF";
        let mut bytes = model_bytes("previewed");
        let length = bytes.len();
        for (index, byte) in bytes[length - 32..].iter_mut().enumerate() {
            *byte = index as u8 + 1;
        }
        let (endpoint, _) = serve_files(vec![(
            repository,
            commit.clone(),
            vec![("Added-Q4_K_M.gguf", bytes.clone())],
        )]);
        let fixture = Fixture::with_hub(endpoint).await;
        let snapshot = crate::refresh_hugging_face_repository(
            &fixture.resolver.inventory,
            magnitude_service_contracts::HuggingFaceRepositoryRequest {
                repository: repository.to_owned(),
                revision: commit.clone(),
            },
        )
        .await
        .unwrap();
        fixture
            .inventory()
            .hugging_face_package(
                &magnitude_service_contracts::ModelPreviewSource {
                    repository: repository.to_owned(),
                    revision: commit.clone(),
                    primary_gguf: PathBuf::from("Added-Q4_K_M.gguf"),
                    additional_components: Vec::new(),
                },
                &snapshot,
            )
            .await
            .unwrap();

        publish_managed(
            &fixture.store,
            repository,
            &commit,
            &[("Added-Q4_K_M.gguf", bytes.clone())],
        );
        let copy = fixture.root.join("Downloads/Added-copy.gguf");
        fs::create_dir_all(copy.parent().unwrap()).unwrap();
        fs::write(&copy, &bytes).unwrap();
        let added_id: ModelId = "hf:owner/Added-GGUF/Added-Q4_K_M.gguf".parse().unwrap();
        let listed = fixture
            .apply(LocalModelsConfiguration {
                declarations: vec![
                    CustomModelDeclaration {
                        model_id: added_id.clone(),
                        origin: CustomModelDeclarationOrigin::Added {
                            repository: HuggingFaceRepositoryId::new(repository).unwrap(),
                            commit: commit.clone(),
                        },
                        target: vec!["Added-Q4_K_M.gguf".to_owned()],
                        projector: None,
                    },
                    CustomModelDeclaration {
                        model_id: local_id(&copy),
                        origin: CustomModelDeclarationOrigin::Imported { path: copy.clone() },
                        target: vec![copy.to_string_lossy().into_owned()],
                        projector: None,
                    },
                ],
                ..LocalModelsConfiguration::default()
            })
            .await;
        assert_eq!(ids(&listed.models), [added_id.to_string()], "{listed:#?}");
    }

    /// Removing an added model whose same bytes are also in Ollama or the llama.cpp cache deletes
    /// only Magnitude's copy; once ACN drops the declaration, the model is listed as found from
    /// that location (by the identity that location gives it) and is not hidden.
    #[tokio::test]
    async fn removing_an_added_model_found_in_ollama_or_llama_cpp_keeps_it_as_a_found_model() {
        let commit = "c".repeat(40);
        let repository = "owner/Added-GGUF";
        let added_id: ModelId = "hf:owner/Added-GGUF/Added-Q4_K_M.gguf".parse().unwrap();
        for source in [FoundSource::LlamaCpp, FoundSource::Ollama] {
            let fixture = Fixture::new().await;
            let bytes = model_bytes("added and found");
            let snapshot = publish_managed(
                &fixture.store,
                repository,
                &commit,
                &[("Added-Q4_K_M.gguf", bytes.clone())],
            );
            let location = fixture.root.join("location");
            let (kind, found_id, external) = match &source {
                FoundSource::LlamaCpp => {
                    fs::create_dir_all(&location).unwrap();
                    let file = location.join("owner_Added-GGUF_Added-Q4_K_M.gguf");
                    fs::write(&file, &bytes).unwrap();
                    (ModelRootKind::LlamaCpp, added_id.clone(), file)
                }
                FoundSource::Ollama => {
                    write_ollama(
                        &location,
                        "registry.ollama.ai",
                        "library",
                        "added",
                        "latest",
                        &[("application/vnd.ollama.image.model", &bytes)],
                    );
                    let file = location
                        .join("blobs")
                        .join(format!("sha256-{}", sha256(&bytes)));
                    (ModelRootKind::Ollama, local_id(&file), file)
                }
                _ => unreachable!(),
            };
            let shown = roots(vec![root(kind, &location)]);
            let declared = LocalModelsConfiguration {
                declarations: vec![CustomModelDeclaration {
                    model_id: added_id.clone(),
                    origin: CustomModelDeclarationOrigin::Added {
                        repository: HuggingFaceRepositoryId::new(repository).unwrap(),
                        commit: commit.clone(),
                    },
                    target: vec!["Added-Q4_K_M.gguf".to_owned()],
                    projector: None,
                }],
                ..shown.clone()
            };
            let listed = fixture.apply(declared).await;
            assert_eq!(ids(&listed.models), [added_id.to_string()], "{source:?}");
            assert!(matches!(listed.models[0].origin, CustomModelOrigin::Added { .. }));

            let removal = fixture.local.remove_local(&added_id).await.unwrap();
            assert!(removal.removed, "{source:?}");
            let listed = fixture.apply(shown).await;
            assert!(
                !snapshot.join("Added-Q4_K_M.gguf").exists(),
                "Magnitude's copy is gone ({source:?})"
            );
            assert!(external.exists(), "the other location's file stays ({source:?})");
            assert_eq!(ids(&listed.models), [found_id.to_string()], "{source:?}");
            assert_eq!(
                listed.models[0].origin,
                CustomModelOrigin::Found { source: source.clone() }
            );
            assert!(listed.hidden.is_empty(), "{source:?}");
        }
    }

    #[tokio::test]
    async fn an_added_model_is_listed_not_installed_until_its_download_lands() {
        let fixture = Fixture::new().await;
        let commit = "c".repeat(40);
        let repository = "owner/Pending-GGUF";
        let id: ModelId = "hf:owner/Pending-GGUF/Pending-Q4_K_M.gguf".parse().unwrap();
        let drafter = DrafterReference::HuggingFace {
            repository: HuggingFaceRepositoryId::new("owner/Pending-DFlash-GGUF").unwrap(),
            commit: "d".repeat(40),
            path: "dflash.gguf".to_owned(),
        };
        let configuration = LocalModelsConfiguration {
            declarations: vec![CustomModelDeclaration {
                model_id: id.clone(),
                origin: CustomModelDeclarationOrigin::Added {
                    repository: HuggingFaceRepositoryId::new(repository).unwrap(),
                    commit: commit.clone(),
                },
                target: vec!["Pending-Q4_K_M.gguf".to_owned()],
                projector: None,
            }],
            drafters: vec![DrafterAttachment {
                model_id: id.clone(),
                drafter: DrafterChoice::Reference {
                    reference: drafter.clone(),
                    method: SpeculativeMethod::DFlash,
                },
            }],
            ..LocalModelsConfiguration::default()
        };

        let pending = fixture.apply(configuration).await;
        assert_eq!(ids(&pending.models), [id.to_string()]);
        let model = &pending.models[0];
        assert_eq!(model.state, CustomModelState::NotInstalled);
        assert_eq!(model.display_name, "Pending");
        assert_eq!(model.variant_label.as_deref(), Some("Q4_K_M"));
        assert_eq!(
            model
                .drafter
                .as_ref()
                .map(|attached| (&attached.reference, &attached.state)),
            Some((&drafter, &AttachedDrafterState::NotInstalled))
        );
        assert!(
            fixture
                .local
                .resolver
                .custom_serving_configuration(&id)
                .is_err()
        );

        publish_managed(
            &fixture.store,
            repository,
            &commit,
            &[("Pending-Q4_K_M.gguf", model_bytes("pending"))],
        );
        let landed = fixture.local.refresh_local().await.unwrap();
        assert_eq!(ids(&landed.models), [id.to_string()]);
        assert!(matches!(
            landed.models[0].state,
            CustomModelState::Ready { .. }
        ));
    }

    #[tokio::test]
    async fn a_rescan_that_finds_nothing_new_keeps_the_catalog_revision() {
        let fixture = Fixture::new().await;
        let folder = fixture.root.join("models");
        write_model(&folder.join("model.gguf"), "model");
        fixture
            .apply(roots(vec![root(ModelRootKind::Folder, &folder)]))
            .await;
        let before = fixture.local.resolver.revision().unwrap();
        fixture.local.refresh_local().await.unwrap();
        fixture.local.refresh_local().await.unwrap();
        assert_eq!(fixture.local.resolver.revision().unwrap(), before);

        write_model(&folder.join("other.gguf"), "other");
        fixture.local.refresh_local().await.unwrap();
        assert!(fixture.local.resolver.revision().unwrap() > before);
    }

    #[tokio::test]
    async fn external_files_are_never_deleted() {
        let fixture = Fixture::new().await;
        let cache = fixture.root.join("hf");
        let snapshot = publish_hf_cache(
            &cache,
            "owner/Found-GGUF",
            &"d".repeat(40),
            &[("found.gguf", model_bytes("found"))],
        );
        let folder = fixture.root.join("models");
        write_model(&folder.join("folder.gguf"), "folder");
        let listed = fixture
            .apply(roots(vec![
                root(ModelRootKind::HuggingFaceCache, &cache),
                root(ModelRootKind::Folder, &folder),
            ]))
            .await;
        assert_eq!(listed.models.len(), 2);

        for model in &listed.models {
            assert!(matches!(
                fixture.local.remove_local(&model.model_id).await,
                Err(InventoryError::Unsupported(_))
            ));
        }
        for entry in fixture.inventory().list().await.unwrap() {
            assert!(matches!(
                fixture.inventory().delete(&entry.id).await,
                Err(InventoryError::Unsupported(_))
            ));
            assert!(
                !fixture
                    .inventory()
                    .plan_delete(&entry.id)
                    .await
                    .unwrap()
                    .supported
            );
        }
        assert!(snapshot.join("found.gguf").exists());
        assert!(folder.join("folder.gguf").exists());
    }

    #[tokio::test]
    async fn a_drafter_downloaded_alone_survives_rescans_and_attaches() {
        let fixture = Fixture::new().await;
        let commit = "e".repeat(40);
        publish_managed(
            &fixture.store,
            "magnitudedev/Model-DFlash-GGUF",
            &commit,
            &[("Model-DFlash-Q8_0.gguf", drafter_bytes("drafter"))],
        );
        let folder = fixture.root.join("models");
        write_model(&folder.join("target.gguf"), "target");
        let target = local_id(&folder.join("target.gguf"));
        let reference = DrafterReference::HuggingFace {
            repository: HuggingFaceRepositoryId::new("magnitudedev/Model-DFlash-GGUF").unwrap(),
            commit,
            path: "Model-DFlash-Q8_0.gguf".to_owned(),
        };
        let configuration = LocalModelsConfiguration {
            roots: vec![root(ModelRootKind::Folder, &folder)],
            drafters: vec![DrafterAttachment {
                model_id: target.clone(),
                drafter: DrafterChoice::Reference {
                    reference: reference.clone(),
                    method: SpeculativeMethod::DFlash,
                },
            }],
            ..LocalModelsConfiguration::default()
        };

        fixture.apply(configuration).await;
        let listed = fixture.local.refresh_local().await.unwrap();

        assert_eq!(ids(&listed.models), [target.to_string()]);
        assert_eq!(
            listed.models[0].drafter,
            Some(AttachedDrafter {
                reference,
                method: SpeculativeMethod::DFlash,
                state: AttachedDrafterState::Ready,
                update_available: false,
            })
        );
        let configuration = fixture.resolver.serving_configuration(&target).unwrap();
        assert!(matches!(
            configuration.bundle,
            ServableModelBundle::SpeculativeDecoding {
                draft_source: SpeculativeDraftSource::Separate { .. },
                method: SpeculativeMethod::DFlash,
                ..
            }
        ));
    }

    /// Knows the bytes of one drafter reference, as the server learns them from its header.
    struct KnowsDrafter {
        reference: DrafterReference,
        content: DrafterContent,
    }

    impl CustomModelEnrichment for KnowsDrafter {
        fn enrich(&self, _model: &CustomModelFacts<'_>) -> CustomModelEnrichmentResult {
            CustomModelEnrichmentResult::default()
        }

        fn drafter_content(&self, reference: &DrafterReference) -> Option<DrafterContent> {
            (reference == &self.reference).then(|| self.content.clone())
        }
    }

    #[tokio::test]
    async fn a_drafter_already_in_another_location_is_used_where_it_is() {
        let fixture = Fixture::new().await;
        let commit = "e".repeat(40);
        let bytes = drafter_bytes("drafter");
        let lm_studio = fixture.root.join("lmstudio");
        let in_lm_studio = lm_studio.join("someone/Mirror-GGUF/drafter-copy.gguf");
        fs::create_dir_all(in_lm_studio.parent().unwrap()).unwrap();
        fs::write(&in_lm_studio, &bytes).unwrap();
        let folder = fixture.root.join("models");
        write_model(&folder.join("target.gguf"), "target");
        let target = local_id(&folder.join("target.gguf"));
        let reference = DrafterReference::HuggingFace {
            repository: HuggingFaceRepositoryId::new("magnitudedev/Model-DFlash-GGUF").unwrap(),
            commit,
            path: "Model-DFlash-Q8_0.gguf".to_owned(),
        };
        fixture
            .resolver
            .set_custom_model_enrichment(Arc::new(KnowsDrafter {
                reference: reference.clone(),
                content: DrafterContent {
                    header_digest: Some(header_digest(&in_lm_studio)),
                    sha256: Some(sha256(&bytes)),
                    size_bytes: bytes.len() as u64,
                },
            }));
        let configuration = LocalModelsConfiguration {
            roots: vec![
                root(ModelRootKind::Folder, &folder),
                root(ModelRootKind::LmStudio, &lm_studio),
            ],
            drafters: vec![DrafterAttachment {
                model_id: target.clone(),
                drafter: DrafterChoice::Reference {
                    reference: reference.clone(),
                    method: SpeculativeMethod::DFlash,
                },
            }],
            ..LocalModelsConfiguration::default()
        };

        let listed = fixture.apply(configuration).await;
        assert_eq!(ids(&listed.models), [target.to_string()]);
        assert_eq!(
            listed.models[0]
                .drafter
                .as_ref()
                .map(|drafter| drafter.state.clone()),
            Some(AttachedDrafterState::Ready),
            "a file with the drafter's header and size is that drafter, wherever it is"
        );
        let ServableModelBundle::SpeculativeDecoding {
            draft_source: SpeculativeDraftSource::Separate { draft },
            ..
        } = fixture
            .resolver
            .serving_configuration(&target)
            .unwrap()
            .bundle
        else {
            panic!("the model serves with the drafter found in LM Studio");
        };
        assert_eq!(
            draft.files[0].path,
            PathBuf::from("someone/Mirror-GGUF/drafter-copy.gguf")
        );
        assert!(
            !fixture.store.join("hub").exists()
                || fs::read_dir(fixture.store.join("hub"))
                    .unwrap()
                    .next()
                    .is_none(),
            "nothing is copied into Magnitude's store"
        );

        fs::remove_file(&in_lm_studio).unwrap();
        let after = fixture.local.refresh_local().await.unwrap();
        assert_eq!(
            after.models[0]
                .drafter
                .as_ref()
                .map(|drafter| drafter.state.clone()),
            Some(AttachedDrafterState::NotInstalled),
            "when the copy disappears the drafter can be downloaded again"
        );
        assert!(matches!(
            fixture
                .resolver
                .serving_configuration(&target)
                .unwrap()
                .bundle,
            ServableModelBundle::Standalone { .. }
        ));

        // A Hugging Face cache blob of another repository holds it by its Hub SHA-256 alone.
        let cache = fixture.root.join("hf");
        publish_hf_cache(
            &cache,
            "someone/Other-GGUF",
            &"f".repeat(40),
            &[("other-name.gguf", bytes.clone())],
        );
        fixture
            .resolver
            .set_custom_model_enrichment(Arc::new(KnowsDrafter {
                reference: reference.clone(),
                content: DrafterContent {
                    header_digest: None,
                    sha256: Some(sha256(&bytes)),
                    size_bytes: bytes.len() as u64,
                },
            }));
        let mut configuration = fixture.local.configuration().await.unwrap();
        configuration
            .roots
            .push(root(ModelRootKind::HuggingFaceCache, &cache));
        let cached = fixture.apply(configuration).await;
        assert_eq!(
            cached.models[0]
                .drafter
                .as_ref()
                .map(|drafter| drafter.state.clone()),
            Some(AttachedDrafterState::Ready)
        );
    }

    #[tokio::test]
    async fn a_drafter_no_model_uses_is_pruned_and_one_in_use_is_kept() {
        let fixture = Fixture::new().await;
        let commit = "e".repeat(40);
        let snapshot = publish_managed(
            &fixture.store,
            "magnitudedev/Model-DFlash-GGUF",
            &commit,
            &[("Model-DFlash-Q8_0.gguf", drafter_bytes("drafter"))],
        );
        let folder = fixture.root.join("models");
        write_model(&folder.join("target.gguf"), "target");
        let target = local_id(&folder.join("target.gguf"));
        let attached = LocalModelsConfiguration {
            roots: vec![root(ModelRootKind::Folder, &folder)],
            drafters: vec![DrafterAttachment {
                model_id: target,
                drafter: DrafterChoice::Reference {
                    reference: DrafterReference::HuggingFace {
                        repository: HuggingFaceRepositoryId::new("magnitudedev/Model-DFlash-GGUF")
                            .unwrap(),
                        commit,
                        path: "Model-DFlash-Q8_0.gguf".to_owned(),
                    },
                    method: SpeculativeMethod::DFlash,
                },
            }],
            ..LocalModelsConfiguration::default()
        };
        fixture.apply(attached.clone()).await;
        let kept = fixture.local.prune_drafters().await.unwrap();
        assert!(
            kept.removed.is_empty(),
            "a drafter a model serves with is kept"
        );
        assert!(snapshot.join("Model-DFlash-Q8_0.gguf").exists());

        fixture
            .apply(LocalModelsConfiguration {
                drafters: Vec::new(),
                ..attached
            })
            .await;
        let pruned = fixture.local.prune_drafters().await.unwrap();
        assert_eq!(pruned.removed.len(), 1);
        assert!(pruned.freed_bytes > 0);
        assert!(!snapshot.join("Model-DFlash-Q8_0.gguf").exists());
        assert!(
            folder.join("target.gguf").exists(),
            "the model's own files are never touched"
        );
    }

    /// Recommends each model its own drafter reference.
    struct RecommendsEach(BTreeMap<ModelId, DrafterReference>);

    impl CustomModelEnrichment for RecommendsEach {
        fn enrich(&self, model: &CustomModelFacts<'_>) -> CustomModelEnrichmentResult {
            CustomModelEnrichmentResult {
                variant_of: None,
                drafters: self
                    .0
                    .get(model.model_id)
                    .map(|reference| DrafterCandidate {
                        category: DrafterCategory::Other,
                        reference: reference.clone(),
                        name: "drafter".to_owned(),
                        label: "Recommended".to_owned(),
                        size_bytes: 1,
                        lineage: DrafterLineage::Lineage,
                        verdict: DrafterVerdict::Bound {
                            method: SpeculativeMethod::DFlash,
                            memory_bytes: 1,
                        },
                        automatic: true,
                        on_this_computer: true,
                        method: Some(SpeculativeMethod::DFlash),
                    })
                    .into_iter()
                    .collect(),
                catalog_drafters: Vec::new(),
                license: None,
            }
        }

        fn knows(&self, _model: &CustomModelFacts<'_>) -> bool {
            true
        }
    }

    struct PruningFixture {
        fixture: Fixture,
        folder: PathBuf,
        models: [ModelId; 2],
        drafters: [(DrafterReference, PathBuf); 2],
    }

    /// Two found models, each recommended its own Magnitude-downloaded drafter.
    async fn pruning_fixture() -> PruningFixture {
        let fixture = Fixture::new().await;
        let folder = fixture.root.join("models");
        write_model(&folder.join("a.gguf"), "model a");
        write_model(&folder.join("b.gguf"), "model b");
        let models = [
            local_id(&folder.join("a.gguf")),
            local_id(&folder.join("b.gguf")),
        ];
        let drafter = |name: &str, seed: char| {
            let repository = format!("magnitudedev/{name}-DFlash-GGUF");
            let commit = seed.to_string().repeat(40);
            let file = format!("{name}-DFlash-Q8_0.gguf");
            let snapshot = publish_managed(
                &fixture.store,
                &repository,
                &commit,
                &[(&file, drafter_bytes(name))],
            );
            (
                DrafterReference::HuggingFace {
                    repository: HuggingFaceRepositoryId::new(repository.as_str()).unwrap(),
                    commit,
                    path: file.clone(),
                },
                snapshot.join(file),
            )
        };
        let drafters = [drafter("A", 'a'), drafter("B", 'b')];
        fixture
            .resolver
            .set_custom_model_enrichment(Arc::new(RecommendsEach(BTreeMap::from([
                (models[0].clone(), drafters[0].0.clone()),
                (models[1].clone(), drafters[1].0.clone()),
            ]))));
        PruningFixture {
            fixture,
            folder,
            models,
            drafters,
        }
    }

    impl PruningFixture {
        fn configuration(&self, choices: [DrafterChoice; 2]) -> LocalModelsConfiguration {
            LocalModelsConfiguration {
                roots: vec![root(ModelRootKind::Folder, &self.folder)],
                drafters: self
                    .models
                    .iter()
                    .cloned()
                    .zip(choices)
                    .map(|(model_id, drafter)| DrafterAttachment { model_id, drafter })
                    .collect(),
                ..LocalModelsConfiguration::default()
            }
        }

        fn reference(&self, index: usize) -> DrafterChoice {
            DrafterChoice::Reference {
                reference: self.drafters[index].0.clone(),
                method: SpeculativeMethod::DFlash,
            }
        }

        async fn remove_model(&self, file: &str) -> DrafterPruning {
            fs::remove_file(self.folder.join(file)).unwrap();
            self.fixture.local.refresh_local().await.unwrap();
            self.fixture.local.prune_drafters().await.unwrap()
        }
    }

    /// Removing one model never deletes a drafter another model here would use: model A's
    /// turned-off drafter stays while A is here, and goes once A is removed and nothing else
    /// references it.
    #[tokio::test]
    async fn pruning_keeps_a_turned_off_drafter_while_its_model_is_here() {
        let pruning = pruning_fixture().await;
        pruning
            .fixture
            .apply(pruning.configuration([DrafterChoice::None, pruning.reference(1)]))
            .await;

        let removed_b = pruning.remove_model("b.gguf").await;
        assert_eq!(removed_b.removed.len(), 1, "B's drafter goes with B");
        assert!(!pruning.drafters[1].1.exists());
        assert!(
            pruning.drafters[0].1.exists(),
            "A's turned-off drafter stays while A is here"
        );

        let removed_a = pruning.remove_model("a.gguf").await;
        assert_eq!(removed_a.removed.len(), 1);
        assert!(
            !pruning.drafters[0].1.exists(),
            "nothing references A's drafter now"
        );
    }

    /// A drafter two models use survives removing one of them.
    #[tokio::test]
    async fn a_drafter_shared_by_two_models_survives_removing_one() {
        let pruning = pruning_fixture().await;
        pruning
            .fixture
            .apply(pruning.configuration([pruning.reference(0), pruning.reference(0)]))
            .await;
        pruning.remove_model("b.gguf").await;
        assert!(
            pruning.drafters[0].1.exists(),
            "A still uses the shared drafter"
        );
    }

    /// A catalog model whose weights are only in a Hugging Face cache, served with its catalog
    /// drafter from Magnitude's store, is still a found copy: its installation is externally
    /// owned, so Remove (which deletes only Magnitude's copy of the weights) has nothing to delete
    /// and Hide is what applies. Hiding it by its found-model ID ends the attribution. Once
    /// Magnitude holds its own copy of the weights too, the installation is mixed and Remove
    /// deletes that copy.
    #[tokio::test]
    async fn an_external_catalog_copy_with_the_catalog_drafter_in_the_store_is_external() {
        use magnitude_service_contracts::models::{
            CatalogModelState, InstalledModelPackages as _, ModelInstallation,
            ModelInstallationOwnership,
        };
        let fixture = Fixture::new().await;
        let cache = fixture.root.join("hf");
        let target_bytes = model_bytes("catalog target");
        publish_hf_cache(
            &cache,
            "owner/Model-GGUF",
            &"a".repeat(40),
            &[("Model-Q4_K_M.gguf", target_bytes.clone())],
        );
        publish_managed(
            &fixture.store,
            "owner/Model-MTP-GGUF",
            &"b".repeat(40),
            &[("Model-MTP.gguf", drafter_bytes("catalog drafter"))],
        );
        let shown = roots(vec![root(ModelRootKind::HuggingFaceCache, &cache)]);
        fixture.apply(shown.clone()).await;
        let packages = fixture.inventory().list_installed().await.unwrap().packages;
        let package = |repository: &str| {
            packages
                .iter()
                .map(|entry| entry.package.clone())
                .find(|package| matches!(
                    &package.source,
                    magnitude_service_contracts::models::ModelPackageSource::HuggingFace { repository: source, .. }
                        if source == repository
                ))
                .unwrap()
        };
        let (target, draft) = (package("owner/Model-GGUF"), package("owner/Model-MTP-GGUF"));
        let mut definition = catalog_definition(target.clone());
        definition.configuration.bundle = ServableModelBundle::SpeculativeDecoding {
            target: target.clone(),
            draft_source: SpeculativeDraftSource::Separate { draft },
            method: SpeculativeMethod::Mtp,
        };
        let catalog_model_id = catalog_id(&definition);
        let catalog = vec![definition];
        let fixture = fixture.restart(catalog.clone()).await;
        fixture.apply(shown.clone()).await;
        let ownership = |fixture: &Fixture| match &fixture.resolver.catalog_snapshot().unwrap().models[0]
            .local_state
        {
            CatalogModelState::Installed {
                installation: ModelInstallation::Resolved { ownership, .. },
                ..
            } => Some(*ownership),
            CatalogModelState::Installed { .. } => panic!("the found target resolves"),
            CatalogModelState::NotInstalled => None,
        };

        assert_eq!(ownership(&fixture), Some(ModelInstallationOwnership::External));
        let plan = fixture
            .resolver
            .catalog_removal_plan(&catalog_model_id)
            .unwrap();
        assert!(plan.installed && plan.externally_owned, "Remove has no copy of the weights to delete");

        let found_id = ModelId::hugging_face(
            &HuggingFaceRepositoryId::new("owner/Model-GGUF").unwrap(),
            &HuggingFaceArtifactSelector::new("Model-Q4_K_M.gguf").unwrap(),
        );
        let listed = fixture
            .apply(LocalModelsConfiguration {
                hidden: vec![found_id.clone()],
                ..shown.clone()
            })
            .await;
        assert_eq!(ids(&listed.hidden), [found_id.to_string()]);
        assert_eq!(ownership(&fixture), None, "hidden, the copy is no longer the catalog model");

        fixture.apply(shown.clone()).await;
        publish_managed(
            &fixture.store,
            "owner/Model-GGUF",
            &"a".repeat(40),
            &[("Model-Q4_K_M.gguf", target_bytes)],
        );
        let fixture = fixture.restart(catalog).await;
        fixture.apply(shown).await;
        assert_eq!(ownership(&fixture), Some(ModelInstallationOwnership::Mixed));
        let plan = fixture
            .resolver
            .catalog_removal_plan(&catalog_model_id)
            .unwrap();
        assert!(!plan.externally_owned);
        assert!(plan.package_ids.contains(&target.id), "Remove deletes Magnitude's copy");
    }

    /// Removing a catalog model deletes its target and keeps its drafter while a found model uses
    /// it; once nothing does, the removal takes the drafter too.
    #[tokio::test]
    async fn removing_a_catalog_model_keeps_a_drafter_another_model_uses() {
        use magnitude_service_contracts::models::InstalledModelPackages as _;
        let fixture = Fixture::new().await;
        publish_managed(
            &fixture.store,
            "owner/Model-GGUF",
            &"a".repeat(40),
            &[("Model-Q4_K_M.gguf", model_bytes("catalog target"))],
        );
        publish_managed(
            &fixture.store,
            "owner/Model-MTP-GGUF",
            &"b".repeat(40),
            &[("Model-MTP.gguf", drafter_bytes("catalog drafter"))],
        );
        fixture.apply(LocalModelsConfiguration::default()).await;
        let packages = fixture.inventory().list_installed().await.unwrap().packages;
        let package = |repository: &str| {
            packages
                .iter()
                .map(|entry| entry.package.clone())
                .find(|package| matches!(
                    &package.source,
                    magnitude_service_contracts::models::ModelPackageSource::HuggingFace { repository: source, .. }
                        if source == repository
                ))
                .unwrap()
        };
        let (target, draft) = (package("owner/Model-GGUF"), package("owner/Model-MTP-GGUF"));
        let mut definition = catalog_definition(target.clone());
        definition.configuration.bundle = ServableModelBundle::SpeculativeDecoding {
            target: target.clone(),
            draft_source: SpeculativeDraftSource::Separate {
                draft: draft.clone(),
            },
            method: SpeculativeMethod::Mtp,
        };
        let catalog_model_id = catalog_id(&definition);
        let fixture = fixture.restart(vec![definition]).await;
        let folder = fixture.root.join("models");
        write_model(&folder.join("found.gguf"), "found");
        let using = LocalModelsConfiguration {
            roots: vec![root(ModelRootKind::Folder, &folder)],
            drafters: vec![DrafterAttachment {
                model_id: local_id(&folder.join("found.gguf")),
                drafter: DrafterChoice::Reference {
                    reference: DrafterReference::Catalog {
                        model_id: catalog_model_id.clone(),
                    },
                    method: SpeculativeMethod::Mtp,
                },
            }],
            ..LocalModelsConfiguration::default()
        };
        fixture.apply(using.clone()).await;

        let plan = fixture
            .resolver
            .catalog_removal_plan(&catalog_model_id)
            .unwrap();
        assert_eq!(plan.package_ids, [target.id.clone()]);

        fixture
            .apply(LocalModelsConfiguration {
                drafters: Vec::new(),
                ..using
            })
            .await;
        let plan = fixture
            .resolver
            .catalog_removal_plan(&catalog_model_id)
            .unwrap();
        assert_eq!(
            plan.package_ids.iter().collect::<BTreeSet<_>>(),
            BTreeSet::from([&target.id, &draft.id])
        );
    }

    /// A catalog drafter with a new revision in the catalog keeps serving with the earlier
    /// revision installed for that catalog model, and reports that an update is available.
    #[tokio::test]
    async fn a_catalog_drafter_with_a_new_revision_serves_the_earlier_one_with_an_update() {
        use magnitude_service_contracts::models::InstalledModelPackages as _;
        let fixture = Fixture::new().await;
        publish_managed(
            &fixture.store,
            "owner/Model-GGUF",
            &"a".repeat(40),
            &[("Model-Q4_K_M.gguf", model_bytes("catalog target"))],
        );
        publish_managed(
            &fixture.store,
            "owner/Model-MTP-GGUF",
            &"b".repeat(40),
            &[("Model-MTP.gguf", drafter_bytes("catalog drafter"))],
        );
        fixture.apply(LocalModelsConfiguration::default()).await;
        let packages = fixture.inventory().list_installed().await.unwrap().packages;
        let package = |repository: &str| {
            packages
                .iter()
                .map(|entry| entry.package.clone())
                .find(|package| matches!(
                    &package.source,
                    magnitude_service_contracts::models::ModelPackageSource::HuggingFace { repository: source, .. }
                        if source == repository
                ))
                .unwrap()
        };
        let (target, old_draft) = (package("owner/Model-GGUF"), package("owner/Model-MTP-GGUF"));
        let with_draft = |draft: ModelPackage| {
            let mut definition = catalog_definition(target.clone());
            definition.configuration.bundle = ServableModelBundle::SpeculativeDecoding {
                target: target.clone(),
                draft_source: SpeculativeDraftSource::Separate { draft },
                method: SpeculativeMethod::Mtp,
            };
            definition
        };
        let catalog_model_id = catalog_id(&with_draft(old_draft.clone()));
        // The installed drafter is recorded as the catalog model's drafter while it is current, and
        // a found model using the current drafter has no update.
        let fixture = fixture.restart(vec![with_draft(old_draft.clone())]).await;
        let folder = fixture.root.join("models");
        write_model(&folder.join("found.gguf"), "found");
        let found = local_id(&folder.join("found.gguf"));
        let using = LocalModelsConfiguration {
            roots: vec![root(ModelRootKind::Folder, &folder)],
            drafters: vec![DrafterAttachment {
                model_id: found.clone(),
                drafter: DrafterChoice::Reference {
                    reference: DrafterReference::Catalog {
                        model_id: catalog_model_id.clone(),
                    },
                    method: SpeculativeMethod::Mtp,
                },
            }],
            ..LocalModelsConfiguration::default()
        };
        let current = fixture.apply(using.clone()).await;
        let attached = current
            .models
            .iter()
            .find(|model| model.model_id == found)
            .and_then(|model| model.drafter.clone())
            .expect("the attached drafter");
        assert_eq!(attached.state, AttachedDrafterState::Ready);
        assert!(
            !attached.update_available,
            "a current drafter has no update"
        );

        let mut new_draft = old_draft.clone();
        new_draft.id = ModelPackageId("owner-model-mtp-next-revision".to_owned());
        for file in &mut new_draft.files {
            file.sha256 = "f".repeat(64);
        }
        if let magnitude_service_contracts::models::ModelPackageSource::HuggingFace {
            revision,
            ..
        } = &mut new_draft.source
        {
            *revision = "c".repeat(40);
        }
        let fixture = fixture.restart(vec![with_draft(new_draft)]).await;
        let listed = fixture.apply(using).await;

        let drafter = listed
            .models
            .iter()
            .find(|model| model.model_id == found)
            .and_then(|model| model.drafter.clone())
            .expect("the attached drafter");
        assert_eq!(drafter.state, AttachedDrafterState::Ready);
        assert!(drafter.update_available, "{drafter:?}");
        assert!(matches!(
            fixture.resolver.serving_configuration(&found).unwrap().bundle,
            ServableModelBundle::SpeculativeDecoding {
                draft_source: SpeculativeDraftSource::Separate { draft },
                ..
            } if draft.id == old_draft.id
        ));
    }

    /// Recommends each model the drafter beside it and reports it matching one catalog model, as
    /// enrichment does for a found or imported model with a drafter in its folder.
    struct BesideAndCatalog {
        beside: DrafterReference,
        catalog: Option<ModelId>,
    }

    impl CustomModelEnrichment for BesideAndCatalog {
        fn enrich(&self, _model: &CustomModelFacts<'_>) -> CustomModelEnrichmentResult {
            CustomModelEnrichmentResult {
                variant_of: self.catalog.clone(),
                drafters: vec![DrafterCandidate {
                    category: DrafterCategory::NextToModel,
                    reference: self.beside.clone(),
                    name: "mtp.gguf".to_owned(),
                    label: "In this folder".to_owned(),
                    size_bytes: 1,
                    lineage: DrafterLineage::Lineage,
                    verdict: DrafterVerdict::Bound {
                        method: SpeculativeMethod::Mtp,
                        memory_bytes: 1,
                    },
                    automatic: true,
                    on_this_computer: true,
                    method: Some(SpeculativeMethod::Mtp),
                }],
                catalog_drafters: self.catalog.iter().cloned().collect(),
                license: None,
            }
        }
    }

    /// The arm64 QA case: removing an added model pruned the catalog drafter of a model still on
    /// this computer, because that model's recommended drafter was the one beside it. A model's
    /// catalog drafter is kept while the model is here, whatever it recommends; with no model
    /// matching that catalog model, the drafter goes.
    #[tokio::test]
    async fn pruning_keeps_the_catalog_drafter_of_a_model_here_that_recommends_another() {
        use magnitude_service_contracts::models::InstalledModelPackages as _;
        let fixture = Fixture::new().await;
        let drafter = publish_managed(
            &fixture.store,
            "owner/Model-MTP-GGUF",
            &"b".repeat(40),
            &[("Model-MTP.gguf", drafter_bytes("catalog drafter"))],
        )
        .join("Model-MTP.gguf");
        fixture.apply(LocalModelsConfiguration::default()).await;
        let draft = fixture
            .inventory()
            .list_installed()
            .await
            .unwrap()
            .packages
            .into_iter()
            .map(|entry| entry.package)
            .next()
            .expect("the catalog drafter package");
        let mut target = draft.clone();
        target.id = ModelPackageId("owner-model-target-not-here".to_owned());
        for file in &mut target.files {
            file.id = magnitude_service_contracts::models::ModelFileId("file-target-not-here".to_owned());
        }
        let mut definition = catalog_definition(target.clone());
        definition.configuration.bundle = ServableModelBundle::SpeculativeDecoding {
            target,
            draft_source: SpeculativeDraftSource::Separate { draft },
            method: SpeculativeMethod::Mtp,
        };
        let catalog_model = catalog_id(&definition);
        let fixture = fixture.restart(vec![definition]).await;

        let folder = fixture.root.join("imported");
        write_model(&folder.join("model.gguf"), "imported");
        write_dflash_drafter(&folder.join("mtp.gguf"), "beside");
        let beside = DrafterReference::Local {
            path: folder.join("mtp.gguf"),
        };
        let enrichment = |catalog: Option<ModelId>| {
            Arc::new(BesideAndCatalog {
                beside: beside.clone(),
                catalog,
            })
        };
        fixture
            .resolver
            .set_custom_model_enrichment(enrichment(Some(catalog_model)));
        let configuration = LocalModelsConfiguration {
            roots: vec![root(ModelRootKind::Folder, &folder)],
            ..LocalModelsConfiguration::default()
        };
        fixture.apply(configuration.clone()).await;
        let kept = fixture.local.prune_drafters().await.unwrap();
        assert!(kept.removed.is_empty(), "{kept:?}");
        assert!(drafter.exists(), "the catalog drafter of a model here is kept");

        fixture.resolver.set_custom_model_enrichment(enrichment(None));
        fixture.apply(configuration).await;
        let pruned = fixture.local.prune_drafters().await.unwrap();
        assert_eq!(pruned.removed.len(), 1, "no model here uses it any more");
        assert!(!drafter.exists());
        assert!(folder.join("mtp.gguf").exists(), "files outside the store are never touched");
    }

    /// After a restart mid-download, the drafter that finished before the target is kept for the
    /// target still arriving, so the resumed download completes the whole model.
    #[tokio::test]
    async fn the_drafter_of_an_added_model_still_downloading_is_not_pruned() {
        let fixture = Fixture::new().await;
        let drafter_commit = "e".repeat(40);
        let snapshot = publish_managed(
            &fixture.store,
            "magnitudedev/Model-DFlash-GGUF",
            &drafter_commit,
            &[("Model-DFlash-Q8_0.gguf", drafter_bytes("drafter"))],
        );
        let id: ModelId = "hf:owner/Pending-GGUF/Pending-Q4_K_M.gguf".parse().unwrap();
        let configuration = LocalModelsConfiguration {
            declarations: vec![CustomModelDeclaration {
                model_id: id.clone(),
                origin: CustomModelDeclarationOrigin::Added {
                    repository: HuggingFaceRepositoryId::new("owner/Pending-GGUF").unwrap(),
                    commit: "c".repeat(40),
                },
                target: vec!["Pending-Q4_K_M.gguf".to_owned()],
                projector: None,
            }],
            drafters: vec![DrafterAttachment {
                model_id: id.clone(),
                drafter: DrafterChoice::Reference {
                    reference: DrafterReference::HuggingFace {
                        repository: HuggingFaceRepositoryId::new("magnitudedev/Model-DFlash-GGUF")
                            .unwrap(),
                        commit: drafter_commit,
                        path: "Model-DFlash-Q8_0.gguf".to_owned(),
                    },
                    method: SpeculativeMethod::DFlash,
                },
            }],
            ..LocalModelsConfiguration::default()
        };
        let pending = fixture.apply(configuration.clone()).await;
        assert_eq!(pending.models[0].state, CustomModelState::NotInstalled);

        let kept = fixture.local.prune_drafters().await.unwrap();
        assert!(
            kept.removed.is_empty(),
            "the pending model's drafter is in use"
        );
        assert!(snapshot.join("Model-DFlash-Q8_0.gguf").exists());

        fixture
            .apply(LocalModelsConfiguration {
                declarations: Vec::new(),
                drafters: Vec::new(),
                ..configuration
            })
            .await;
        let pruned = fixture.local.prune_drafters().await.unwrap();
        assert_eq!(
            pruned.removed.len(),
            1,
            "once the model is removed its drafter goes too"
        );
    }

    /// A Qwen3.5 4B quant with its MTP block, as unsloth's Qwen3.5-4B-MTP-GGUF ships them: every
    /// quant has the same name and file type, so only `description` tells them apart.
    fn qwen35_mtp_quant(description: &str) -> Vec<u8> {
        gguf_bytes(
            &[
                ("general.architecture", Meta::Str("qwen35")),
                ("general.name", Meta::Str("Qwen3.5-4B")),
                ("general.description", Meta::Str(description)),
                ("general.file_type", Meta::U32(18)),
                ("qwen35.block_count", Meta::U32(2)),
                ("qwen35.nextn_predict_layers", Meta::U32(1)),
            ],
            &[
                "token_embd.weight",
                "blk.0.attn_q.weight",
                "blk.1.attn_q.weight",
                "blk.1.nextn.eh_proj.weight",
            ],
        )
    }

    struct CatalogSibling {
        fixture: Fixture,
        installations: Arc<crate::catalog_installations::ManagedCatalogInstallations>,
        catalog_model: ModelId,
        target: ModelPackage,
        draft: ModelPackage,
        catalog_weights: PathBuf,
        sibling: CustomModelDeclaration,
    }

    const SIBLING_REPOSITORY: &str = "unsloth/Qwen3.5-4B-MTP-GGUF";

    /// The Arch round 3b setup: the catalog model installed (UD-Q6_K_XL with built-in heads and
    /// mmproj-BF16, plus the catalog DFlash), then the non-target Q6_K added from the same snapshot
    /// with mmproj-F16. With two projectors in the snapshot, the catalog weights are also found
    /// on their own, as a second package of the same file.
    async fn catalog_with_an_added_sibling() -> CatalogSibling {
        use magnitude_service_contracts::models::{InstalledModelPackages as _, ModelFileRole};
        let commit = "8".repeat(40);
        let draft_commit = "0".repeat(40);
        let weights = qwen35_mtp_quant("UD-Q6_K_XL");
        let projector = |name: &str| {
            gguf_bytes(
                &[
                    ("general.architecture", Meta::Str("clip")),
                    ("general.name", Meta::Str(name)),
                ],
                &["v.patch_embd.weight"],
            )
        };
        let drafter = drafter_bytes("Qwen3.5-4B-DFlash");
        let catalog_files = [
            ("Qwen3.5-4B-UD-Q6_K_XL.gguf", weights),
            ("mmproj-BF16.gguf", projector("bf16")),
        ];
        let drafter_files = [("Qwen3.5-4B-DFlash-Q8_0.gguf", drafter)];

        let release = Fixture::new().await;
        publish_managed(&release.store, SIBLING_REPOSITORY, &commit, &catalog_files);
        publish_managed(
            &release.store,
            "magnitudedev/Qwen3.5-4B-DFlash-GGUF",
            &draft_commit,
            &drafter_files,
        );
        release.apply(LocalModelsConfiguration::default()).await;
        let packages = release.inventory().list_installed().await.unwrap().packages;
        let target = packages
            .iter()
            .map(|installed| installed.package.clone())
            .find(|package| package.files.iter().any(|file| file.role == ModelFileRole::Projector))
            .expect("the catalog target pairs its projector");
        let draft = packages
            .iter()
            .map(|installed| installed.package.clone())
            .find(|package| package.id != target.id)
            .expect("the catalog drafter");
        let mut definition = catalog_definition(target.clone());
        definition.configuration.bundle = ServableModelBundle::SpeculativeDecoding {
            target: target.clone(),
            draft_source: SpeculativeDraftSource::Separate {
                draft: draft.clone(),
            },
            method: SpeculativeMethod::DFlash,
        };
        let catalog_model = catalog_id(&definition);

        let fixture = Fixture::new().await.restart(vec![definition]).await;
        let snapshot =
            publish_managed(&fixture.store, SIBLING_REPOSITORY, &commit, &catalog_files);
        publish_managed(
            &fixture.store,
            "magnitudedev/Qwen3.5-4B-DFlash-GGUF",
            &draft_commit,
            &drafter_files,
        );
        fixture.apply(LocalModelsConfiguration::default()).await;
        publish_managed(
            &fixture.store,
            SIBLING_REPOSITORY,
            &commit,
            &[
                ("Qwen3.5-4B-Q6_K.gguf", qwen35_mtp_quant("Q6_K")),
                ("mmproj-F16.gguf", projector("f16")),
            ],
        );
        let sibling = CustomModelDeclaration {
            model_id: format!("hf:{SIBLING_REPOSITORY}/Qwen3.5-4B-Q6_K.gguf")
                .parse()
                .unwrap(),
            origin: CustomModelDeclarationOrigin::Added {
                repository: HuggingFaceRepositoryId::new(SIBLING_REPOSITORY).unwrap(),
                commit,
            },
            target: vec!["Qwen3.5-4B-Q6_K.gguf".to_owned()],
            projector: Some("mmproj-F16.gguf".to_owned()),
        };
        let installations = crate::catalog_installations::ManagedCatalogInstallations::new(
            fixture.resolver.clone(),
            Arc::new(
                crate::ManagedModelDownloads::open(fixture.resolver.inventory.clone())
                    .await
                    .unwrap(),
            ),
            Arc::new(StoreRemover(fixture.resolver.inventory.clone())),
            Arc::new(NoOptimizer),
        );
        CatalogSibling {
            fixture,
            installations,
            catalog_model,
            target,
            draft,
            catalog_weights: snapshot.join("Qwen3.5-4B-UD-Q6_K_XL.gguf"),
            sibling,
        }
    }

    impl CatalogSibling {
        fn catalog_installed(&self) -> bool {
            use magnitude_service_contracts::models::CatalogModelState;
            matches!(
                self.fixture.resolver.catalog_snapshot().unwrap().models[0].local_state,
                CatalogModelState::Installed { .. }
            )
        }

        /// The sibling declared with `drafter`, then everything that removes packages afterwards:
        /// catalog maintenance and drafter pruning.
        async fn choose_sibling_drafter(&self, drafter: DrafterChoice) {
            self.fixture
                .apply(LocalModelsConfiguration {
                    declarations: vec![self.sibling.clone()],
                    drafters: vec![DrafterAttachment {
                        model_id: self.sibling.model_id.clone(),
                        drafter,
                    }],
                    ..LocalModelsConfiguration::default()
                })
                .await;
            self.installations
                .cleanup_model(&self.catalog_model)
                .await
                .unwrap();
            self.fixture.local.prune_drafters().await.unwrap();
        }
    }

    /// The Arch round 3b BLOCKER: turning the added Q6_K's drafter off (or to its built-in heads)
    /// deleted the catalog model's UD-Q6_K_XL. Catalog maintenance took the weights found on their
    /// own (with two projectors in the snapshot they pair with neither) for a superseded revision
    /// of the catalog model and deleted the file the catalog target serves from. The catalog
    /// target's files are never deleted, whatever the sibling's drafter.
    #[tokio::test(flavor = "multi_thread")]
    async fn an_added_siblings_drafter_change_never_deletes_the_catalog_target() {
        use magnitude_service_contracts::models::InstalledModelPackages as _;
        for drafter in [
            DrafterChoice::Reference {
                reference: DrafterReference::Embedded,
                method: SpeculativeMethod::Mtp,
            },
            DrafterChoice::None,
        ] {
            let setup = catalog_with_an_added_sibling().await;
            assert!(setup.catalog_installed());
            setup.choose_sibling_drafter(drafter.clone()).await;
            let packages = setup.fixture.inventory().list_installed().await.unwrap().packages;
            assert!(
                packages.iter().any(|installed| installed.package.id != setup.target.id
                    && installed
                        .package
                        .files
                        .iter()
                        .any(|file| setup.target.files.iter().any(|held| held.id == file.id
                            && file.role == ModelFileRole::Weights))),
                "the catalog weights are also found on their own"
            );
            assert!(
                setup.catalog_weights.exists(),
                "the catalog target's weights are kept with the sibling's drafter {drafter:?}"
            );
            setup.fixture.local.refresh_local().await.unwrap();
            assert!(setup.catalog_installed(), "{drafter:?}");
            assert!(
                packages.iter().any(|installed| installed.package.id == setup.draft.id),
                "the installed catalog model's drafter is kept"
            );
        }
    }

    /// Drafter pruning's own check at the deletion, whatever the reference accounting says: it
    /// refuses the catalog target, the catalog weights found on their own (a full model with
    /// built-in heads, not a drafter), and the drafter of an installed catalog model.
    #[tokio::test(flavor = "multi_thread")]
    async fn drafter_pruning_refuses_target_weights_and_an_installed_catalog_bundle() {
        use magnitude_service_contracts::models::InstalledModelPackages as _;
        let setup = catalog_with_an_added_sibling().await;
        setup
            .fixture
            .apply(LocalModelsConfiguration {
                declarations: vec![setup.sibling.clone()],
                ..LocalModelsConfiguration::default()
            })
            .await;
        let weights_alone = setup
            .fixture
            .inventory()
            .list_installed()
            .await
            .unwrap()
            .packages
            .into_iter()
            .map(|installed| installed.package)
            .find(|package| {
                package.id != setup.target.id
                    && package.files.iter().any(|file| {
                        file.role == ModelFileRole::Weights
                            && setup.target.files.iter().any(|held| held.id == file.id)
                    })
            })
            .expect("the catalog weights found on their own");
        let refusal =
            |package: &ModelPackage| setup.fixture.resolver.drafter_prune_refusal(&package.id).unwrap();
        assert_eq!(refusal(&setup.target), Some("it is not a drafter-only package"));
        assert_eq!(refusal(&weights_alone), Some("it is not a drafter-only package"));
        assert_eq!(refusal(&setup.draft), Some("an installed catalog model uses it"));
        let pruned = setup.fixture.local.prune_drafters().await.unwrap();
        assert!(pruned.removed.is_empty(), "{pruned:?}");
        assert!(setup.catalog_weights.exists());
    }

    /// The residency refuses removing a package a resident model has loaded.
    struct LoadedRemover;

    impl CatalogPackageRemover for LoadedRemover {
        fn remove_catalog_packages(
            &self,
            package_ids: Vec<ModelPackageId>,
        ) -> futures_util::future::BoxFuture<'_, Result<u64, InventoryError>> {
            Box::pin(async move { Err(InventoryError::Loaded(package_ids[0].0.clone())) })
        }
    }

    /// Removing an added model deletes its own files and the separate drafter it downloaded,
    /// once nothing uses it, unless a resident model has that drafter loaded. The catalog model
    /// beside it in the same snapshot keeps its target and its drafter.
    #[tokio::test(flavor = "multi_thread")]
    async fn removing_an_added_model_deletes_its_unused_drafter_and_nothing_of_the_catalog_models() {
        let setup = catalog_with_an_added_sibling().await;
        let drafter_commit = "d".repeat(40);
        let drafter = publish_managed(
            &setup.fixture.store,
            "magnitudedev/Other-DFlash-GGUF",
            &drafter_commit,
            &[("Other-DFlash-Q8_0.gguf", drafter_bytes("other drafter"))],
        )
        .join("Other-DFlash-Q8_0.gguf");
        let with_sibling = LocalModelsConfiguration {
            declarations: vec![setup.sibling.clone()],
            drafters: vec![DrafterAttachment {
                model_id: setup.sibling.model_id.clone(),
                drafter: DrafterChoice::Reference {
                    reference: DrafterReference::HuggingFace {
                        repository: HuggingFaceRepositoryId::new("magnitudedev/Other-DFlash-GGUF")
                            .unwrap(),
                        commit: drafter_commit,
                        path: "Other-DFlash-Q8_0.gguf".to_owned(),
                    },
                    method: SpeculativeMethod::DFlash,
                },
            }],
            ..LocalModelsConfiguration::default()
        };
        setup.fixture.apply(with_sibling).await;
        let kept = setup.fixture.local.prune_drafters().await.unwrap();
        assert!(kept.removed.is_empty(), "the sibling serves with it: {kept:?}");

        setup
            .fixture
            .local
            .remove_local(&setup.sibling.model_id)
            .await
            .unwrap();
        setup.fixture.apply(LocalModelsConfiguration::default()).await;
        setup
            .installations
            .cleanup_model(&setup.catalog_model)
            .await
            .unwrap();
        let loaded = ManagedLocalModels::new(setup.fixture.resolver.clone(), Arc::new(LoadedRemover));
        let refused = loaded.prune_drafters().await.unwrap();
        assert!(refused.removed.is_empty(), "a loaded drafter is kept: {refused:?}");
        assert!(drafter.exists());

        let pruned = setup.fixture.local.prune_drafters().await.unwrap();
        assert_eq!(pruned.removed.len(), 1, "{pruned:?}");
        assert!(!drafter.exists(), "the removed model's drafter goes");
        assert!(setup.catalog_weights.exists());
        assert!(setup.catalog_installed());
        use magnitude_service_contracts::models::InstalledModelPackages as _;
        assert!(
            setup
                .fixture
                .inventory()
                .list_installed()
                .await
                .unwrap()
                .packages
                .iter()
                .any(|installed| installed.package.id == setup.draft.id),
            "the catalog model's drafter stays"
        );
    }

    #[tokio::test]
    async fn built_in_prediction_heads_attach_as_chosen_and_the_engine_decides_whether_they_bind() {
        let fixture = Fixture::new().await;
        let folder = fixture.root.join("models");
        write_mtp_model(&folder.join("with-heads.gguf"), "heads");
        write_model(&folder.join("without-heads.gguf"), "plain");
        let with_heads = local_id(&folder.join("with-heads.gguf"));
        let without_heads = local_id(&folder.join("without-heads.gguf"));
        let embedded = |model_id: &ModelId| DrafterAttachment {
            model_id: model_id.clone(),
            drafter: DrafterChoice::Reference {
                reference: DrafterReference::Embedded,
                method: SpeculativeMethod::Mtp,
            },
        };

        fixture
            .apply(LocalModelsConfiguration {
                roots: vec![root(ModelRootKind::Folder, &folder)],
                drafters: vec![embedded(&with_heads), embedded(&without_heads)],
                ..LocalModelsConfiguration::default()
            })
            .await;

        // Whether built-in heads bind is the engine's drafter verdict for the model's assessment
        // ("MTP was requested but the target has no draft head"); the service attaches them as
        // chosen and never judges the file itself.
        for model_id in [&with_heads, &without_heads] {
            assert!(matches!(
                fixture
                    .resolver
                    .serving_configuration(model_id)
                    .unwrap()
                    .bundle,
                ServableModelBundle::SpeculativeDecoding {
                    draft_source: SpeculativeDraftSource::Embedded,
                    method: SpeculativeMethod::Mtp,
                    ..
                }
            ));
        }
        let listed = fixture.local.list_local().await.unwrap();
        for model_id in [&with_heads, &without_heads] {
            let drafter = listed
                .models
                .iter()
                .find(|model| &model.model_id == model_id)
                .and_then(|model| model.drafter.as_ref())
                .unwrap();
            assert_eq!(drafter.state, AttachedDrafterState::Ready);
        }
    }

    #[tokio::test]
    async fn turning_the_drafter_off_serves_a_model_with_heads_without_them() {
        let fixture = Fixture::new().await;
        let folder = fixture.root.join("models");
        write_mtp_model(&folder.join("with-heads.gguf"), "heads");
        let with_heads = local_id(&folder.join("with-heads.gguf"));

        fixture
            .apply(LocalModelsConfiguration {
                roots: vec![root(ModelRootKind::Folder, &folder)],
                drafters: vec![DrafterAttachment {
                    model_id: with_heads.clone(),
                    drafter: DrafterChoice::None,
                }],
                ..LocalModelsConfiguration::default()
            })
            .await;

        assert!(
            matches!(
                fixture
                    .resolver
                    .serving_configuration(&with_heads)
                    .unwrap()
                    .bundle,
                ServableModelBundle::Standalone { .. }
            ),
            "no drafter is a standalone bundle, which the engine serves plain"
        );
        let listed = fixture.local.list_local().await.unwrap();
        let model = listed
            .models
            .iter()
            .find(|model| model.model_id == with_heads)
            .unwrap();
        assert!(model.drafter.is_none());
    }

    struct Recommends {
        drafter: PathBuf,
        variant_of: ModelId,
        method: SpeculativeMethod,
    }

    impl CustomModelEnrichment for Recommends {
        fn enrich(&self, _model: &CustomModelFacts<'_>) -> CustomModelEnrichmentResult {
            CustomModelEnrichmentResult {
                variant_of: Some(self.variant_of.clone()),
                drafters: vec![DrafterCandidate {
                    category: DrafterCategory::NextToModel,
                    reference: DrafterReference::Local {
                        path: self.drafter.clone(),
                    },
                    name: "drafter.gguf".to_owned(),
                    label: "In this folder".to_owned(),
                    size_bytes: 1,
                    lineage: DrafterLineage::Lineage,
                    verdict: DrafterVerdict::None,
                    automatic: false,
                    on_this_computer: true,
                    method: Some(self.method.clone()),
                }],
                catalog_drafters: Vec::new(),
                license: None,
            }
        }
    }

    #[tokio::test]
    async fn the_enrichment_fills_lineage_and_the_automatic_drafter() {
        let fixture = Fixture::new().await;
        let folder = fixture.root.join("models");
        let drafters = fixture.root.join("drafters");
        write_model(&folder.join("target.gguf"), "target");
        write_dflash_drafter(&drafters.join("drafter.gguf"), "drafter");
        let target = local_id(&folder.join("target.gguf"));
        let variant_of: ModelId = "qwen3.6-35b-a3b:gguf:q4".parse().unwrap();
        let mut changes = fixture.local.watch_local();
        futures_util::StreamExt::next(&mut changes).await;

        fixture
            .resolver
            .set_custom_model_enrichment(Arc::new(Recommends {
                drafter: drafters.join("drafter.gguf"),
                variant_of: variant_of.clone(),
                method: SpeculativeMethod::DFlash,
            }));
        assert!(futures_util::StreamExt::next(&mut changes).await.is_some());
        fixture
            .apply(LocalModelsConfiguration {
                roots: vec![root(ModelRootKind::Folder, &folder)],
                // A drafter at a path outside every location is observed because a choice names it.
                drafters: vec![DrafterAttachment {
                    model_id: target.clone(),
                    drafter: DrafterChoice::Reference {
                        reference: DrafterReference::Local {
                            path: drafters.join("drafter.gguf"),
                        },
                        method: SpeculativeMethod::DFlash,
                    },
                }],
                ..LocalModelsConfiguration::default()
            })
            .await;
        assert_eq!(
            fixture.local.list_local().await.unwrap().models[0]
                .drafter
                .as_ref()
                .map(|drafter| drafter.state.clone()),
            Some(AttachedDrafterState::Ready)
        );

        // Without a choice the model uses the recommended drafter.
        fixture
            .apply(roots(vec![
                root(ModelRootKind::Folder, &folder),
                root(ModelRootKind::Folder, &drafters),
            ]))
            .await;

        let listed = fixture.local.list_local().await.unwrap();
        assert_eq!(ids(&listed.models), [target.to_string()]);
        assert_eq!(listed.models[0].variant_of, Some(variant_of));
        assert_eq!(
            listed.models[0]
                .drafter
                .as_ref()
                .map(|drafter| drafter.state.clone()),
            Some(AttachedDrafterState::Ready)
        );
        assert!(matches!(
            fixture
                .resolver
                .serving_configuration(&target)
                .unwrap()
                .bundle,
            ServableModelBundle::SpeculativeDecoding { .. }
        ));
    }

    #[tokio::test]
    async fn a_missing_drafter_is_not_installed_and_none_turns_the_recommendation_off() {
        let fixture = Fixture::new().await;
        let folder = fixture.root.join("models");
        let drafters = fixture.root.join("drafters");
        write_model(&folder.join("target.gguf"), "target");
        write_dflash_drafter(&drafters.join("drafter.gguf"), "drafter");
        let target = local_id(&folder.join("target.gguf"));
        fixture
            .resolver
            .set_custom_model_enrichment(Arc::new(Recommends {
                drafter: drafters.join("drafter.gguf"),
                variant_of: "qwen3.6-35b-a3b:gguf:q4".parse().unwrap(),
                method: SpeculativeMethod::DFlash,
            }));
        let attach = |drafter: DrafterChoice| LocalModelsConfiguration {
            roots: vec![
                root(ModelRootKind::Folder, &folder),
                root(ModelRootKind::Folder, &drafters),
            ],
            drafters: vec![DrafterAttachment {
                model_id: target.clone(),
                drafter,
            }],
            ..LocalModelsConfiguration::default()
        };

        fixture
            .apply(attach(DrafterChoice::Reference {
                reference: DrafterReference::Local {
                    path: drafters.join("absent.gguf"),
                },
                method: SpeculativeMethod::DFlash,
            }))
            .await;
        let listed = fixture.local.list_local().await.unwrap();
        assert_eq!(
            listed.models[0]
                .drafter
                .as_ref()
                .map(|drafter| (drafter.method.clone(), drafter.state.clone())),
            Some((
                SpeculativeMethod::DFlash,
                AttachedDrafterState::NotInstalled
            ))
        );
        assert!(matches!(
            fixture
                .resolver
                .serving_configuration(&target)
                .unwrap()
                .bundle,
            ServableModelBundle::Standalone { .. }
        ));

        fixture.apply(attach(DrafterChoice::None)).await;
        let listed = fixture.local.list_local().await.unwrap();
        assert_eq!(listed.models[0].drafter, None);
        assert_eq!(
            listed.models[0].recommended_drafter, None,
            "a model whose drafter is turned off is offered none"
        );
    }

    #[tokio::test]
    async fn configuration_is_whole_and_idempotent() {
        let fixture = Fixture::new().await;
        let folder = fixture.root.join("models");
        write_model(&folder.join("model.gguf"), "model");
        let configuration = roots(vec![root(ModelRootKind::Folder, &folder)]);

        fixture
            .local
            .configure(configuration.clone())
            .await
            .unwrap();
        let first = fixture.local.list_local().await.unwrap();
        fixture
            .local
            .configure(configuration.clone())
            .await
            .unwrap();
        let second = fixture.local.list_local().await.unwrap();

        assert_eq!(fixture.local.configuration().await.unwrap(), configuration);
        assert_eq!(first.models.len(), 1);
        assert_eq!(
            first.revision, second.revision,
            "applying the same snapshot changes nothing"
        );

        let disabled = LocalModelsConfiguration {
            roots: vec![ModelRoot {
                enabled: false,
                ..root(ModelRootKind::Folder, &folder)
            }],
            ..LocalModelsConfiguration::default()
        };
        fixture.local.configure(disabled).await.unwrap();
        assert!(fixture.local.list_local().await.unwrap().models.is_empty());
        assert!(matches!(
            fixture
                .local
                .configure(roots(vec![root(
                    ModelRootKind::Folder,
                    Path::new("relative")
                )]))
                .await,
            Err(InventoryError::InvalidRequest(_))
        ));
    }

    #[tokio::test]
    async fn a_scoped_rescan_reads_only_its_locations() {
        let fixture = Fixture::new().await;
        let first = fixture.root.join("first");
        let second = fixture.root.join("second");
        fs::create_dir_all(&first).unwrap();
        fs::create_dir_all(&second).unwrap();
        fixture
            .apply(roots(vec![
                root(ModelRootKind::Folder, &first),
                root(ModelRootKind::Folder, &second),
            ]))
            .await;
        write_model(&first.join("one.gguf"), "one");
        write_model(&second.join("two.gguf"), "two");

        fixture
            .inventory()
            .rescan_model_roots(BTreeSet::from([first.clone()]))
            .await
            .unwrap();
        assert_eq!(
            ids(&fixture.local.list_local().await.unwrap().models),
            [local_id(&first.join("one.gguf")).to_string()]
        );

        fs::remove_file(first.join("one.gguf")).unwrap();
        fixture
            .inventory()
            .rescan_model_roots(BTreeSet::from([second.clone()]))
            .await
            .unwrap();
        assert_eq!(
            ids(&fixture.local.list_local().await.unwrap().models),
            [local_id(&second.join("two.gguf")).to_string()],
            "a location whose files vanished is rescanned rather than reused"
        );
    }

    async fn eventually(
        local: &ManagedLocalModels,
        condition: impl Fn(&LocalModelsResponse) -> bool,
    ) -> bool {
        for _ in 0..100 {
            if condition(&local.list_local().await.unwrap()) {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        false
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn watched_locations_add_and_remove_models_without_a_refresh() {
        let fixture = Fixture::new().await;
        let folder = fixture.root.join("models");
        fs::create_dir_all(&folder).unwrap();
        let later = fixture.root.join("later");
        fixture
            .local
            .configure(roots(vec![
                root(ModelRootKind::Folder, &folder),
                root(ModelRootKind::Folder, &later),
            ]))
            .await
            .unwrap();
        assert!(
            fixture
                .local
                .watched()
                .iter()
                .any(|target| target.path == folder && target.recursive)
        );
        tokio::time::sleep(Duration::from_millis(500)).await;

        fs::write(folder.join("arriving.gguf.part"), b"partial").unwrap();
        write_model(&folder.join("arriving.gguf"), "arriving");
        assert!(eventually(&fixture.local, |listed| listed.models.len() == 1).await);

        fs::remove_file(folder.join("arriving.gguf")).unwrap();
        assert!(eventually(&fixture.local, |listed| listed.models.is_empty()).await);

        // A location created after it was configured is noticed too.
        write_model(&later.join("model.gguf"), "later");
        assert!(eventually(&fixture.local, |listed| listed.models.len() == 1).await);
    }

    /// A location that could not be read (a permission it lacked, say) is scanned as soon as it
    /// can be, without a restart.
    #[tokio::test]
    async fn a_location_that_becomes_readable_is_scanned_without_a_restart() {
        use std::os::unix::fs::PermissionsExt as _;
        let fixture = Fixture::new().await;
        let folder = fixture.root.join("models");
        write_model(&folder.join("model.gguf"), "model");
        fs::set_permissions(&folder, fs::Permissions::from_mode(0o000)).unwrap();
        if fs::read_dir(&folder).is_ok() {
            // Running as a user that reads everything (root): nothing to observe.
            fs::set_permissions(&folder, fs::Permissions::from_mode(0o755)).unwrap();
            return;
        }
        let listed = fixture
            .apply(roots(vec![root(ModelRootKind::Folder, &folder)]))
            .await;
        assert!(listed.models.is_empty());
        Arc::clone(&fixture.local).recheck_roots(false).await;

        fs::set_permissions(&folder, fs::Permissions::from_mode(0o755)).unwrap();
        Arc::clone(&fixture.local).recheck_roots(false).await;
        assert_eq!(fixture.local.list_local().await.unwrap().models.len(), 1);
    }

    /// A location whose watch could not start is rescanned periodically instead.
    #[tokio::test]
    async fn a_location_without_a_watch_is_rescanned_periodically() {
        let fixture = Fixture::new().await;
        let folder = fixture.root.join("models");
        fs::create_dir_all(&folder).unwrap();
        fixture
            .apply(roots(vec![root(ModelRootKind::Folder, &folder)]))
            .await;
        // As when the platform refuses the watch: no watcher, the location recorded as unwatched.
        *fixture.local.watcher.lock().unwrap() = None;
        *fixture.local.unwatched.lock().unwrap() =
            BTreeSet::from([crate::inventory::canonical_or_self(&folder)]);
        write_model(&folder.join("model.gguf"), "model");

        Arc::clone(&fixture.local).recheck_roots(false).await;
        assert!(fixture.local.list_local().await.unwrap().models.is_empty());
        Arc::clone(&fixture.local).recheck_roots(true).await;
        assert_eq!(fixture.local.list_local().await.unwrap().models.len(), 1);
    }

    #[tokio::test]
    async fn a_watch_that_cannot_start_is_reported_as_unwatched() {
        let temporary = tempfile::tempdir().unwrap();
        let missing = temporary.path().join("missing");
        let watcher = RootWatcher::start(
            vec![crate::local_watch::WatchTarget {
                path: missing.clone(),
                recursive: true,
                interest: missing.clone(),
            }],
            |_| {},
        )
        .expect("the watcher starts");
        assert_eq!(watcher.unwatched(), &BTreeSet::from([missing]));
    }

    /// QA (Debian, round 3b): a model kept "Drafter None" with its drafter back in its folder,
    /// after the drafter was moved out and back and another folder was added at once. Adding the
    /// folder replaced the watcher while the move back was still settling, and the burst was
    /// dropped. A watch that stops reports what it had received at once.
    #[tokio::test(start_paused = true)]
    async fn a_watch_stopped_while_a_change_settles_still_reports_it() {
        let (events, received) = tokio::sync::mpsc::unbounded_channel();
        let (reported, mut reports) = tokio::sync::mpsc::unbounded_channel();
        let debounce = tokio::spawn(crate::local_watch::debounced(received, move |affected| {
            let _ = reported.send(affected);
        }));
        let folder = PathBuf::from("/models/A");
        events.send(BTreeSet::from([folder.clone()])).unwrap();
        tokio::time::sleep(crate::local_watch::WATCH_DEBOUNCE / 3).await;
        assert!(reports.try_recv().is_err(), "still settling");

        let stopped = tokio::time::Instant::now();
        drop(events);
        assert_eq!(reports.recv().await, Some(BTreeSet::from([folder])));
        assert_eq!(stopped.elapsed(), Duration::ZERO, "reported as the watch stops");
        debounce.await.unwrap();
    }

    fn imported(file: &Path) -> CustomModelDeclaration {
        CustomModelDeclaration {
            model_id: local_id(file),
            origin: CustomModelDeclarationOrigin::Imported {
                path: file.to_path_buf(),
            },
            target: vec![file.to_string_lossy().into_owned()],
            projector: None,
        }
    }

    /// The separate drafter file a model serves with, if any.
    fn served_drafter(fixture: &Fixture, model_id: &ModelId) -> Option<PathBuf> {
        match fixture
            .resolver
            .serving_configuration(model_id)
            .unwrap()
            .bundle
        {
            ServableModelBundle::SpeculativeDecoding {
                draft_source: SpeculativeDraftSource::Separate { draft },
                ..
            } => Some(draft.files[0].path.clone()),
            _ => None,
        }
    }

    fn drafter_state(
        listed: &LocalModelsResponse,
        model_id: &ModelId,
    ) -> Option<AttachedDrafterState> {
        listed
            .models
            .iter()
            .find(|model| &model.model_id == model_id)
            .and_then(|model| model.drafter.as_ref())
            .map(|drafter| drafter.state.clone())
    }

    /// A target and the drafter file beside it, as the real pairs are shaped: a DSpark/DFlash
    /// drafter (LFM2.5 with magnitudedev's DSpark), Gemma 4's `mtp-*` assistant file, and an MTP
    /// file of prediction-head tensors only.
    fn write_pair(target: &Path, drafter: &Path, shape: &str) -> SpeculativeMethod {
        let arch = |name| ("general.architecture", Meta::Str(name));
        match shape {
            "dspark" => {
                write_model(target, "LFM2.5 2.6B");
                write_dflash_drafter(drafter, "LFM2.5 2.6B DSpark");
                SpeculativeMethod::DSpark
            }
            "gemma-mtp" => {
                crate::test_gguf::write_gguf(
                    target,
                    &[
                        arch("gemma4"),
                        ("general.name", Meta::Str("gemma-4-E2B-it")),
                    ],
                    &["token_embd.weight", "blk.0.attn_q.weight"],
                );
                crate::test_gguf::write_gguf(
                    drafter,
                    &[
                        arch("gemma4-assistant"),
                        ("general.name", Meta::Str("mtp-gemma-4-E2B-it")),
                    ],
                    &["blk.0.attn_q.weight"],
                );
                SpeculativeMethod::Mtp
            }
            "nextn-mtp" => {
                crate::test_gguf::write_gguf(
                    target,
                    &[arch("qwen35"), ("general.name", Meta::Str("Qwen3.5"))],
                    &["token_embd.weight", "blk.0.attn_q.weight"],
                );
                crate::test_gguf::write_gguf(
                    drafter,
                    &[arch("qwen35"), ("general.name", Meta::Str("Qwen3.5 MTP"))],
                    &["blk.3.nextn.eh_proj.weight"],
                );
                SpeculativeMethod::Mtp
            }
            _ => unreachable!("{shape}"),
        }
    }

    #[tokio::test]
    async fn an_imported_models_automatic_drafter_beside_it_is_attached_in_place_outside_every_location()
     {
        for shape in ["dspark", "gemma-mtp", "nextn-mtp"] {
            let fixture = Fixture::new().await;
            let folder = fixture.root.join("Downloads/imported");
            let target = folder.join("target-Q4_K_M.gguf");
            let drafter = folder.join("drafter-beside.gguf");
            let method = write_pair(&target, &drafter, shape);
            write_model(&folder.join("unrelated-model.gguf"), "unrelated");
            let declaration = imported(&target);
            let id = declaration.model_id.clone();
            fixture
                .resolver
                .set_custom_model_enrichment(Arc::new(Recommends {
                    drafter: drafter.clone(),
                    variant_of: "lfm2.5-2.6b:gguf:q4".parse().unwrap(),
                    method: method.clone(),
                }));

            // Auto: no drafter choice is stored, as a pick of the recommended drafter saves.
            let listed = fixture
                .apply(LocalModelsConfiguration {
                    declarations: vec![declaration],
                    ..LocalModelsConfiguration::default()
                })
                .await;

            assert_eq!(
                ids(&listed.models),
                [id.to_string()],
                "{shape}: only the imported file is a model; its folder's other models are not picked up"
            );
            assert_eq!(
                drafter_state(&listed, &id),
                Some(AttachedDrafterState::Ready),
                "{shape}"
            );
            assert_eq!(
                served_drafter(&fixture, &id).as_deref(),
                Some(Path::new("drafter-beside.gguf")),
                "{shape}: the automatic drafter is served from where it is"
            );
            assert!(matches!(
                fixture.resolver.serving_configuration(&id).unwrap().bundle,
                ServableModelBundle::SpeculativeDecoding { method: served, .. } if served == method
            ));
            assert!(
                fixture
                    .local
                    .watched()
                    .iter()
                    .any(|watched| watched.path == folder),
                "{shape}: the folder holding the model and its drafter is watched"
            );

            fs::remove_file(&drafter).unwrap();
            let gone = fixture.local.refresh_local().await.unwrap();
            assert_eq!(
                drafter_state(&gone, &id),
                Some(AttachedDrafterState::NotInstalled),
                "{shape}"
            );
            assert_eq!(
                served_drafter(&fixture, &id),
                None,
                "{shape}: a model whose drafter disappeared serves plain"
            );
            assert!(matches!(
                gone.models[0].state,
                CustomModelState::Ready { .. }
            ));

            write_pair(&target, &drafter, shape);
            let back = fixture.local.refresh_local().await.unwrap();
            assert_eq!(
                drafter_state(&back, &id),
                Some(AttachedDrafterState::Ready),
                "{shape}: the drafter is found again when it comes back"
            );
            assert!(served_drafter(&fixture, &id).is_some(), "{shape}");
        }
    }

    #[tokio::test]
    async fn an_imported_models_explicit_local_drafter_elsewhere_is_attached_and_watched() {
        let fixture = Fixture::new().await;
        let target = fixture.root.join("Downloads/lfm-Q4_K_M.gguf");
        let elsewhere = fixture.root.join("Elsewhere");
        let drafter = elsewhere.join("other-drafter.gguf");
        write_model(&target, "target");
        write_dflash_drafter(&drafter, "drafter");
        write_dflash_drafter(&elsewhere.join("loose-drafter.gguf"), "loose");
        let declaration = imported(&target);
        let id = declaration.model_id.clone();
        let configuration = LocalModelsConfiguration {
            declarations: vec![declaration],
            drafters: vec![DrafterAttachment {
                model_id: id.clone(),
                drafter: DrafterChoice::Reference {
                    reference: DrafterReference::Local {
                        path: drafter.clone(),
                    },
                    method: SpeculativeMethod::DFlash,
                },
            }],
            ..LocalModelsConfiguration::default()
        };

        let listed = fixture.apply(configuration).await;

        assert_eq!(
            drafter_state(&listed, &id),
            Some(AttachedDrafterState::Ready)
        );
        assert_eq!(
            served_drafter(&fixture, &id).as_deref(),
            Some(Path::new("other-drafter.gguf"))
        );
        assert!(
            fixture
                .local
                .watched()
                .iter()
                .any(|watched| watched.path == elsewhere),
            "the chosen drafter's folder is watched"
        );

        fs::remove_file(&drafter).unwrap();
        let gone = fixture.local.refresh_local().await.unwrap();
        assert_eq!(
            drafter_state(&gone, &id),
            Some(AttachedDrafterState::NotInstalled)
        );
        assert_eq!(served_drafter(&fixture, &id), None);
        assert!(matches!(
            gone.models[0].state,
            CustomModelState::Ready { .. }
        ));
    }

    /// Lists the automatic drafters in order: a DFlash file, then built-in heads.
    struct InOrder(PathBuf);

    impl CustomModelEnrichment for InOrder {
        fn enrich(&self, _model: &CustomModelFacts<'_>) -> CustomModelEnrichmentResult {
            let candidate = |reference, method: SpeculativeMethod| DrafterCandidate {
                category: DrafterCategory::NextToModel,
                reference,
                name: "drafter".to_owned(),
                label: "In this folder".to_owned(),
                size_bytes: 1,
                lineage: DrafterLineage::Lineage,
                verdict: DrafterVerdict::Bound {
                    method: method.clone(),
                    memory_bytes: 1,
                },
                automatic: true,
                on_this_computer: true,
                method: Some(method),
            };
            CustomModelEnrichmentResult {
                variant_of: None,
                drafters: vec![
                    candidate(
                        DrafterReference::Local {
                            path: self.0.clone(),
                        },
                        SpeculativeMethod::DFlash,
                    ),
                    candidate(DrafterReference::Embedded, SpeculativeMethod::Mtp),
                ],
                catalog_drafters: Vec::new(),
                license: None,
            }
        }
    }

    #[tokio::test]
    async fn an_automatic_drafter_that_goes_missing_gives_way_to_the_next_one_on_this_computer() {
        let fixture = Fixture::new().await;
        let folder = fixture.root.join("models");
        let drafter = folder.join("drafter.gguf");
        write_mtp_model(&folder.join("with-heads.gguf"), "heads");
        write_dflash_drafter(&drafter, "drafter");
        let id = local_id(&folder.join("with-heads.gguf"));
        fixture
            .resolver
            .set_custom_model_enrichment(Arc::new(InOrder(drafter.clone())));
        let listed = fixture
            .apply(roots(vec![root(ModelRootKind::Folder, &folder)]))
            .await;
        assert_eq!(
            drafter_state(&listed, &id),
            Some(AttachedDrafterState::Ready)
        );
        assert_eq!(
            served_drafter(&fixture, &id).as_deref(),
            Some(Path::new("drafter.gguf"))
        );

        fs::remove_file(&drafter).unwrap();
        let listed = fixture.local.refresh_local().await.unwrap();
        let model = listed
            .models
            .iter()
            .find(|model| model.model_id == id)
            .unwrap();
        assert_eq!(
            model.drafter.as_ref().map(|drafter| &drafter.reference),
            Some(&DrafterReference::Embedded),
            "the next automatic drafter, already here, is attached at once"
        );
        assert!(matches!(
            fixture.resolver.serving_configuration(&id).unwrap().bundle,
            ServableModelBundle::SpeculativeDecoding {
                draft_source: SpeculativeDraftSource::Embedded,
                method: SpeculativeMethod::Mtp,
                ..
            }
        ));
        assert_eq!(
            model
                .recommended_drafter
                .as_ref()
                .map(|candidate| &candidate.reference),
            Some(&DrafterReference::Local {
                path: drafter.clone()
            }),
            "the recommendation is the enumeration's until it is recomputed"
        );
    }

    #[test]
    fn watch_targets_cover_enabled_roots_and_declared_files() {
        let temporary = tempfile::tempdir().unwrap();
        let present = temporary.path().join("present");
        fs::create_dir_all(&present).unwrap();
        let drafters = temporary.path().join("drafters");
        fs::create_dir_all(&drafters).unwrap();
        let configuration = LocalModelsConfiguration {
            roots: vec![
                root(ModelRootKind::Folder, &present),
                ModelRoot {
                    enabled: false,
                    ..root(ModelRootKind::Folder, &temporary.path().join("disabled"))
                },
                root(
                    ModelRootKind::Ollama,
                    &temporary.path().join("missing/models"),
                ),
            ],
            declarations: vec![CustomModelDeclaration {
                model_id: format!("local:{}", "a".repeat(64)).parse().unwrap(),
                origin: CustomModelDeclarationOrigin::Imported {
                    path: present.join("file.gguf"),
                },
                target: vec![present.join("file.gguf").to_string_lossy().into_owned()],
                projector: None,
            }],
            drafters: vec![DrafterAttachment {
                model_id: format!("local:{}", "a".repeat(64)).parse().unwrap(),
                drafter: DrafterChoice::Reference {
                    reference: DrafterReference::Local {
                        path: drafters.join("drafter.gguf"),
                    },
                    method: SpeculativeMethod::DFlash,
                },
            }],
            ..LocalModelsConfiguration::default()
        };
        let targets = crate::local_watch::watch_targets(&configuration);
        let watched = targets
            .iter()
            .map(|target| (target.path.clone(), target.recursive))
            .collect::<BTreeSet<_>>();
        assert_eq!(
            watched,
            BTreeSet::from([
                (present.clone(), true),
                (present.clone(), false),
                (drafters.clone(), false),
                (temporary.path().to_path_buf(), false),
            ])
        );
    }
}
