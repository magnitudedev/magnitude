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
    AttachedDrafter, AttachedDrafterState, CustomModel, CustomModelCatalogAttribution,
    CustomModelDeclaration, CustomModelDeclarationOrigin, CustomModelFile, CustomModelOrigin,
    CustomModelState, DrafterCandidate, DrafterChoice, DrafterPruning, DrafterReference, EffectiveModel,
    FoundSource, HuggingFaceArtifactSelector, HuggingFaceRepositoryId, InstalledCatalogAttribution,
    InstalledModelPackage, LocalModelDigest, LocalModelRemoval, LocalModels,
    LocalModelsConfiguration, LocalModelsResponse, ModelDomainInvalidation, ModelFailure,
    ModelFileRole, ModelId, ModelPackage, ModelPackageId, ModelPackageInstallationOrigin,
    ModelServingConfiguration, PackageValidation, RecommendableModelCatalog, ServableModelBundle,
    ServingProfile, SpeculativeDraftSource, SpeculativeMethod, UnsupportedLocalModel,
    UnsupportedReason,
};
use magnitude_service_contracts::{
    ExternalOrigin, HubFile, InventoryError, ModelLocation, ModelSource,
};

use crate::catalog_models::catalog_id;
use crate::gguf::{GgufBaseModel, GgufRole};
use crate::inventory::{
    InstalledPackageRecord, InstalledPackageSnapshot, ManagedModelStore, hf_repo_dir,
    origin_precedence,
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
    pub embedded_mtp: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CustomModelEnrichmentResult {
    pub variant_of: Option<ModelId>,
    pub recommended_drafter: Option<DrafterCandidate>,
    pub license: Option<String>,
}

/// Lineage and drafter recommendation for custom models. It is called while the local-models
/// list is projected, so it answers from what it already knows and does no I/O; when its
/// knowledge changes it calls [`ModelDomainResolver::custom_model_enrichment_changed`]. A model
/// whose drafter choice is `Auto` uses `recommended_drafter`.
pub trait CustomModelEnrichment: Send + Sync + 'static {
    fn enrich(&self, model: &CustomModelFacts<'_>) -> CustomModelEnrichmentResult;
    /// Whether `enrich` answers from knowledge rather than from not having computed it yet.
    fn knows(&self, _model: &CustomModelFacts<'_>) -> bool {
        true
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
    changes: tokio::sync::broadcast::Sender<ModelDomainInvalidation>,
}

impl LocalModelsContext {
    pub(crate) fn new() -> Self {
        let (changes, _) = tokio::sync::broadcast::channel(32);
        Self {
            enrichment: RwLock::new(Arc::new(NoEnrichment)),
            generation: AtomicU64::new(0),
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

    /// Republishes the local-models list after the enrichment's knowledge changed.
    pub fn custom_model_enrichment_changed(&self) {
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
        let mut projection = project(
            &records,
            &configuration,
            &self.catalog,
            enrichment.as_ref(),
            &self.inventory.config.root,
        );
        projection.revision = installed.revision + self.local.generation.load(Ordering::Acquire);
        projection.reconciliation_complete = installed.reconciliation_complete;
        Ok(projection)
    }

    /// The Magnitude-owned drafter packages no model uses: not the drafter of an installed
    /// catalog model, and not the drafter any custom model serves with.
    fn unused_drafter_packages(&self) -> Result<Vec<ModelPackageId>, InventoryError> {
        let projection = self.local_projection()?;
        let installed = self.inventory.installed_packages_response()?;
        let present = installed
            .packages
            .iter()
            .map(|entry| &entry.package.id)
            .collect::<BTreeSet<_>>();
        let separate_draft = |bundle: &ServableModelBundle| match bundle {
            ServableModelBundle::SpeculativeDecoding {
                draft_source: SpeculativeDraftSource::Separate { draft },
                ..
            } => Some(draft.id.clone()),
            _ => None,
        };
        // A model whose automatic drafter isn't worked out yet may be about to use any catalog
        // drafter, so none is pruned until it is.
        let drafter_pending = projection
            .models
            .iter()
            .chain(&projection.hidden)
            .any(|projected| projected.drafter_pending);
        let mut used = self
            .catalog
            .models
            .iter()
            .filter(|definition| {
                drafter_pending
                    || present.contains(&crate::lineage::catalog_target(definition).id)
            })
            .filter_map(|definition| separate_draft(&definition.configuration.bundle))
            .collect::<BTreeSet<_>>();
        used.extend(
            projection
                .models
                .iter()
                .chain(&projection.hidden)
                .filter_map(|projected| projected.configuration.as_ref().ok())
                .filter_map(|configuration| separate_draft(&configuration.bundle)),
        );
        let records = self.inventory.installed_packages.read().map_err(|_| {
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
                    .is_some_and(|evidence| matches!(evidence.role, GgufRole::Drafter { .. }))
            })
            .map(|record| record.installed.package.id.clone())
            .filter(|package_id| !used.contains(package_id))
            .collect::<BTreeSet<_>>();
        Ok(unused.into_iter().collect())
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
    configuration: Result<ModelServingConfiguration, ModelFailure>,
    managed_packages: Vec<ModelPackageId>,
    drafter_pending: bool,
}

#[derive(Clone)]
struct HubIdentity {
    file: HubFile,
    commit: Option<String>,
    current: bool,
}

/// One observed occurrence of a package, with what identity needs from it.
struct Occurrence<'a> {
    record: &'a InstalledPackageRecord,
    evidence: Option<&'a PrimaryEvidence>,
    hub: Option<HubIdentity>,
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

    fn content_key(&self) -> Option<(String, u64)> {
        self.evidence
            .map(|evidence| (evidence.header_digest.clone(), self.primary_size))
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
    let (hub, ollama_name) = match (&record.model.location, &record.model.source) {
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
        ),
        (ModelLocation::External { origin, .. }, _) => {
            let (hub, name) = match origin {
                ExternalOrigin::LmStudio { hub } | ExternalOrigin::LlamaCpp { hub } => {
                    (hub.as_ref(), None)
                }
                ExternalOrigin::Ollama { name, hub } => (hub.as_ref(), Some(name.as_str())),
                ExternalOrigin::Folder | ExternalOrigin::Imported => (None, None),
            };
            (
                hub.map(|file| HubIdentity {
                    file: file.clone(),
                    commit: None,
                    current: false,
                }),
                name,
            )
        }
        _ => (None, None),
    };
    Some(Occurrence {
        record,
        evidence: record.primary.as_ref(),
        hub,
        ollama_name,
        primary: base.join(&primary_file.path),
        primary_size: primary_file.size_bytes,
        base,
        precedence: origin_precedence(record.installed.origin),
    })
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
                if matches!(evidence.role, GgufRole::Model { .. })
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

    // One model per identity: the declared one, else the best-represented content.
    let mut identified =
        BTreeMap::<ModelId, (Option<&CustomModelDeclaration>, ContentGroup)>::new();
    for ((digest, _), mut members) in groups {
        members.sort_by_key(|occurrence| occurrence_rank(occurrence));
        let group = ContentGroup {
            digest,
            occurrences: members,
        };
        let declaration = configuration
            .declarations
            .iter()
            .find(|declaration| group.matches(declaration));
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
        reasons: vec![UnsupportedReason::Unreadable],
        detail,
    })
}

/// What every model's projection consults: the configuration, the catalog, the enrichment, every
/// occurrence (drafters resolve among them) and the installed packages by identity.
struct ModelContext<'a> {
    configuration: &'a LocalModelsConfiguration,
    catalog: &'a RecommendableModelCatalog,
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
            &hub.map_or_else(|| representative.primary.clone(), |hub| PathBuf::from(&hub.selector)),
            hub.map(|hub| hub.repository.as_str()),
            evidence.name.as_deref(),
            evidence.quantization.as_deref(),
        );
        let display_name = match (representative.ollama_name, hub) {
            (Some(name), None) => name.to_owned(),
            _ => presented_name,
        };
        let embedded_mtp = matches!(evidence.role, GgufRole::Model { embedded_mtp: true });
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
            embedded_mtp,
        };
        let enriched = enrichment.enrich(&facts);
        let enrichment_known = enrichment.knows(&facts);

        let choice = configuration
            .drafters
            .iter()
            .find(|attachment| attachment.model_id == model_id)
            .map_or(DrafterChoice::Auto, |attachment| attachment.drafter.clone());
        let drafter = match &choice {
            DrafterChoice::None => None,
            DrafterChoice::Auto => enriched.recommended_drafter.as_ref().and_then(|candidate| {
                Some((candidate.reference.clone(), candidate.method.clone()?))
            }),
            DrafterChoice::Reference { reference, method } => {
                Some((reference.clone(), method.clone()))
            }
        }
        .map(|(reference, method)| {
            resolve_drafter(reference, method, embedded_mtp, catalog, occurrences)
        });

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
        let installation = resolved_installation(installed);
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
            drafter_pending: matches!(choice, DrafterChoice::Auto) && !enrichment_known,
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
                variant_of: enriched.variant_of,
                files,
                size_bytes,
                projector,
                drafter: drafter.map(|drafter| drafter.attached),
                recommended_drafter: enriched.recommended_drafter,
                license: enriched.license.or_else(|| evidence.license.clone()),
                profile,
                state,
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
        let drafter = self
            .configuration
            .drafters
            .iter()
            .find(|attachment| attachment.model_id == declaration.model_id)
            .and_then(|attachment| match &attachment.drafter {
                DrafterChoice::Reference { reference, method } => Some(
                    resolve_drafter(
                        reference.clone(),
                        method.clone(),
                        false,
                        self.catalog,
                        self.occurrences,
                    )
                    .attached,
                ),
                DrafterChoice::Auto | DrafterChoice::None => None,
            });
        let (display_name, variant_label) = crate::candidate_files::model_presentation(
            Path::new(primary),
            Some(repository.as_str()),
            None,
            None,
        );
        Some(ProjectedModel {
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
            },
            configuration: Err(ModelFailure {
                code: "model_not_installed".to_owned(),
                message: format!("{} has not finished downloading", declaration.model_id),
                retryable: true,
            }),
            managed_packages: Vec::new(),
            drafter_pending: false,
        })
    }
}

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
                reasons: vec![UnsupportedReason::TargetMismatch {
                    detail: detail.clone(),
                }],
                detail,
            },
        },
        source: None,
    }
}

/// Resolves a drafter reference against the files on this computer. A drafter that is not here
/// yet leaves the model serving without one until it is.
fn resolve_drafter(
    reference: DrafterReference,
    method: SpeculativeMethod,
    embedded_mtp: bool,
    catalog: &RecommendableModelCatalog,
    occurrences: &[Occurrence<'_>],
) -> ResolvedDrafter {
    let separate = |reference: DrafterReference,
                    method: SpeculativeMethod,
                    found: Option<&Occurrence<'_>>| match found {
        None => ResolvedDrafter {
            attached: AttachedDrafter {
                reference,
                method,
                state: AttachedDrafterState::NotInstalled,
            },
            source: None,
        },
        Some(occurrence) => match occurrence.usable() {
            Ok(()) => ResolvedDrafter {
                attached: AttachedDrafter {
                    reference,
                    method,
                    state: AttachedDrafterState::Ready,
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
                        reasons: vec![UnsupportedReason::Unreadable],
                        detail: failure.message,
                    },
                },
                source: None,
            },
        },
    };
    let embedded = |reference: DrafterReference, method: SpeculativeMethod| {
        if embedded_mtp {
            ResolvedDrafter {
                attached: AttachedDrafter {
                    reference,
                    method,
                    state: AttachedDrafterState::Ready,
                },
                source: Some(SpeculativeDraftSource::Embedded),
            }
        } else {
            rejected(
                reference,
                method,
                "the model has no built-in prediction heads".to_owned(),
            )
        }
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
                    let found = occurrences
                        .iter()
                        .filter(|occurrence| occurrence.record.installed.package.id == draft.id)
                        .min_by_key(|occurrence| occurrence.precedence);
                    separate(reference, method, found)
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
            let found = occurrences
                .iter()
                .filter(|occurrence| {
                    occurrence.hub.as_ref().is_some_and(|identity| {
                        identity.file.repository == repository.as_str()
                            && identity.file.selector == *path
                            && identity.commit.as_deref() == Some(commit.as_str())
                    })
                })
                .min_by_key(|occurrence| occurrence.precedence);
            separate(reference, method, found)
        }
        DrafterReference::Local { path } => {
            let found = occurrences
                .iter()
                .filter(|occurrence| same_file(&occurrence.primary, path))
                .min_by_key(|occurrence| occurrence.precedence);
            separate(reference, method, found)
        }
    }
}

pub struct ManagedLocalModels {
    resolver: Arc<ModelDomainResolver>,
    watcher: Mutex<Option<RootWatcher>>,
    this: Weak<Self>,
}

impl ManagedLocalModels {
    pub(crate) fn new(resolver: Arc<ModelDomainResolver>) -> Arc<Self> {
        let models = Arc::new_cyclic(|this| Self {
            resolver,
            watcher: Mutex::new(None),
            this: this.clone(),
        });
        models.rewatch();
        models
    }

    pub(crate) fn installed_packages_changed(&self, _revision: u64) {
        self.resolver.publish_local_change();
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
        if watcher
            .as_ref()
            .is_some_and(|watcher| watcher.targets() == targets.as_slice())
        {
            return;
        }
        let this = self.this.clone();
        *watcher = RootWatcher::start(targets, move |affected| {
            if let Some(models) = this.upgrade() {
                tokio::spawn(async move { models.rescan_after_change(affected).await });
            }
        });
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
                    .generation
                    .fetch_add(1, Ordering::AcqRel);
                self.resolver.publish_local_change();
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
            use magnitude_service_contracts::models::InstalledModelPackages as _;
            self.inventory().ensure_installed_model_inventory().await?;
            let unused = self.resolver.unused_drafter_packages()?;
            let mut removed = Vec::new();
            let mut freed_bytes = 0_u64;
            for package_id in unused {
                let removal = self.inventory().remove_installed(&package_id).await?;
                if removal.removed {
                    freed_bytes = freed_bytes.saturating_add(removal.freed_bytes);
                    removed.push(package_id);
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
        ModelRoot, ModelRootKind,
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
            let temporary = tempfile::tempdir().unwrap();
            let root = temporary.path().canonicalize().unwrap();
            let store = root.join("store");
            let mut config = InventoryConfig::with_roots(store.clone(), root.join("cache")).unwrap();
            config.hub_endpoint = Some(endpoint);
            let inventory = Arc::new(ManagedModelStore::open(config).await.unwrap());
            inventory.ensure_installed_model_inventory().await.unwrap();
            let resolver = ModelDomainResolver::new(
                inventory,
                RecommendableModelCatalog {
                    models: Vec::new(),
                    diagnostics: Vec::new(),
                },
            );
            let local = ManagedLocalModels::new(resolver.clone());
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
        ModelId::local(&LocalModelDigest::new(header_digest(path)).unwrap())
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
            .find(|model| matches!(
                &model.source,
                ModelSource::HuggingFace { repository, .. } if repository == "owner/Shared-GGUF"
            ))
            .expect("inventory publishes the shared-blob snapshot");
        assert_eq!(
            model.location.components()[0].content,
            ContentIdentity::Sha256 {
                value: sha256(&bytes)
            },
            "the repository blob's name, not the shared store's, identifies the content"
        );
        let resolved = inventory.resolve_ready(&model.id).await.unwrap();
        assert_eq!(resolved.components[0].path, snapshot.join("Shared-Q4_K_M.gguf"));
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
        assert_eq!(source("hf:bartowski/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"), llama_cpp);
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
            &[("application/vnd.ollama.image.model", &model_bytes("user only"))],
        );
        write_ollama(
            &system,
            "registry.ollama.ai",
            "library",
            "gemma3",
            "4b",
            &[("application/vnd.ollama.image.model", &model_bytes("system only"))],
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
                let found = request_line.contains(&format!("/api/models/{repository}/revision/main"));
                let (status, body) = if found { ("200 OK", body.as_slice()) } else { ("404 Not Found", &b""[..]) };
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
    async fn an_ollama_hub_pull_takes_the_id_of_the_same_file_in_the_hugging_face_cache() {
        let pulled = model_bytes("hub pull");
        let endpoint = serve_repository(
            "unsloth/Qwen3-8B-GGUF",
            vec![("Qwen3-8B-Q4_K_M.gguf", pulled.clone()), ("Qwen3-8B-Q8_0.gguf", model_bytes("other"))],
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
        let expected = "hf:unsloth/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf";
        let mut listed = fixture
            .apply(roots(vec![root(ModelRootKind::Ollama, &ollama)]))
            .await;
        for _ in 0..100 {
            if ids(&listed.models) == [expected] {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            listed = fixture.local.refresh_local().await.unwrap();
        }
        assert_eq!(ids(&listed.models), [expected], "Ollama alone resolves the real file name");

        let hub = fixture.root.join("hf/hub");
        publish_hf_cache(&hub, "unsloth/Qwen3-8B-GGUF", &"c".repeat(40), &[("Qwen3-8B-Q4_K_M.gguf", pulled)]);
        let both = fixture
            .apply(roots(vec![
                root(ModelRootKind::Ollama, &ollama),
                root(ModelRootKind::HuggingFaceCache, &hub),
            ]))
            .await;
        assert_eq!(ids(&both.models), [expected], "the same bytes in two locations are one model with one ID");
        let cache_only = fixture
            .apply(roots(vec![root(ModelRootKind::HuggingFaceCache, &hub)]))
            .await;
        assert_eq!(ids(&cache_only.models), [expected]);
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

    #[tokio::test]
    async fn identity_never_reads_tensor_data() {
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
        .expect("a 512 GiB file is identified from its header alone");

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
            [UnsupportedReason::Unreadable]
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
            [UnsupportedReason::Unreadable]
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
            origin: CustomModelDeclarationOrigin::Imported { path: folder.clone() },
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
        assert!(matches!(listed.models[0].state, CustomModelState::Ready { .. }));
        assert_eq!(listed.models[0].files[0].path, link);
        let inventory = fixture.inventory();
        let model = inventory
            .list()
            .await
            .unwrap()
            .into_iter()
            .find(|model| matches!(
                model.location,
                ModelLocation::External { origin: ExternalOrigin::Imported, .. }
            ))
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
        assert!(kept.removed.is_empty(), "a drafter a model serves with is kept");
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
        assert!(folder.join("target.gguf").exists(), "the model's own files are never touched");
    }

    #[tokio::test]
    async fn embedded_prediction_heads_attach_only_to_models_that_have_them() {
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

        assert!(matches!(
            fixture
                .resolver
                .serving_configuration(&with_heads)
                .unwrap()
                .bundle,
            ServableModelBundle::SpeculativeDecoding {
                draft_source: SpeculativeDraftSource::Embedded,
                ..
            }
        ));
        assert!(matches!(
            fixture
                .resolver
                .serving_configuration(&without_heads)
                .unwrap()
                .bundle,
            ServableModelBundle::Standalone { .. }
        ));
        let listed = fixture.local.list_local().await.unwrap();
        let rejected = listed
            .models
            .iter()
            .find(|model| model.model_id == without_heads)
            .and_then(|model| model.drafter.as_ref())
            .unwrap();
        assert!(matches!(
            rejected.state,
            AttachedDrafterState::Rejected { .. }
        ));
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
                fixture.resolver.serving_configuration(&with_heads).unwrap().bundle,
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
    }

    impl CustomModelEnrichment for Recommends {
        fn enrich(&self, _model: &CustomModelFacts<'_>) -> CustomModelEnrichmentResult {
            CustomModelEnrichmentResult {
                variant_of: Some(self.variant_of.clone()),
                recommended_drafter: Some(DrafterCandidate {
                    category: DrafterCategory::NextToModel,
                    reference: DrafterReference::Local {
                        path: self.drafter.clone(),
                    },
                    name: "drafter.gguf".to_owned(),
                    label: "In this folder".to_owned(),
                    size_bytes: 1,
                    lineage: DrafterLineage::Lineage,
                    verdict: DrafterVerdict::None,
                    on_this_computer: true,
                    method: Some(SpeculativeMethod::DFlash),
                }),
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
            Some((SpeculativeMethod::DFlash, AttachedDrafterState::NotInstalled))
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
        assert!(listed.models[0].recommended_drafter.is_some());
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

    #[test]
    fn watch_targets_cover_enabled_roots_and_declared_files() {
        let temporary = tempfile::tempdir().unwrap();
        let present = temporary.path().join("present");
        fs::create_dir_all(&present).unwrap();
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
                (temporary.path().to_path_buf(), false),
            ])
        );
    }
}
