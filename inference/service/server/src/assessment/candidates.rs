//! Candidates: files not yet added (a Hugging Face repository at a commit, or a local path), and
//! the drafters any model can use. Inspection groups a candidate's GGUFs into versions and asks
//! the engine for each version's verdict from its headers alone; drafter enumeration lists the
//! drafters offered for a target in automatic-choice order, each with the engine's binding
//! verdict. Assessments here are cached by the exact header material they read, like every
//! assessment.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Weak};

use futures_util::StreamExt;
use futures_util::future::BoxFuture;
use magnitude_artifacts::Error as ArtifactError;
use magnitude_artifacts::gguf::{FileHeader, HeaderSource};
use magnitude_engine::assessment::{
    ModelPackagePaths, PreparedModelAssessment, finish_model_assessment,
    prepare_model_assessment_from,
};
use magnitude_engine::fingerprint::ModelFingerprint as EngineModelFingerprint;
use magnitude_engine::options::ModelMethod;
use magnitude_service_contracts::models::{
    CandidateDrafter, CandidateFile, CandidateInspection, CandidateInspectionProgressSink,
    CandidateInspectionRequest, CandidateSource, CandidateVersion, CatalogModelState, CatalogModels, CustomModel,
    CustomModelOrigin, CustomModelState, DrafterCandidate, DrafterCandidatesRequest,
    DrafterCandidatesResponse, DrafterCategory, DrafterChoice, DrafterLineage, DrafterReference,
    DrafterVerdict, HuggingFaceRepositoryId, LocalModels, ModelAssessment, ModelAssessmentProfile,
    ModelAssessmentSubject, ModelCandidate, ModelCandidates, ModelFileRole, ModelFingerprint,
    ModelId, ModelPackage, ModelPackageSource, ParsedModelId, ServableModelBundle, ServingProfile,
    SpeculativeDraftSource, SpeculativeMethod,
};
use magnitude_service_contracts::{
    ComponentRole, ContentIdentity, HuggingFaceRepositoryFile, HuggingFaceRepositoryRequest,
    HuggingFaceRepositorySnapshot, InventoryError, ModelPreviewComponentRole,
    ModelPreviewComponentSource, ModelPreviewSource, ResolvedModel,
};
use magnitude_service_models::candidate_files::{
    CandidateFileRole, CandidateGguf, GgufGroup, RecognizedArchitectures, display_name,
    base_model_repositories, group_ggufs, header_inspection,
    hugging_face_repository_from_url, is_q4_k_class, local_directory, version_label,
};
use magnitude_service_models::lineage::{
    CatalogLineage, CatalogLineageMatch,
};
use magnitude_service_models::{
    CachedModelAssessment, CatalogDrafterFit, CatalogFingerprints, ModelShape, CustomModelEnrichment, CustomModelEnrichmentResult,
    CustomModelFacts, DrafterContent, ManagedModelStore, ModelDomainResolver,
    ServableModelBundleKey, refresh_hugging_face_repository,
};
use sha2::{Digest, Sha256};

use super::assessor::{ModelAssessor, model_assessment, performance_depths, preparation_error};
use super::inspection_progress::InspectionProgress;

/// Concurrent header requests while classifying a repository's files.
const HEADER_CONCURRENCY: usize = 12;
/// The most concurrent engine assessments of one request's versions or drafters; fewer on
/// machines with fewer cores.
const MAX_ASSESSMENT_CONCURRENCY: usize = 6;
const MAX_LOCAL_SCAN_DEPTH: usize = 2;
/// Drafters "in this folder" are the folder's own files, never a subfolder's.
const DRAFTER_SCAN_DEPTH: usize = 1;
const MAX_LOCAL_SCAN_FILES: usize = 512;
/// How long a custom model whose enrichment failed waits before it is tried again.
const ENRICHMENT_RETRY_DELAY: std::time::Duration = std::time::Duration::from_secs(300);

type Guard = Arc<dyn Send + Sync>;

/// A target's header material: its first GGUF (shards beside it), its projector, the identity
/// of exactly those bytes, and the header digest of each file whose digest is known, so the engine
/// reads that header from the header cache rather than the file.
struct TargetMaterial {
    target: PathBuf,
    projector: Option<PathBuf>,
    identity: String,
    headers: BTreeMap<PathBuf, String>,
    _guards: Vec<Guard>,
}

/// A separate drafter's header material.
struct DrafterMaterial {
    path: PathBuf,
    identity: String,
    headers: BTreeMap<PathBuf, String>,
    _guards: Vec<Guard>,
}

/// GGUF headers for the engine from this service's caches: a header whose digest the material
/// names comes from the content-addressed header cache, and any other file's through its identity
/// cache, so assessing a file again never rereads it.
struct StoredHeaders {
    store: Arc<ManagedModelStore>,
    known: BTreeMap<PathBuf, String>,
}

impl StoredHeaders {
    fn bytes(&self, path: &Path) -> Result<Vec<u8>, ArtifactError> {
        if let Some(bytes) = self
            .known
            .get(path)
            .and_then(|digest| self.store.cached_header(digest))
        {
            return Ok(bytes);
        }
        self.store
            .local_gguf_header(path)
            .map(|(_, header)| header)
            .map_err(|error| match error {
                InventoryError::Integrity(detail) => ArtifactError::Invalid(detail),
                other => ArtifactError::Io(std::io::Error::other(other.to_string())),
            })
    }
}

impl HeaderSource for StoredHeaders {
    fn header(&self, path: &Path) -> Result<FileHeader, ArtifactError> {
        let size = std::fs::metadata(path)?.len();
        Ok(FileHeader {
            directory: magnitude_artifacts::gguf::read_header(&self.bytes(path)?)?,
            size,
        })
    }
}

/// Everything drafter enumeration knows about one target.
struct DrafterTarget {
    material: TargetMaterial,
    repository: Option<(String, String)>,
    primary: Option<PathBuf>,
    folders: Vec<DrafterFolder>,
    snapshot: Option<HuggingFaceRepositorySnapshot>,
    lineage: BTreeSet<String>,
    catalog: Option<ModelId>,
    reach: Reach,
}

/// The folder whose own files are offered as drafters "in this folder". A model from a known Hub
/// repository whose snapshot can't be read (offline) is offered only that repository's files a
/// flat cache root names with its prefix (llama.cpp's `<owner>_<repository>_<file>`).
#[derive(PartialEq, Eq)]
struct DrafterFolder {
    path: PathBuf,
    prefix: Option<String>,
}

impl DrafterFolder {
    fn whole(path: PathBuf) -> Self {
        Self { path, prefix: None }
    }

    fn admits(&self, relative: &Path) -> bool {
        self.prefix.as_deref().is_none_or(|prefix| {
            relative
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with(prefix))
        })
    }
}

/// One classified complete GGUF of a candidate. A file whose header is invalid stays a target
/// version with `unreadable` naming why, so the rest of its repository is still inspected.
struct ClassifiedGroup {
    group: GgufGroup,
    role: CandidateFileRole,
    header_digest: String,
    local_digest: Option<String>,
    quantization: Option<String>,
    embedded_mtp: bool,
    unreadable: Option<String>,
}

/// What the release headers say about each catalog model, for matching models without lineage.
#[derive(Default)]
struct CatalogFacts {
    fingerprints: Vec<(ModelId, ModelFingerprint)>,
}

/// What an enrichment reads: only what is on this computer (the files here, the release catalog,
/// and Hub answers already cached), or the Hugging Face Hub as well (repository snapshots, Hub
/// lineage, drafter headers). What the files here say never waits for the Hub.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Reach {
    Local,
    Hub,
}

/// Computed lineage hints and automatic drafters, keyed by model and exact target package, each
/// with the custom-model inputs revision it was computed at and what it read, and the bytes of
/// every drafter reference whose header has been read, keyed by reference.
#[derive(Default)]
struct CandidateEnrichment {
    results:
        std::sync::RwLock<BTreeMap<(ModelId, String), (u64, Reach, CustomModelEnrichmentResult)>>,
    drafter_contents: std::sync::RwLock<BTreeMap<String, DrafterContent>>,
    catalog_drafter_fits: std::sync::RwLock<BTreeMap<ModelId, CatalogDrafterFit>>,
}

impl CandidateEnrichment {
    /// Whether the Hub-reaching result for `key` is known: the model's automatic drafter is
    /// worked out, not only from what is on this computer.
    fn knows_key(&self, key: &(ModelId, String)) -> bool {
        self.results.read().is_ok_and(|results| {
            results
                .get(key)
                .is_some_and(|(_, reach, _)| *reach == Reach::Hub)
        })
    }

    /// Whether any result of `reach` is known for the model, for whichever target.
    fn knows_model(&self, model_id: &ModelId, reach: Reach) -> bool {
        self.results.read().is_ok_and(|results| {
            results
                .iter()
                .any(|((known, _), (_, known_reach, _))| known == model_id && *known_reach == reach)
        })
    }

    /// Whether `key` needs no computing with `reach` at `revision`: its result of that reach was
    /// computed from the inputs at `revision`, or, for a local result, the model already has a
    /// Hub-reaching one, which keeps answering until it is recomputed.
    fn is_current(&self, key: &(ModelId, String), revision: u64, reach: Reach) -> bool {
        (reach == Reach::Local && self.knows_model(&key.0, Reach::Hub))
            || self.results.read().is_ok_and(|results| {
                results
                    .get(key)
                    .is_some_and(|(computed_at, known_reach, _)| {
                        *computed_at == revision && *known_reach == reach
                    })
            })
    }

    /// Record a result computed at `revision` with `reach`; whether it changed what the
    /// projection sees. A local result never replaces a Hub-reaching one: the Hub only adds to
    /// what the files here say.
    fn record(
        &self,
        key: (ModelId, String),
        revision: u64,
        reach: Reach,
        result: CustomModelEnrichmentResult,
    ) -> bool {
        let Ok(mut results) = self.results.write() else {
            return false;
        };
        let previous = results
            .iter()
            .find(|((model_id, _), _)| model_id == &key.0)
            .map(|(_, (_, reach, previous))| (*reach, previous.clone()));
        if reach == Reach::Local
            && previous
                .as_ref()
                .is_some_and(|(previous, _)| *previous == Reach::Hub)
        {
            return false;
        }
        results.retain(|(model_id, _), _| model_id != &key.0);
        results.insert(key, (revision, reach, result.clone()));
        previous.map(|(_, previous)| previous).unwrap_or_default() != result
    }

    fn knows_catalog_drafter_fit(&self, model_id: &ModelId) -> bool {
        self.catalog_drafter_fits
            .read()
            .is_ok_and(|fits| fits.contains_key(model_id))
    }

    /// Record how a catalog model serves without its drafter; whether that changed what it serves.
    fn record_catalog_drafter_fit(&self, model_id: ModelId, fit: CatalogDrafterFit) -> bool {
        let Ok(mut known) = self.catalog_drafter_fits.write() else {
            return false;
        };
        known.insert(model_id, fit).unwrap_or_default() != fit
    }

    /// Record a drafter reference's bytes; whether they are new.
    fn record_drafter_content(
        &self,
        reference: &DrafterReference,
        content: DrafterContent,
    ) -> bool {
        let Ok(mut contents) = self.drafter_contents.write() else {
            return false;
        };
        contents.insert(reference_key(reference), content.clone()) != Some(content)
    }
}

impl CustomModelEnrichment for CandidateEnrichment {
    /// The result for the model's exact target, or, while that is being computed (its target
    /// package changed: a projector appeared beside it, say), the model's previous result. The
    /// projection checks every automatic candidate against the files present, so a drafter that
    /// is attached and serving is never shown as none while a recompute is pending.
    fn enrich(&self, model: &CustomModelFacts<'_>) -> CustomModelEnrichmentResult {
        self.results
            .read()
            .ok()
            .and_then(|results| {
                results
                    .get(&(model.model_id.clone(), model.target.id.0.clone()))
                    .or_else(|| {
                        results
                            .iter()
                            .find(|((model_id, _), _)| model_id == model.model_id)
                            .map(|(_, result)| result)
                    })
                    .map(|(_, _, result)| result.clone())
            })
            .unwrap_or_default()
    }

    fn knows(&self, model: &CustomModelFacts<'_>) -> bool {
        self.knows_key(&(model.model_id.clone(), model.target.id.0.clone()))
    }

    fn drafter_content(&self, reference: &DrafterReference) -> Option<DrafterContent> {
        self.drafter_contents
            .read()
            .ok()
            .and_then(|contents| contents.get(&reference_key(reference)).cloned())
    }

    fn catalog_drafter_fit(&self, model_id: &ModelId) -> CatalogDrafterFit {
        self.catalog_drafter_fits
            .read()
            .ok()
            .and_then(|fits| fits.get(model_id).copied())
            .unwrap_or_default()
    }
}

pub struct ManagedModelCandidates {
    assessor: Arc<ModelAssessor>,
    lineage: Arc<CatalogLineage>,
    architectures: RecognizedArchitectures,
    catalog: Arc<dyn CatalogModels>,
    local: Arc<dyn LocalModels>,
    catalog_facts: tokio::sync::OnceCell<CatalogFacts>,
    gates: tokio::sync::Mutex<BTreeMap<String, Weak<tokio::sync::Mutex<()>>>>,
    enrichment: Arc<CandidateEnrichment>,
    resolver: std::sync::OnceLock<Arc<ModelDomainResolver>>,
}

/// The engine's recognized architectures, which search and file classification use.
#[must_use]
pub fn engine_architectures() -> RecognizedArchitectures {
    let architectures = magnitude_engine::families::architectures();
    RecognizedArchitectures::new(architectures.targets, architectures.drafters)
}

impl ManagedModelCandidates {
    pub fn new(
        assessor: Arc<ModelAssessor>,
        lineage: Arc<CatalogLineage>,
        architectures: RecognizedArchitectures,
        catalog: Arc<dyn CatalogModels>,
        local: Arc<dyn LocalModels>,
    ) -> Arc<Self> {
        Arc::new(Self {
            assessor,
            lineage,
            architectures,
            catalog,
            local,
            catalog_facts: tokio::sync::OnceCell::new(),
            gates: tokio::sync::Mutex::new(BTreeMap::new()),
            enrichment: Arc::new(CandidateEnrichment::default()),
            resolver: std::sync::OnceLock::new(),
        })
    }

    fn models(&self) -> &Arc<ManagedModelStore> {
        self.assessor.models()
    }

    /// The drafter a model is recommended: the first candidate the engine binds.
    pub async fn recommended_drafter(
        &self,
        model_id: &ModelId,
    ) -> Result<Option<DrafterCandidate>, InventoryError> {
        let target = self.model_target(model_id, Reach::Hub).await?;
        Ok(recommended(
            self.enumerate_drafters(&target, None, true).await?,
        ))
    }

    /// How a catalog model serves without its catalog drafter. Catalog models use their catalog
    /// drafter unless it doesn't fit: withheld when the engine binds it and the model fits served
    /// plain and not with it, so the model still loads, served plain. Its built-in heads serve
    /// while the drafter is not on this computer when they are automatic: the engine binds them
    /// and they keep the model fitting.
    async fn catalog_drafter_fit(
        &self,
        model_id: &ModelId,
    ) -> Result<CatalogDrafterFit, InventoryError> {
        let target = self.model_target(model_id, Reach::Hub).await?;
        let candidates = self.enumerate_drafters(&target, None, false).await?;
        Ok(catalog_drafter_fit(&candidates))
    }

    /// A custom model's lineage hint and recommended drafter, from what `reach` reads.
    async fn enrich_model(
        &self,
        model_id: &ModelId,
        reach: Reach,
    ) -> Result<CustomModelEnrichmentResult, InventoryError> {
        let target = self.model_target(model_id, reach).await?;
        let plain = self
            .assess(&target.material, None, ModelMethod::Auto)
            .await?;
        let exact = match (&target.repository, &target.primary) {
            (Some((repository, commit)), Some(primary)) => {
                Some((repository.as_str(), commit.as_str(), primary.as_path()))
            }
            _ => None,
        };
        let matches = self
            .catalog_matches(exact, &target.lineage, plain.fingerprint.as_ref(), reach)
            .await;
        let variant_of = match matches
            .first()
            .filter(|(_, lineage)| *lineage == DrafterLineage::Exact)
        {
            Some((exact, _)) => Some(exact.model_id.clone()),
            None => self.shape_variant_of(plain.fingerprint.as_ref()).await,
        };
        let catalog_drafters = matches
            .into_iter()
            .map(|(matched, _)| matched.model_id)
            .collect();
        let license = target
            .snapshot
            .as_ref()
            .and_then(|snapshot| snapshot.license.clone());
        // Every automatic candidate, not only the first: the projection attaches the first one on
        // this computer, so built-in heads serve while a better drafter is still to be added.
        let drafters = automatic(self.enumerate_drafters(&target, None, false).await?);
        if reach == Reach::Hub
            && let Some(chosen) = self.chosen_drafter(model_id).await
        {
            self.learn_drafter_content(&chosen).await;
        }
        Ok(CustomModelEnrichmentResult {
            variant_of,
            drafters,
            catalog_drafters,
            license,
        })
    }

    /// Fill every custom model's lineage hint and automatic drafters. The projection asks the
    /// enrichment synchronously, so results are computed here in the background, one model at a
    /// time, keyed by the model and its exact target package, and the list is republished when
    /// they change. A result is recomputed whenever the custom-model inputs revision moves past
    /// the one it was computed at: a file appearing, moving or disappearing in any location (a
    /// model's folder or repository included), a drafter package installed or removed, a catalog
    /// installation, or a configuration change. Candidate enumeration is the only source of the
    /// automatic order, so the list and the attachment never disagree for long.
    ///
    /// Two passes run side by side: one publishes what is on this computer (the file's shape,
    /// the lineage its header and its repository's Hub answers already here declare, drafters in
    /// its folder, its built-in heads, the catalog drafter of a catalog model it matches by shape
    /// or that lineage) for every model with no Hub-reaching result yet,
    /// and the other adds what the Hub says. A model is never kept waiting on a slow or
    /// unreachable Hub for what its own files decide, and a load meanwhile uses that.
    pub fn start_enrichment(self: &Arc<Self>, resolver: Arc<ModelDomainResolver>) {
        let warmed = Arc::clone(self);
        tokio::spawn(async move {
            warmed.catalog_facts().await;
        });
        resolver.set_custom_model_enrichment(self.enrichment.clone());
        let _ = self.resolver.set(Arc::clone(&resolver));
        for reach in [Reach::Local, Reach::Hub] {
            let candidates = Arc::clone(self);
            let resolver = Arc::clone(&resolver);
            tokio::spawn(async move { candidates.enrich_custom_models(&resolver, reach).await });
        }
    }

    /// The enrichment pass of `reach` (see `start_enrichment`); the Hub pass also works out each
    /// catalog model's drafter fit.
    async fn enrich_custom_models(&self, resolver: &ModelDomainResolver, reach: Reach) {
        let enrichment = &self.enrichment;
        let mut changes = self.local.watch_local();
        let mut failed = BTreeMap::<(ModelId, String), (u64, std::time::Instant)>::new();
        loop {
            let Ok(revision) = resolver.custom_model_inputs_revision() else {
                return;
            };
            let models = self
                .local
                .list_local()
                .await
                .map(|response| {
                    response
                        .models
                        .into_iter()
                        .chain(response.hidden)
                        .map(|model| model.model_id)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            // Models with no result yet come first, so a model just added or installed is
            // never kept waiting behind recomputing the others; each result is published as
            // soon as it is known, and a pass restarts as soon as the inputs move again.
            let mut pending = models
                .into_iter()
                .filter_map(|model_id| {
                    let configuration = self
                        .assessor
                        .configuration_for(&ModelAssessmentSubject::Custom {
                            model_id: model_id.clone(),
                        })
                        .ok()?;
                    let key = (
                        model_id.clone(),
                        bundle_target(&configuration.bundle).id.0.clone(),
                    );
                    Some((enrichment.knows_model(&model_id, reach), key))
                })
                .collect::<Vec<_>>();
            pending.sort_by_key(|(known, _)| *known);
            for (_, key) in pending {
                if resolver.custom_model_inputs_revision().ok() != Some(revision) {
                    break;
                }
                if enrichment.is_current(&key, revision, reach)
                    || failed.get(&key).is_some_and(|(failed_at, at)| {
                        *failed_at == revision && at.elapsed() < ENRICHMENT_RETRY_DELAY
                    })
                {
                    continue;
                }
                match self.enrich_model(&key.0, reach).await {
                    Ok(result) => {
                        failed.remove(&key);
                        if enrichment.record(key, revision, reach, result) {
                            resolver.custom_model_enrichment_changed();
                        }
                    }
                    Err(error) => {
                        tracing::warn!(model.id = %key.0.as_str(), ?reach, %error, "custom model enrichment failed");
                        failed.insert(key, (revision, std::time::Instant::now()));
                    }
                }
            }
            let catalog_failed =
                reach == Reach::Hub && !self.fit_catalog_drafters(resolver, revision).await;
            tokio::select! {
                change = changes.next() => if change.is_none() { return },
                () = tokio::time::sleep(ENRICHMENT_RETRY_DELAY), if !failed.is_empty() || catalog_failed => {}
            }
        }
    }

    /// Whether each catalog model with a drafter fits only without it, once: it depends on the
    /// release material and this computer, never on the files here. Whether every one is known.
    async fn fit_catalog_drafters(&self, resolver: &ModelDomainResolver, revision: u64) -> bool {
        let catalog = self
            .catalog
            .list_catalog()
            .await
            .map(|response| response.models)
            .unwrap_or_default();
        let mut complete = true;
        for model in catalog {
            if resolver.custom_model_inputs_revision().ok() != Some(revision) {
                break;
            }
            if model.desired.speculative_method.is_none()
                || self.enrichment.knows_catalog_drafter_fit(&model.id)
            {
                continue;
            }
            match self.catalog_drafter_fit(&model.id).await {
                Ok(fit) => {
                    if self.enrichment.record_catalog_drafter_fit(model.id, fit) {
                        resolver.catalog_drafter_fit_changed();
                    }
                }
                Err(error) => {
                    tracing::warn!(model.id = %model.id.as_str(), %error, "catalog drafter fit failed");
                    complete = false;
                }
            }
        }
        complete
    }

    /// The drafter reference a model's configuration chooses explicitly.
    async fn chosen_drafter(&self, model_id: &ModelId) -> Option<DrafterReference> {
        let configuration = self.local.configuration().await.ok()?;
        configuration
            .drafters
            .into_iter()
            .find(|attachment| &attachment.model_id == model_id)
            .and_then(|attachment| match attachment.drafter {
                DrafterChoice::Reference { reference, .. } => Some(reference),
                DrafterChoice::Auto | DrafterChoice::None => None,
            })
    }

    /// Remembers a drafter reference's bytes for the local-models projection, which then uses a
    /// copy of them found anywhere on this computer, and republishes the list when they are new.
    fn remember_drafter_content(&self, reference: &DrafterReference, content: &DrafterContent) {
        if self
            .enrichment
            .record_drafter_content(reference, content.clone())
            && let Some(resolver) = self.resolver.get()
        {
            resolver.custom_model_enrichment_changed();
        }
    }

    /// Reads a drafter reference's header to learn its bytes, and remembers them.
    async fn learn_drafter_content(&self, reference: &DrafterReference) -> Option<DrafterContent> {
        let content = match reference {
            DrafterReference::Catalog { model_id } => {
                let matched = self.lineage.variant(model_id)?;
                let file = catalog_draft_reference(&matched.model)?;
                Box::pin(self.learn_drafter_content(&file)).await?
            }
            DrafterReference::HuggingFace {
                repository,
                commit,
                path,
            } => {
                let snapshot = self.snapshot(repository.as_str(), commit).await.ok()?;
                let file = snapshot_file(&snapshot, Path::new(path)).ok()?;
                let material = self
                    .hugging_face_drafter(&snapshot, Path::new(path))
                    .await
                    .ok()?;
                hugging_face_drafter_content(file, &material)
            }
            DrafterReference::Local { .. } | DrafterReference::Embedded => return None,
        };
        self.remember_drafter_content(reference, &content);
        Some(content)
    }

    // Assessment ---------------------------------------------------------------------------

    /// Assess exact header material: the engine prepares it, its supported context becomes the
    /// profile, and the result is cached by the material's identity and the method.
    async fn assess(
        &self,
        target: &TargetMaterial,
        drafter: Option<&DrafterMaterial>,
        method: ModelMethod,
    ) -> Result<CachedModelAssessment, InventoryError> {
        let environment = Arc::clone(self.assessor.environment());
        let evidence = self.assessment_evidence(
            &target.identity,
            drafter.map(|drafter| drafter.identity.as_str()),
            &method,
        )?;
        if let Some(cached) = self.models().read_model_assessment(&evidence) {
            return Ok(cached);
        }
        let gate = self.gate(&evidence).await;
        let _admitted = gate.lock().await;
        if let Some(cached) = self.models().read_model_assessment(&evidence) {
            return Ok(cached);
        }
        let paths = ModelPackagePaths {
            target: target.target.clone(),
            projector: target.projector.clone(),
            draft: drafter.map(|drafter| drafter.path.clone()),
            method,
        };
        let headers = StoredHeaders {
            store: Arc::clone(self.models()),
            known: target
                .headers
                .iter()
                .chain(drafter.iter().flat_map(|drafter| &drafter.headers))
                .map(|(path, digest)| (path.clone(), digest.clone()))
                .collect(),
        };
        let setup_environment = Arc::clone(&environment);
        let (prepared, fallback_context) = crate::spawn_blocking_traced(move || {
            let prepared = prepare_model_assessment_from(&paths, &headers, &setup_environment.setup);
            // An unsupported file is profiled at its training context when its header names one;
            // without one it has no profile.
            let fallback_context = match &prepared {
                Ok(PreparedModelAssessment::Unsupported { .. }) => headers
                    .header(&paths.target)
                    .ok()
                    .and_then(|header| {
                        magnitude_service_models::gguf::training_context_length(&header.directory)
                    })
                    .unwrap_or_default(),
                _ => 0,
            };
            (prepared, fallback_context)
        })
        .await
        .map_err(|error| {
            InventoryError::Internal(format!("candidate preparation failed: {error}"))
        })?;
        let prepared = prepared.map_err(preparation_error)?;
        let context = match &prepared {
            PreparedModelAssessment::Planned { facts, .. } => facts.context_limit,
            PreparedModelAssessment::Unsupported { .. } => fallback_context,
        };
        let profile = ModelAssessmentProfile {
            profile: ServingProfile {
                context_length: context,
            },
            performance_context_tokens: performance_depths(context),
        };
        let depths = profile.performance_context_tokens.clone();
        let finish_environment = Arc::clone(&environment);
        let assessed = crate::spawn_blocking_traced(move || {
            finish_model_assessment(&prepared, &finish_environment.setup, &depths)
        })
        .await
        .map_err(|error| InventoryError::Internal(format!("candidate assessment failed: {error}")))?
        .map_err(preparation_error)?;
        let key = ServableModelBundleKey(format!(
            "candidate_{:x}",
            Sha256::digest(evidence.as_bytes())
        ));
        let cached = model_assessment(assessed, &key, profile, &environment)?;
        self.models().write_model_assessment(&evidence, &cached);
        Ok(cached)
    }

    /// The cache key of an assessment in this service's environment.
    fn assessment_evidence(
        &self,
        target: &str,
        drafter: Option<&str>,
        method: &ModelMethod,
    ) -> Result<String, InventoryError> {
        assessment_evidence(&self.assessor.environment().id, target, drafter, method)
    }

    /// The identity `hugging_face_target` would give a version's material, from cached header
    /// digests alone, or `None` when a header was never fetched.
    fn cached_hugging_face_identity(
        &self,
        snapshot: &HuggingFaceRepositorySnapshot,
        group: &GgufGroup,
        projector: Option<&Path>,
    ) -> Option<String> {
        let mut headers = Vec::new();
        for path in group
            .files
            .iter()
            .map(|file| file.path.as_path())
            .chain(projector)
        {
            let file = snapshot_file(snapshot, path).ok()?;
            let digest = self.models().cached_hugging_face_header_digest(
                &snapshot.repository,
                &snapshot.commit,
                file,
            )?;
            headers.push((path_string(path), digest));
        }
        Some(header_identity(headers))
    }

    async fn gate(&self, key: &str) -> Arc<tokio::sync::Mutex<()>> {
        let mut gates = self.gates.lock().await;
        gates.retain(|_, gate| gate.strong_count() > 0);
        if let Some(gate) = gates.get(key).and_then(Weak::upgrade) {
            return gate;
        }
        let gate = Arc::new(tokio::sync::Mutex::new(()));
        gates.insert(key.to_owned(), Arc::downgrade(&gate));
        gate
    }

    // Material -------------------------------------------------------------------------------

    async fn snapshot(
        &self,
        repository: &str,
        revision: &str,
    ) -> Result<HuggingFaceRepositorySnapshot, InventoryError> {
        let request = HuggingFaceRepositoryRequest {
            repository: repository.to_owned(),
            revision: revision.to_owned(),
        };
        if is_commit(revision) {
            magnitude_service_models::pinned_hugging_face_repository(self.models(), request).await
        } else {
            refresh_hugging_face_repository(self.models(), request).await
        }
    }

    async fn hugging_face_target(
        &self,
        snapshot: &HuggingFaceRepositorySnapshot,
        primary: &Path,
        projector: Option<&Path>,
    ) -> Result<TargetMaterial, InventoryError> {
        let source = ModelPreviewSource {
            repository: snapshot.repository.clone(),
            revision: snapshot.commit.clone(),
            primary_gguf: primary.to_path_buf(),
            additional_components: projector
                .map(|projector| ModelPreviewComponentSource {
                    path: projector.to_path_buf(),
                    role: ModelPreviewComponentRole::Projector,
                })
                .into_iter()
                .collect(),
        };
        let prepared = self
            .models()
            .prepare_preview_from_repository_snapshot(&source, snapshot)
            .await?;
        let (target, projector) = resolved_paths(&prepared.model)?;
        let identity = header_identity(
            prepared
                .headers
                .iter()
                .map(|header| (path_string(&header.path), header.digest.clone())),
        );
        Ok(TargetMaterial {
            target,
            projector,
            identity,
            headers: preview_headers(&prepared),
            _guards: vec![Arc::new(prepared)],
        })
    }

    async fn hugging_face_drafter(
        &self,
        snapshot: &HuggingFaceRepositorySnapshot,
        path: &Path,
    ) -> Result<DrafterMaterial, InventoryError> {
        let source = ModelPreviewSource {
            repository: snapshot.repository.clone(),
            revision: snapshot.commit.clone(),
            primary_gguf: path.to_path_buf(),
            additional_components: Vec::new(),
        };
        let prepared = self
            .models()
            .prepare_preview_from_repository_snapshot(&source, snapshot)
            .await?;
        let (path, _) = resolved_paths(&prepared.model)?;
        let identity = prepared
            .headers
            .iter()
            .map(|header| header.digest.clone())
            .collect::<Vec<_>>()
            .join(",");
        Ok(DrafterMaterial {
            path,
            identity,
            headers: preview_headers(&prepared),
            _guards: vec![Arc::new(prepared)],
        })
    }

    /// Local material, identified by header digests from the identity cache: an unchanged file is
    /// never read again, and the engine reads its header from the header cache.
    fn local_target(
        &self,
        target: PathBuf,
        projector: Option<PathBuf>,
    ) -> Result<TargetMaterial, InventoryError> {
        let mut identity = Vec::new();
        let mut headers = BTreeMap::new();
        for path in std::iter::once(&target).chain(projector.iter()) {
            let digest = self.models().local_gguf(path)?.inspection.header_digest;
            identity.push(digest.clone());
            headers.insert(path.clone(), digest);
        }
        Ok(TargetMaterial {
            target,
            projector,
            identity: identity.join(","),
            headers,
            _guards: Vec::new(),
        })
    }

    fn local_drafter(&self, path: PathBuf) -> Result<DrafterMaterial, InventoryError> {
        let identity = self.models().local_gguf(&path)?.inspection.header_digest;
        Ok(DrafterMaterial {
            headers: BTreeMap::from([(path.clone(), identity.clone())]),
            identity,
            path,
            _guards: Vec::new(),
        })
    }

    /// The repositories a GGUF names as its base models, from its cached facts.
    fn header_lineage(&self, path: &Path, known: &BTreeMap<PathBuf, String>) -> BTreeSet<String> {
        let base_models = match known
            .get(path)
            .and_then(|digest| self.models().cached_header(digest))
        {
            Some(header) => header_inspection(&header)
                .map(|(_, inspection)| inspection.base_models)
                .unwrap_or_default(),
            None => self
                .models()
                .local_gguf(path)
                .map(|gguf| gguf.inspection.base_models)
                .unwrap_or_default(),
        };
        base_model_repositories(&base_models)
    }

    /// A resolved bundle's target (and projector) as material, identified by its exact package.
    fn resolved_target(
        resolved: magnitude_service_contracts::models::ResolvedServableModelBundle,
    ) -> Result<TargetMaterial, InventoryError> {
        let (target, projector) = resolved_paths(&resolved.target_model)?;
        let identity = format!("package:{}", bundle_target(&resolved.bundle).id.0);
        Ok(TargetMaterial {
            target,
            projector,
            identity,
            headers: BTreeMap::new(),
            _guards: vec![Arc::new(resolved)],
        })
    }

    /// The separate draft of a resolved bundle as material.
    fn resolved_drafter(
        resolved: magnitude_service_contracts::models::ResolvedServableModelBundle,
    ) -> Result<Option<DrafterMaterial>, InventoryError> {
        let ServableModelBundle::SpeculativeDecoding {
            draft_source: SpeculativeDraftSource::Separate { draft },
            ..
        } = &resolved.bundle
        else {
            return Ok(None);
        };
        let identity = format!("package:{}", draft.id.0);
        let path = resolved
            .draft_model
            .as_ref()
            .and_then(|model| {
                model
                    .components
                    .iter()
                    .filter(|component| {
                        matches!(
                            component.role,
                            ComponentRole::Weights | ComponentRole::Draft
                        )
                    })
                    .map(|component| component.path.clone())
                    .min()
            })
            .ok_or_else(|| {
                InventoryError::Integrity("the draft resolved without weights".to_owned())
            })?;
        Ok(Some(DrafterMaterial {
            path,
            identity,
            headers: BTreeMap::new(),
            _guards: vec![Arc::new(resolved)],
        }))
    }

    // Classification -------------------------------------------------------------------------

    /// Every complete GGUF of a snapshot, classified from its header. Classifications are cached
    /// by header, so only files never seen before are read: one whole header first, whose large
    /// metadata entries (the tokenizer) the others then share (see the store's
    /// `hugging_face_gguf_header`), read concurrently.
    async fn classify_hugging_face(
        &self,
        snapshot: &HuggingFaceRepositorySnapshot,
        progress: &InspectionProgress,
    ) -> Result<Vec<ClassifiedGroup>, InventoryError> {
        let files = snapshot
            .gguf_files
            .iter()
            .map(|file| CandidateGguf {
                path: file.path.clone(),
                size_bytes: file.size_bytes,
            })
            .collect::<Vec<_>>();
        let groups = group_ggufs(&files);
        // Until their headers are read, the rows are the files that may be versions, named by
        // their files; a name only orders and labels them here, and headers decide every role.
        progress.reading_headers(
            groups
                .iter()
                .filter(|group| !is_projector_name(&group.primary().path))
                .map(|group| (version_label(&group.primary().path, None), group.size_bytes()))
                .collect(),
            0,
            groups.len(),
        );
        let mut cached = Vec::new();
        let mut uncached = Vec::new();
        for group in groups {
            let file = snapshot_file(snapshot, &group.primary().path)?;
            match self.models().cached_hugging_face_classification(
                &snapshot.repository,
                &snapshot.commit,
                file,
                &self.architectures,
            ) {
                Some((digest, classification)) => {
                    cached.push(classified_group(group, digest, classification));
                    progress.header_read();
                }
                None => uncached.push(group),
            }
        }
        let mut classified = cached;
        if uncached.is_empty() {
            return Ok(classified);
        }
        // Read one header whole to lend its tokenizer to the others: an already cached one, or
        // the largest file's, the likeliest to be a complete model carrying the tokenizer (a
        // projector, an imatrix or a drafter may carry none). Names only order the requests;
        // headers decide every role. A probe whose header holds nothing to share (an imatrix)
        // or can't be read at all leaves the next largest file to be read whole.
        let probe_order =
            |group: &GgufGroup| (!is_projector_name(&group.primary().path), group.size_bytes());
        let mut reference = classified
            .iter()
            .map(|classified| &classified.group)
            .max_by_key(|group| probe_order(group))
            .and_then(|group| {
                let file = snapshot_file(snapshot, &group.primary().path).ok()?;
                self.models().hugging_face_reference_header(
                    &snapshot.repository,
                    &snapshot.commit,
                    file,
                )
            });
        while reference.is_none() && !uncached.is_empty() {
            let probe = uncached
                .iter()
                .enumerate()
                .max_by_key(|(_, group)| probe_order(group))
                .map_or(0, |(index, _)| index);
            let probe = uncached.remove(probe);
            let probe_file = snapshot_file(snapshot, &probe.primary().path)?;
            classified.push(
                self.classify_hugging_face_group(snapshot, probe, &[])
                    .await?,
            );
            progress.header_read();
            reference = self.models().hugging_face_reference_header(
                &snapshot.repository,
                &snapshot.commit,
                probe_file,
            );
        }
        let references = reference.as_deref().into_iter().collect::<Vec<_>>();
        let rest = futures_util::stream::iter(
            uncached.into_iter().map(|group| async {
                let classified = self
                    .classify_hugging_face_group(snapshot, group, &references)
                    .await;
                progress.header_read();
                classified
            }),
        )
        .buffered(HEADER_CONCURRENCY)
        .collect::<Vec<_>>()
        .await;
        for group in rest {
            classified.push(group?);
        }
        let order = files
            .iter()
            .enumerate()
            .map(|(index, file)| (file.path.clone(), index))
            .collect::<BTreeMap<_, _>>();
        classified.sort_by_key(|group| order.get(&group.group.primary().path).copied());
        Ok(classified)
    }

    async fn classify_hugging_face_group(
        &self,
        snapshot: &HuggingFaceRepositorySnapshot,
        group: GgufGroup,
        references: &[&[u8]],
    ) -> Result<ClassifiedGroup, InventoryError> {
        let file = snapshot_file(snapshot, &group.primary().path)?;
        match self
            .models()
            .classify_hugging_face_gguf(
                &snapshot.repository,
                &snapshot.commit,
                file,
                &self.architectures,
                references,
            )
            .await
        {
            Ok((header_digest, classification)) => {
                Ok(classified_group(group, header_digest, classification))
            }
            Err(InventoryError::Integrity(detail)) => Ok(unreadable_group(group, detail)),
            Err(error) => Err(error),
        }
    }

    /// A version's verdict, when its header is readable and its architecture is one the engine
    /// runs; an invalid header is `Unreadable`. A failure to read its material leaves the version
    /// without a verdict rather than failing the inspection.
    async fn version_assessment(
        &self,
        group: &ClassifiedGroup,
        identity: Option<String>,
        material: impl std::future::Future<Output = Result<TargetMaterial, InventoryError>>,
    ) -> Option<CachedModelAssessment> {
        if let Some(detail) = &group.unreadable {
            return Some(super::assessor::unreadable_assessment(detail.clone()));
        }
        // Every readable version gets the engine's plain verdict, an architecture it doesn't
        // recognize included, so its reasons are the engine's own. Its drafter (built-in heads
        // among them) is composed only for the recommended version; see `compose_recommended`.
        if let Some(cached) = identity
            .and_then(|identity| {
                self.assessment_evidence(&identity, None, &ModelMethod::Plain)
                    .ok()
            })
            .and_then(|evidence| self.models().read_model_assessment(&evidence))
        {
            return Some(cached);
        }
        let material = material
            .await
            .map_err(|error| tracing::warn!(%error, "candidate version material is unavailable"))
            .ok()?;
        self.assess(&material, None, ModelMethod::Plain)
            .await
            .map_err(|error| tracing::warn!(%error, "candidate version assessment failed"))
            .ok()
    }

    /// A version's verdict is the one adding it gives the installed model: its target and
    /// projector with the drafter preselected for it (see `enumerate_drafters`). An unsupported
    /// version keeps its own verdict, and so does one whose drafters can't be worked out.
    /// The versions' verdicts with the recommended version's composed with its preselected drafter,
    /// as it installs; every other version keeps its plain verdict, and its drafters and composed
    /// verdict come from the per-version drafter candidates when it is selected. Composing every
    /// version would assess each with every drafter candidate.
    async fn compose_recommended(
        &self,
        source: &CandidateSource,
        targets: &[&ClassifiedGroup],
        mut assessments: Vec<Option<CachedModelAssessment>>,
        projector: Option<CandidateFile>,
        progress: &InspectionProgress,
    ) -> Vec<Option<CachedModelAssessment>> {
        let recommended = recommended_version(
            targets
                .iter()
                .zip(&assessments)
                .map(|(group, assessment)| (*group, assessment.as_ref())),
        );
        if let Some(index) = recommended {
            progress.checking_drafters(index);
            let plain = assessments[index].take();
            assessments[index] = self
                .with_preselected_drafter(
                    version_candidate(source, targets[index], projector),
                    plain,
                    progress,
                )
                .await;
        }
        assessments
    }

    async fn with_preselected_drafter(
        &self,
        candidate: ModelCandidate,
        plain: Option<CachedModelAssessment>,
        progress: &InspectionProgress,
    ) -> Option<CachedModelAssessment> {
        let plain = plain?;
        if matches!(plain.profile, ModelAssessment::Unsupported { .. }) {
            return Some(plain);
        }
        match self.preselected_assessment(&candidate, progress).await {
            Ok(Some(assessed)) => Some(assessed),
            Ok(None) => Some(plain),
            Err(error) => {
                tracing::warn!(%error, "the preselected drafter of a candidate version is unavailable");
                Some(plain)
            }
        }
    }

    /// The assessment of a candidate with its preselected drafter, assessed exactly as its drafter
    /// candidate was (so it is a cache hit): served plain when none is preselected, as the
    /// installed model is, with its built-in heads when those are preselected, or `None` when the
    /// preselected candidate names no method (its plain verdict stands).
    async fn preselected_assessment(
        &self,
        candidate: &ModelCandidate,
        progress: &InspectionProgress,
    ) -> Result<Option<CachedModelAssessment>, InventoryError> {
        let target = self.candidate_target(candidate).await?;
        let drafters = self.drafter_list(&target, None, true, progress).await?;
        let Some(chosen) = recommended(drafters.candidates) else {
            return Ok(Some(
                self.assess(&target.material, None, ModelMethod::Plain)
                    .await?,
            ));
        };
        let Some(method) = chosen.method else {
            return Ok(None);
        };
        let engine = match &chosen.reference {
            DrafterReference::Embedded => {
                return Ok(Some(
                    self.assess(&target.material, None, ModelMethod::Auto)
                        .await?,
                ));
            }
            DrafterReference::Catalog { .. } => engine_method(&method),
            DrafterReference::HuggingFace { .. } | DrafterReference::Local { .. } => {
                ModelMethod::Auto
            }
        };
        let drafter = self
            .candidate_drafter_material(&CandidateDrafter {
                reference: chosen.reference,
                method,
            })
            .await?;
        Ok(Some(
            self.assess(&target.material, drafter.as_ref(), engine)
                .await?,
        ))
    }

    /// A local folder's GGUFs, found and classified off the async runtime: reading headers is
    /// file I/O and hashing.
    async fn scan_local(
        &self,
        root: &Path,
        depth: usize,
    ) -> (Vec<CandidateGguf>, Vec<ClassifiedGroup>) {
        let root = root.to_path_buf();
        let architectures = self.architectures.clone();
        let store = Arc::clone(self.models());
        crate::spawn_blocking_traced(move || {
            let files = scan_ggufs(&root, depth);
            let classified = classify_local(&store, &root, &files, &architectures);
            (files, classified)
        })
        .await
        .unwrap_or_default()
    }

    // Inspection -----------------------------------------------------------------------------

    async fn inspect_hugging_face(
        &self,
        repository: HuggingFaceRepositoryId,
        revision: String,
        progress: &InspectionProgress,
    ) -> Result<CandidateInspection, InventoryError> {
        let snapshot = self.snapshot(repository.as_str(), &revision).await?;
        let classified = self.classify_hugging_face(&snapshot, progress).await?;
        let projector = choose_projector(&classified);
        let targets = classified
            .iter()
            .filter(|group| matches!(group.role, CandidateFileRole::Target { .. }))
            .collect::<Vec<_>>();
        let unsupported_architecture = self.unsupported_architecture(&targets);
        let custom = self.custom_models().await;
        let catalog_installed = self.installed_catalog_models().await;
        let projector_path = projector.map(|projector| projector.group.primary().path.clone());
        progress.checking_versions(version_rows(&targets));
        let mut pending = Vec::new();
        for group in &targets {
            pending.push(self.version_assessment(
                group,
                self.cached_hugging_face_identity(
                    &snapshot,
                    &group.group,
                    projector_path.as_deref(),
                ),
                self.hugging_face_target(
                    &snapshot,
                    &group.group.primary().path,
                    projector_path.as_deref(),
                ),
            ));
        }
        let assessments = checked_versions(pending, progress).await;
        let source = CandidateSource::HuggingFace {
            repository: repository.clone(),
            commit: snapshot.commit.clone(),
        };
        let projector_file = projector.map(|projector| candidate_file(projector.group.primary()));
        let assessments = self
            .compose_recommended(&source, &targets, assessments, projector_file.clone(), progress)
            .await;
        let versions = targets
            .iter()
            .copied()
            .zip(assessments)
            .filter_map(|(group, assessment)| {
                let model_id = hugging_face_model_id(&snapshot.repository, &group.group)?;
                // The catalog target at its commit, or its bytes at another commit of its
                // repository, which install as that catalog model all the same.
                let catalog_model = self
                    .lineage
                    .exact_target(
                        &snapshot.repository,
                        &snapshot.commit,
                        &group.group.primary().path,
                    )
                    .or_else(|| {
                        self.lineage.target_with_weights(
                            &snapshot.repository,
                            &snapshot_weights(&snapshot, &group.group)?,
                        )
                    })
                    .map(|matched| matched.model_id);
                let on_this_computer = catalog_model
                    .clone()
                    .filter(|model_id| catalog_installed.contains(model_id))
                    .or_else(|| {
                        on_this_computer(
                            self.models(),
                            &custom,
                            Some((&model_id, snapshot.commit.as_str())),
                            &group.group,
                            &group.header_digest,
                            group.local_digest.as_deref(),
                            None,
                        )
                    });
                Some(AssessedVersion {
                    group,
                    assessment,
                    model_id,
                    catalog_model,
                    on_this_computer,
                })
            })
            .collect::<Vec<_>>();
        let variant_of = self.inspection_variant_of(&versions).await;
        let projector_bytes = projector.map_or(0, |projector| projector.group.size_bytes());
        Ok(CandidateInspection {
            source: CandidateSource::HuggingFace {
                repository: HuggingFaceRepositoryId::new(snapshot.repository.clone())
                    .map_err(|error| InventoryError::InvalidRequest(error.to_string()))?,
                commit: snapshot.commit.clone(),
            },
            display_name: display_name(&snapshot.repository),
            license: snapshot.license.clone(),
            variant_of,
            versions: candidate_versions(versions, Some(projector_bytes)),
            projector: projector.map(|projector| candidate_file(projector.group.primary())),
            unsupported_architecture,
        })
    }

    async fn inspect_local(
        &self,
        path: PathBuf,
        progress: &InspectionProgress,
    ) -> Result<CandidateInspection, InventoryError> {
        let metadata = std::fs::metadata(&path)
            .map_err(|error| InventoryError::NotFound(format!("{}: {error}", path.display())))?;
        let (root, selected) = if metadata.is_file() {
            let root = path.parent().map(Path::to_path_buf).ok_or_else(|| {
                InventoryError::InvalidRequest("a file needs a folder".to_owned())
            })?;
            let name = path
                .file_name()
                .map(PathBuf::from)
                .ok_or_else(|| InventoryError::InvalidRequest("invalid file path".to_owned()))?;
            (root, Some(name))
        } else {
            (path.clone(), None)
        };
        let depth = if selected.is_some() {
            1
        } else {
            MAX_LOCAL_SCAN_DEPTH
        };
        let (_, classified) = self.scan_local(&root, depth).await;
        let projector = choose_projector(&classified);
        let targets = classified
            .iter()
            .filter(|group| matches!(group.role, CandidateFileRole::Target { .. }))
            .filter(|group| {
                selected.as_ref().is_none_or(|selected| {
                    group.group.files.iter().any(|file| &file.path == selected)
                })
            })
            .collect::<Vec<_>>();
        if selected.is_some() && targets.is_empty() {
            if let Err(error) = local_directory(&path) {
                return Err(InventoryError::InvalidRequest(format!(
                    "{} can't be read as a GGUF model: {error}",
                    path.display()
                )));
            }
            return Err(InventoryError::InvalidRequest(format!(
                "{} is not a model the engine could run on its own (it is a projector, a drafter or an incomplete split file)",
                path.display()
            )));
        }
        let unsupported_architecture = self.unsupported_architecture(&targets);
        let custom = self.custom_models().await;
        let projector_path = projector.map(|projector| root.join(&projector.group.primary().path));
        progress.checking_versions(version_rows(&targets));
        let mut pending = Vec::new();
        for group in &targets {
            let material = self.local_target(
                root.join(&group.group.primary().path),
                projector_path.clone(),
            );
            let unreadable = unreadable_local_group(self.models(), &root, &group.group);
            pending.push(async move {
                match unreadable {
                    Some(assessment) => Some(assessment),
                    None => {
                        self.version_assessment(group, None, async { material })
                            .await
                    }
                }
            });
        }
        let assessments = checked_versions(pending, progress).await;
        let source = CandidateSource::Local { path: root.clone() };
        let projector_file = projector.map(|projector| candidate_file(projector.group.primary()));
        let assessments = self
            .compose_recommended(&source, &targets, assessments, projector_file.clone(), progress)
            .await;
        let versions = targets
            .iter()
            .copied()
            .zip(assessments)
            .filter_map(|(group, assessment)| {
                // A file with an installed catalog target's bytes (a copy, or a link to one) is
                // that catalog model, which is on this computer.
                let catalog_model = group.local_digest.as_deref().and_then(|digest| {
                    self.models()
                        .catalog_model_with_content(digest, group.group.primary().size_bytes)
                });
                let on_this_computer = catalog_model.clone().or_else(|| {
                    on_this_computer(
                        self.models(),
                        &custom,
                        None,
                        &group.group,
                        &group.header_digest,
                        group.local_digest.as_deref(),
                        Some(&root),
                    )
                });
                // A file a listed model is found at keeps that model's identity (a Hugging Face one
                // when its location gives it); any other file, a copy of a listed model's bytes
                // included, is identified by its header digest.
                let model_id = listed_at(&custom, &group.group, &root)
                    .or_else(|| local_model_id(group.local_digest.as_deref()?))?;
                Some(AssessedVersion {
                    group,
                    assessment,
                    model_id,
                    catalog_model,
                    on_this_computer,
                })
            })
            .collect::<Vec<_>>();
        let variant_of = self.inspection_variant_of(&versions).await;
        let name = selected
            .as_ref()
            .and_then(|selected| selected.file_stem())
            .or_else(|| root.file_name())
            .and_then(|name| name.to_str())
            .unwrap_or_default();
        Ok(CandidateInspection {
            source: CandidateSource::Local { path: root.clone() },
            display_name: display_name(name),
            license: None,
            variant_of,
            versions: candidate_versions(versions, None),
            projector: projector.map(|projector| candidate_file(projector.group.primary())),
            unsupported_architecture,
        })
    }

    fn recognized(&self, group: &ClassifiedGroup) -> bool {
        matches!(
            &group.role,
            CandidateFileRole::Target { architecture: Some(architecture) }
                if self.architectures.targets.contains(architecture)
        )
    }

    /// When no version's architecture is one the engine runs, why: one reason per architecture.
    fn unsupported_architecture(
        &self,
        targets: &[&ClassifiedGroup],
    ) -> Option<Vec<magnitude_service_contracts::models::UnsupportedReason>> {
        let readable = targets
            .iter()
            .filter(|group| group.unreadable.is_none())
            .collect::<Vec<_>>();
        if readable.is_empty() || readable.iter().any(|group| self.recognized(group)) {
            return None;
        }
        Some(
            readable
                .iter()
                .filter_map(|group| match &group.role {
                    CandidateFileRole::Target { architecture } => Some(architecture.clone()),
                    _ => None,
                })
                .collect::<BTreeSet<_>>()
                .into_iter()
                .map(|architecture| {
                    // A file whose header names no architecture can't be read as a model, as in
                    // search; an architecture is never empty.
                    magnitude_service_contracts::models::UnsupportedReason::Known(match architecture {
                        Some(architecture) => {
                            magnitude_service_contracts::models::KnownUnsupportedReason::Architecture {
                                architecture,
                            }
                        }
                        None => magnitude_service_contracts::models::KnownUnsupportedReason::Unreadable,
                    })
                })
                .collect(),
        )
    }

    /// Every custom model ICN lists, hidden ones included: hiding never changes a file's identity.
    async fn custom_models(&self) -> Vec<CustomModel> {
        self.local
            .list_local()
            .await
            .map(|response| response.models.into_iter().chain(response.hidden).collect())
            .unwrap_or_default()
    }

    async fn installed_catalog_models(&self) -> BTreeSet<ModelId> {
        self.catalog
            .list_catalog()
            .await
            .map(|response| {
                response
                    .models
                    .into_iter()
                    .filter(|model| {
                        matches!(model.local_state, CatalogModelState::Installed { .. })
                    })
                    .map(|model| model.id)
                    .collect()
            })
            .unwrap_or_default()
    }

    // Catalog matching ----------------------------------------------------------------------

    /// Each catalog model's shape fingerprint, computed once and concurrently from the release
    /// header bundle.
    async fn catalog_facts(&self) -> &CatalogFacts {
        self.catalog_facts
            .get_or_init(|| async {
                let representatives = self.lineage.representatives().collect::<Vec<_>>();
                let facts = futures_util::stream::iter(representatives.into_iter().map(|candidate| async move {
                    let resolved = self
                        .assessor
                        .resolve_material(candidate.model.configuration.bundle.clone())
                        .await
                        .map_err(|error| tracing::warn!(%error, model.id = %candidate.model_id.as_str(), "catalog fingerprint material is unavailable"))
                        .ok()?;
                    let model_id = candidate.model_id;
                    crate::spawn_blocking_traced(move || {
                        let target = resolved_paths(&resolved.target_model).ok()?.0;
                        let fingerprint = EngineModelFingerprint::from_path(&target)
                            .ok()
                            .map(super::assessor::model_fingerprint);
                        Some((model_id, fingerprint))
                    })
                    .await
                    .ok()
                    .flatten()
                }))
                .buffered(assessment_concurrency())
                .filter_map(std::future::ready)
                .collect::<Vec<_>>()
                .await;
                let mut catalog = CatalogFacts::default();
                for (model_id, fingerprint) in facts {
                    if let Some(fingerprint) = fingerprint {
                        catalog.fingerprints.push((model_id, fingerprint));
                    }
                }
                catalog
            })
            .await
    }

    /// The catalog model a model of `fingerprint`'s shape is a variant of, by the rule search uses
    /// (see `ModelShape::variant_of`).
    async fn shape_variant_of(&self, fingerprint: Option<&ModelFingerprint>) -> Option<ModelId> {
        ModelShape::of(fingerprint?).variant_of(&self.catalog_facts().await.fingerprints)
    }

    /// The catalog model an inspected candidate is a variant of: the catalog model a version is,
    /// else the shape match of the first version whose header states a shape, read from its
    /// metadata as search reads it (so a weight type the engine doesn't run never hides it), else
    /// of the first version the engine fingerprinted.
    async fn inspection_variant_of(&self, versions: &[AssessedVersion<'_>]) -> Option<ModelId> {
        if let Some(catalog_model) = versions.iter().find_map(|version| version.catalog_model.clone()) {
            return Some(catalog_model);
        }
        let header_shape = versions.iter().find_map(|version| {
            let bytes = self.models().cached_header(&version.group.header_digest)?;
            ModelShape::from_header_prefix(&bytes)
        });
        match header_shape {
            Some(shape) => shape.variant_of(&self.catalog_facts().await.fingerprints),
            None => {
                let fingerprint = versions
                    .iter()
                    .find_map(|version| version.assessment.as_ref()?.fingerprint.clone());
                self.shape_variant_of(fingerprint.as_ref()).await
            }
        }
    }

    /// The catalog models a target derives from, best first: an exact catalog target alone; else
    /// the lineage match that is the target's fingerprint match, else the first lineage match in
    /// catalog order; then the fingerprint match when it is another one, whose drafter is the
    /// fallback when the lineage match's is rejected. Locally, catalog upstreams match as far as
    /// the Hub's answers are on this computer.
    ///
    /// The fingerprint match is the catalog model whose fingerprint equals the target's, else the
    /// one the target is a variant of by shape (see `ModelShape::variant_of`), so every model shown
    /// as a variant of a catalog model is offered that model's drafter even when its converter
    /// wrote another token table (Ollama writes unused entries as `[PAD<n>]`). The engine's verdict
    /// decides whether it binds, and a drafter matched this way is never automatic (see
    /// `known_to_work`).
    async fn catalog_matches(
        &self,
        exact: Option<(&str, &str, &Path)>,
        lineage: &BTreeSet<String>,
        fingerprint: Option<&ModelFingerprint>,
        reach: Reach,
    ) -> Vec<(CatalogLineageMatch, DrafterLineage)> {
        if let Some((repository, commit, primary)) = exact
            && let Some(matched) = self.lineage.exact_target(repository, commit, primary)
        {
            return vec![(matched, DrafterLineage::Exact)];
        }
        let fingerprinted = match fingerprint {
            Some(fingerprint) => {
                let catalog = &self.catalog_facts().await.fingerprints;
                catalog
                    .iter()
                    .find(|(_, candidate)| candidate == fingerprint)
                    .map(|(model_id, _)| model_id.clone())
                    .or_else(|| ModelShape::of(fingerprint).variant_of(catalog))
            }
            None => None,
        };
        let mut lineage_matches = match reach {
            Reach::Local => self.lineage.match_known_lineages(lineage),
            Reach::Hub => self.lineage.match_lineages(lineage).await,
        };
        if let Some(fingerprinted) = &fingerprinted {
            lineage_matches.sort_by_key(|matched| matched.model_id != *fingerprinted);
        }
        let mut matches = lineage_matches
            .into_iter()
            .take(1)
            .map(|matched| (matched, DrafterLineage::Lineage))
            .collect::<Vec<_>>();
        if let Some(fingerprinted) = fingerprinted
            && !matches
                .iter()
                .any(|(matched, _)| matched.model_id == fingerprinted)
            && let Some(matched) = self.lineage.variant(&fingerprinted)
        {
            matches.push((matched, DrafterLineage::Fingerprint));
        }
        matches
    }

    // Drafter targets ------------------------------------------------------------------------

    async fn candidate_target(
        &self,
        candidate: &ModelCandidate,
    ) -> Result<DrafterTarget, InventoryError> {
        let primary = candidate.target.first().map(PathBuf::from).ok_or_else(|| {
            InventoryError::InvalidRequest("a candidate names its target".to_owned())
        })?;
        match &candidate.source {
            CandidateSource::HuggingFace { repository, commit } => {
                let snapshot = self.snapshot(repository.as_str(), commit).await?;
                let projector = candidate
                    .projector
                    .as_ref()
                    .map(|file| PathBuf::from(&file.path));
                let material = self
                    .hugging_face_target(&snapshot, &primary, projector.as_deref())
                    .await?;
                let mut lineage = self
                    .lineage
                    .hub()
                    .lineage(&snapshot.repository, Some(snapshot.base_models.clone()))
                    .await;
                lineage.extend(self.header_lineage(&material.target, &material.headers));
                Ok(DrafterTarget {
                    material,
                    repository: Some((snapshot.repository.clone(), snapshot.commit.clone())),
                    primary: Some(primary),
                    folders: Vec::new(),
                    snapshot: Some(snapshot),
                    lineage,
                    catalog: None,
                    reach: Reach::Hub,
                })
            }
            CandidateSource::Local { path } => {
                let target = path.join(&primary);
                let projector = candidate
                    .projector
                    .as_ref()
                    .map(|file| path.join(&file.path));
                let material = self.local_target(target, projector)?;
                let lineage = self.header_lineage(&material.target, &material.headers);
                let folders = material
                    .target
                    .parent()
                    .map(|parent| DrafterFolder::whole(parent.to_path_buf()))
                    .into_iter()
                    .collect();
                Ok(DrafterTarget {
                    material,
                    repository: None,
                    primary: None,
                    folders,
                    snapshot: None,
                    lineage,
                    catalog: None,
                    reach: Reach::Hub,
                })
            }
        }
    }

    /// A model's target: a catalog model's release material, or a custom model's installed
    /// target package. Locally, a model from a known Hub repository has no snapshot, and the
    /// lineage its header declares and its repository's Hub lineage already on this computer.
    async fn model_target(
        &self,
        model_id: &ModelId,
        reach: Reach,
    ) -> Result<DrafterTarget, InventoryError> {
        if let ParsedModelId::Catalog { .. } = model_id.parsed() {
            let matched = self
                .lineage
                .variant(model_id)
                .ok_or_else(|| InventoryError::NotFound(model_id.as_str().to_owned()))?;
            let resolved = self
                .assessor
                .resolve_material(matched.model.configuration.bundle.clone())
                .await?;
            return Ok(DrafterTarget {
                material: Self::resolved_target(resolved)?,
                repository: None,
                primary: None,
                folders: Vec::new(),
                snapshot: None,
                lineage: BTreeSet::new(),
                catalog: Some(model_id.clone()),
                reach,
            });
        }
        let model = self
            .custom_models()
            .await
            .into_iter()
            .find(|model| &model.model_id == model_id)
            .ok_or_else(|| InventoryError::NotFound(model_id.as_str().to_owned()))?;
        let configuration = self
            .assessor
            .configuration_for(&ModelAssessmentSubject::Custom {
                model_id: model_id.clone(),
            })?;
        let target = bundle_target(&configuration.bundle).clone();
        let resolved = self
            .assessor
            .resolve_material(ServableModelBundle::Standalone {
                package: target.clone(),
            })
            .await?;
        let material = Self::resolved_target(resolved)?;
        let repository = match (&model.origin, &target.source) {
            (CustomModelOrigin::Added { repository, commit }, _) => {
                Some((repository.as_str().to_owned(), commit.clone()))
            }
            (
                _,
                ModelPackageSource::HuggingFace {
                    repository,
                    revision,
                },
            ) => Some((repository.clone(), revision.clone())),
            _ => match model_id.parsed() {
                ParsedModelId::HuggingFace { repository_id, .. } => {
                    Some((repository_id.as_str().to_owned(), String::new()))
                }
                // A `local:` model whose location names a repository (an Ollama `hf.co` pull)
                // takes its lineage and drafters from it, even when no file there has its bytes.
                _ => self
                    .resolver
                    .get()
                    .and_then(|resolver| resolver.custom_model_lineage_repository(model_id).ok())
                    .flatten()
                    .map(|repository| (repository, String::new())),
            },
        };
        let primary = target
            .files
            .iter()
            .filter(|file| file.role == ModelFileRole::Weights)
            .map(|file| file.path.clone())
            .min();
        let mut lineage = self.header_lineage(&material.target, &material.headers);
        // A model from a known Hub repository takes the lineage that repository declares, at its
        // commit or else at `main`, and lists that repository's drafters; never whatever else
        // shares its folder. Locally, the lineage is the Hub's answers already on this computer.
        let snapshot = match &repository {
            Some((repository, commit)) => {
                let revision = if is_commit(commit) {
                    commit.as_str()
                } else {
                    "main"
                };
                match reach {
                    Reach::Hub => {
                        let snapshot = self.snapshot(repository, revision).await.ok();
                        let first_hop = snapshot
                            .as_ref()
                            .map(|snapshot| snapshot.base_models.clone());
                        lineage.extend(self.lineage.hub().lineage(repository, first_hop).await);
                        snapshot
                    }
                    Reach::Local => {
                        let first_hop = magnitude_service_models::cached_hugging_face_repository(
                            self.models(),
                            &HuggingFaceRepositoryRequest {
                                repository: repository.clone(),
                                revision: revision.to_owned(),
                            },
                        )
                        .map(|snapshot| snapshot.base_models);
                        lineage.extend(self.lineage.hub().known_lineage(repository, first_hop));
                        None
                    }
                }
            }
            None => None,
        };
        // Each place this model's own bytes are found gives its folder; never the folder of the
        // file its target package resolves to, nor of another model sharing its header.
        let locations = self
            .resolver
            .get()
            .and_then(|resolver| resolver.custom_model_locations(model_id).ok())
            .filter(|locations| !locations.is_empty())
            .unwrap_or_else(|| {
                model
                    .files
                    .iter()
                    .map(|file| file.path.clone())
                    .take(1)
                    .collect()
            });
        let mut folders = Vec::<DrafterFolder>::new();
        for location in &locations {
            if let Some(folder) = drafter_folder(
                &model.origin,
                repository
                    .as_ref()
                    .map(|(repository, _)| repository.as_str()),
                snapshot.is_some(),
                location.parent(),
            ) && !folders.contains(&folder)
            {
                folders.push(folder);
            }
        }
        Ok(DrafterTarget {
            material,
            repository,
            primary,
            folders,
            snapshot,
            lineage,
            catalog: None,
            reach,
        })
    }

    // Drafter enumeration --------------------------------------------------------------------

    /// The drafters offered for a target, in automatic-choice order (see `in_choice_order`), each
    /// with the engine's verdict, then the explicit `other` reference when one is given. With
    /// `stop_at_first_automatic`, sources after the first one automatic choice may take are not
    /// assessed.
    ///
    /// A drafter the engine binds is automatic unless the model fits without a drafter and not
    /// with it: a drafter never makes a model that fits not fit, so the next candidate, or none, is
    /// chosen instead. "Without a drafter" is the model served plain, never with its built-in heads.
    /// When the model doesn't fit even without one, fit doesn't decide. A catalog model, and a
    /// catalog model's exact target, are offered only the catalog drafter, under the same rule.
    async fn enumerate_drafters(
        &self,
        target: &DrafterTarget,
        other: Option<&str>,
        stop_at_first_automatic: bool,
    ) -> Result<Vec<DrafterCandidate>, InventoryError> {
        self.drafter_list(
            target,
            other,
            stop_at_first_automatic,
            &InspectionProgress::silent(),
        )
        .await
        .map(|list| list.candidates)
    }

    /// The drafter list of `enumerate_drafters`, with the listed reference a different drafter
    /// resolves to: a candidate already listed that is the same file (the catalog drafter's
    /// package, or the same repository, commit and path), else its own Other row.
    /// `progress` counts the drafter checks: the target's own built-in heads, each drafter next to
    /// it and each catalog drafter, as each is about to run.
    async fn drafter_list(
        &self,
        target: &DrafterTarget,
        other: Option<&str>,
        stop_at_first_automatic: bool,
        progress: &InspectionProgress,
    ) -> Result<DrafterList, InventoryError> {
        let bare = self
            .assess(&target.material, None, ModelMethod::Plain)
            .await?;
        if matches!(bare.profile, ModelAssessment::Unsupported { .. }) {
            return Ok(DrafterList::default());
        }
        let fit_decides = fits(&bare.profile);
        // The target as the engine takes it on its own: with its built-in heads when it has any.
        progress.drafters_planned(1);
        let own = self
            .assess(&target.material, None, ModelMethod::Auto)
            .await;
        progress.drafter_checked();
        let own = own?;
        let regardless_of_fit = |mut candidate: DrafterCandidate| {
            candidate.automatic = known_to_work(&candidate) && is_bound(&candidate.verdict);
            candidate
        };
        let exact = match (&target.repository, &target.primary) {
            (Some((repository, commit)), Some(primary)) => {
                Some((repository.as_str(), commit.as_str(), primary.as_path()))
            }
            _ => None,
        };
        let matches = match &target.catalog {
            Some(model_id) => self
                .lineage
                .variant(model_id)
                .map(|matched| (matched, DrafterLineage::Exact))
                .into_iter()
                .collect(),
            None => {
                self.catalog_matches(
                    exact,
                    &target.lineage,
                    own.fingerprint.as_ref(),
                    target.reach,
                )
                .await
            }
        };
        let matched = matches.first().cloned();
        let decide = |candidate: DrafterCandidate| {
            if fit_decides {
                candidate
            } else {
                regardless_of_fit(candidate)
            }
        };
        let embedded_catalog = matched.as_ref().is_some_and(|(matched, _)| {
            matches!(
                &matched.model.configuration.bundle,
                ServableModelBundle::SpeculativeDecoding {
                    draft_source: SpeculativeDraftSource::Embedded,
                    ..
                }
            )
        });
        // The target's built-in heads, when it has any and the catalog drafter isn't them.
        let built_in = || {
            let verdict = verdict_of(&own.profile);
            (!embedded_catalog && !matches!(verdict, DrafterVerdict::None)).then(|| {
                decide(DrafterCandidate {
                    category: DrafterCategory::BuiltIn,
                    automatic: is_bound(&verdict) && fits(&own.profile),
                    reference: DrafterReference::Embedded,
                    name: embedded_drafter_name(bound_method(&verdict).as_ref()),
                    label: "Built into this model".to_owned(),
                    size_bytes: 0,
                    lineage: DrafterLineage::Exact,
                    method: bound_method(&verdict),
                    verdict,
                    on_this_computer: true,
                })
            })
        };
        // A catalog model, and a file that is exactly a catalog model's target, use only that
        // catalog model's drafter, unless it would keep a model that fits without it from fitting,
        // and their built-in heads, which serve while that drafter is not on this computer.
        let exact_catalog = matches!(&matched, Some((_, DrafterLineage::Exact)));
        if target.catalog.is_some() || exact_catalog {
            let Some((matched, lineage)) = &matched else {
                return Ok(DrafterList::default());
            };
            let catalog = self
                .catalog_drafter(&target, matched, *lineage, progress)
                .await
                .map(decide);
            return Ok(DrafterList {
                candidates: in_choice_order(catalog.into_iter().chain(built_in()).collect()),
                entered: None,
            });
        }
        let done = |candidates: &[DrafterCandidate]| {
            stop_at_first_automatic && candidates.iter().any(|candidate| candidate.automatic)
        };
        let catalog_file = matched
            .as_ref()
            .and_then(|(matched, _)| catalog_draft_key(&matched.model));
        let mut candidates = self
            .next_to_model(&target, catalog_file.as_deref(), progress)
            .await
            .into_iter()
            .map(decide)
            .collect::<Vec<_>>();
        // The best catalog match's drafter; when the engine rejects it, the fingerprint-matched
        // catalog model's drafter is offered too.
        for (matched, lineage) in &matches {
            if done(&candidates) {
                break;
            }
            let Some(candidate) = self
                .catalog_drafter(&target, matched, *lineage, progress)
                .await
            else {
                continue;
            };
            let bound = is_bound(&candidate.verdict);
            candidates.push(decide(candidate));
            if bound {
                break;
            }
        }
        if !done(&candidates) {
            candidates.extend(built_in());
        }
        let mut candidates = in_choice_order(candidates);
        let mut entered = None;
        if let Some(other) = other {
            let mut related = target.lineage.clone();
            if let Some((matched, _)) = &matched {
                related.extend(self.lineage.lineage_of(&matched.model_id).await);
            }
            let candidate = self.other_drafter(&target, &related, other).await?;
            let key = reference_key(&candidate.reference);
            match candidates
                .iter()
                .find(|listed| self.drafter_file_key(&listed.reference) == key)
            {
                Some(listed) => entered = Some(listed.reference.clone()),
                None => {
                    entered = Some(candidate.reference.clone());
                    candidates.push(candidate);
                }
            }
        }
        Ok(DrafterList {
            candidates,
            entered,
        })
    }

    /// The file a listed drafter is, as a Hugging Face reference key: a catalog drafter's package
    /// file, else its own reference.
    fn drafter_file_key(&self, reference: &DrafterReference) -> String {
        match reference {
            DrafterReference::Catalog { model_id } => self
                .lineage
                .variant(model_id)
                .and_then(|matched| catalog_draft_key(&matched.model)),
            _ => None,
        }
        .unwrap_or_else(|| reference_key(reference))
    }

    async fn catalog_drafter(
        &self,
        target: &DrafterTarget,
        matched: &CatalogLineageMatch,
        lineage: DrafterLineage,
        progress: &InspectionProgress,
    ) -> Option<DrafterCandidate> {
        let ServableModelBundle::SpeculativeDecoding {
            draft_source,
            method,
            ..
        } = &matched.model.configuration.bundle
        else {
            return None;
        };
        progress.drafters_planned(1);
        let candidate = self
            .catalog_drafter_candidate(target, matched, lineage, draft_source, method)
            .await;
        progress.drafter_checked();
        candidate
    }

    async fn catalog_drafter_candidate(
        &self,
        target: &DrafterTarget,
        matched: &CatalogLineageMatch,
        lineage: DrafterLineage,
        draft_source: &SpeculativeDraftSource,
        method: &SpeculativeMethod,
    ) -> Option<DrafterCandidate> {
        let (drafter, size_bytes, on_this_computer) = match draft_source {
            SpeculativeDraftSource::Embedded => (None, 0, true),
            SpeculativeDraftSource::Separate { draft } => {
                let resolved = self
                    .assessor
                    .resolve_material(matched.model.configuration.bundle.clone())
                    .await
                    .map_err(
                        |error| tracing::warn!(%error, "catalog drafter material is unavailable"),
                    )
                    .ok()?;
                let material = Self::resolved_drafter(resolved).ok().flatten()?;
                let reference = DrafterReference::Catalog {
                    model_id: matched.model_id.clone(),
                };
                let content = match target.reach {
                    Reach::Local => self.enrichment.drafter_content(&reference),
                    Reach::Hub => self.learn_drafter_content(&reference).await,
                };
                let on_this_computer = self.models().has_installed_package(&draft.id)
                    || content
                        .or_else(|| DrafterContent::of_package(draft))
                        .is_some_and(|content| self.models().has_installed_content(&content));
                (Some(material), package_bytes(draft), on_this_computer)
            }
        };
        let assessed = self
            .assess(&target.material, drafter.as_ref(), engine_method(method))
            .await
            .map_err(|error| tracing::warn!(%error, "catalog drafter assessment failed"))
            .ok()?;
        let verdict = verdict_of(&assessed.profile);
        let trained_for = format!("Trained for {}", matched.model.display_name);
        let mut candidate = DrafterCandidate {
            category: DrafterCategory::CatalogDrafter,
            automatic: false,
            reference: DrafterReference::Catalog {
                model_id: matched.model_id.clone(),
            },
            name: catalog_drafter_name(method, &matched.model.display_name),
            label: match lineage {
                DrafterLineage::Fingerprint => format!("Untested · {trained_for}"),
                _ => trained_for,
            },
            size_bytes,
            lineage,
            method: bound_method(&verdict).or(Some(method.clone())),
            verdict,
            on_this_computer,
        };
        candidate.automatic =
            known_to_work(&candidate) && is_bound(&candidate.verdict) && fits(&assessed.profile);
        Some(candidate)
    }

    /// The separate drafters in the target's own repository snapshot, or its own folder when it
    /// has no repository snapshot: GGUFs whose header declares a drafter role. The catalog
    /// drafter's own file is left to the catalog candidate.
    async fn next_to_model(
        &self,
        target: &DrafterTarget,
        catalog_file: Option<&str>,
        progress: &InspectionProgress,
    ) -> Vec<DrafterCandidate> {
        if let Some(snapshot) = &target.snapshot {
            let Ok(classified) = self
                .classify_hugging_face(snapshot, &InspectionProgress::silent())
                .await
            else {
                return Vec::new();
            };
            let pending = classified
                .into_iter()
                .filter(|group| matches!(group.role, CandidateFileRole::Drafter { .. }))
                .filter(|group| group.group.files.len() == 1)
                .map(|group| group.group.primary().path.clone())
                .filter(|path| {
                    let key = reference_key(&DrafterReference::HuggingFace {
                        repository: hugging_face_id(&snapshot.repository),
                        commit: snapshot.commit.clone(),
                        path: path_string(path),
                    });
                    catalog_file != Some(key.as_str())
                })
                .map(|path| self.hugging_face_drafter_candidate(target, snapshot, path))
                .collect::<Vec<_>>();
            return collect_candidates(pending, progress).await;
        }
        let mut pending = Vec::new();
        for folder in &target.folders {
            let (_, classified) = self.scan_local(&folder.path, DRAFTER_SCAN_DEPTH).await;
            pending.extend(
                classified
                    .into_iter()
                    .filter(|group| matches!(group.role, CandidateFileRole::Drafter { .. }))
                    .filter(|group| group.group.files.len() == 1)
                    .filter(|group| folder.admits(&group.group.primary().path))
                    .map(|group| {
                        self.local_drafter_candidate(
                            target,
                            folder.path.join(&group.group.primary().path),
                            group.group.size_bytes(),
                        )
                    }),
            );
        }
        collect_candidates(pending, progress).await
    }

    /// A drafter file in the target's Hugging Face repository, with the engine's verdict.
    async fn hugging_face_drafter_candidate(
        &self,
        target: &DrafterTarget,
        snapshot: &HuggingFaceRepositorySnapshot,
        path: PathBuf,
    ) -> Option<DrafterCandidate> {
        let file = snapshot_file(snapshot, &path).ok()?;
        let material = self.hugging_face_drafter(snapshot, &path).await.ok()?;
        let (verdict, fits) = self.drafter_verdict(target, &material).await?;
        let reference = DrafterReference::HuggingFace {
            repository: hugging_face_id(&snapshot.repository),
            commit: snapshot.commit.clone(),
            path: path_string(&path),
        };
        let content = hugging_face_drafter_content(file, &material);
        self.remember_drafter_content(&reference, &content);
        Some(DrafterCandidate {
            category: DrafterCategory::NextToModel,
            automatic: is_bound(&verdict) && fits,
            reference,
            // The whole path: a repository may ship same-named drafters in several subfolders.
            name: format!("{}/{}", snapshot.repository, path_string(&path)),
            label: "In this repository".to_owned(),
            size_bytes: file.size_bytes,
            lineage: DrafterLineage::Lineage,
            method: bound_method(&verdict),
            verdict,
            on_this_computer: self.models().has_installed_content(&content),
        })
    }

    /// A drafter file in the target's folder, with the engine's verdict.
    async fn local_drafter_candidate(
        &self,
        target: &DrafterTarget,
        path: PathBuf,
        size_bytes: u64,
    ) -> Option<DrafterCandidate> {
        let material = self.local_drafter(path.clone()).ok()?;
        let (verdict, fits) = self.drafter_verdict(target, &material).await?;
        Some(DrafterCandidate {
            category: DrafterCategory::NextToModel,
            automatic: is_bound(&verdict) && fits,
            name: file_name(&path),
            reference: DrafterReference::Local { path },
            label: "In this folder".to_owned(),
            size_bytes,
            lineage: DrafterLineage::Lineage,
            method: bound_method(&verdict),
            verdict,
            on_this_computer: true,
        })
    }

    /// An explicit drafter: a Hugging Face reference (`owner/repo`, `owner/repo:path`,
    /// `owner/repo/path`, `owner/repo@revision:path`, any of them after `hf:`, or a model URL) or
    /// a local file.
    async fn other_drafter(
        &self,
        target: &DrafterTarget,
        related: &BTreeSet<String>,
        reference: &str,
    ) -> Result<DrafterCandidate, InventoryError> {
        let reference = reference.trim();
        let local = Path::new(reference);
        if local.is_absolute() {
            let material = self.local_drafter(local.to_path_buf())?;
            let lineage = lineage_kind(
                &self.header_lineage(&material.path, &material.headers),
                related,
            );
            let (verdict, _) = self
                .drafter_verdict(target, &material)
                .await
                .ok_or_else(|| InventoryError::Internal("drafter assessment failed".to_owned()))?;
            return Ok(DrafterCandidate {
                category: DrafterCategory::Other,
                automatic: false,
                reference: DrafterReference::Local {
                    path: local.to_path_buf(),
                },
                name: file_name(local),
                label: other_drafter_label(lineage),
                size_bytes: std::fs::metadata(local)
                    .map(|metadata| metadata.len())
                    .unwrap_or(0),
                lineage,
                method: bound_method(&verdict),
                verdict,
                on_this_computer: true,
            });
        }
        let (repository, revision, path) = parse_hugging_face_reference(reference)?;
        let snapshot = self.snapshot(&repository, &revision).await?;
        let path = match path {
            Some(path) => PathBuf::from(path),
            None => {
                let classified = self
                    .classify_hugging_face(&snapshot, &InspectionProgress::silent())
                    .await?;
                classified
                    .into_iter()
                    .filter(|group| matches!(group.role, CandidateFileRole::Drafter { .. }))
                    .filter(|group| group.group.files.len() == 1)
                    .min_by_key(|group| group.group.size_bytes())
                    .map(|group| group.group.primary().path.clone())
                    .ok_or_else(|| {
                        InventoryError::InvalidRequest(format!("{repository} has no drafter GGUF"))
                    })?
            }
        };
        let file = snapshot_file(&snapshot, &path)?;
        let material = self.hugging_face_drafter(&snapshot, &path).await?;
        let mut lineage = self
            .lineage
            .hub()
            .lineage(&snapshot.repository, Some(snapshot.base_models.clone()))
            .await;
        lineage.extend(self.header_lineage(&material.path, &material.headers));
        let lineage = lineage_kind(&lineage, related);
        let (verdict, _) = self
            .drafter_verdict(target, &material)
            .await
            .ok_or_else(|| InventoryError::Internal("drafter assessment failed".to_owned()))?;
        let reference = DrafterReference::HuggingFace {
            repository: hugging_face_id(&snapshot.repository),
            commit: snapshot.commit.clone(),
            path: path_string(&path),
        };
        let content = hugging_face_drafter_content(file, &material);
        self.remember_drafter_content(&reference, &content);
        Ok(DrafterCandidate {
            category: DrafterCategory::Other,
            automatic: false,
            reference,
            name: format!("{}/{}", snapshot.repository, path_string(&path)),
            label: other_drafter_label(lineage),
            size_bytes: file.size_bytes,
            lineage,
            method: bound_method(&verdict),
            verdict,
            on_this_computer: self.models().has_installed_content(&content),
        })
    }

    /// The engine's verdict on a drafter for the target, and whether the model fits with it.
    async fn drafter_verdict(
        &self,
        target: &DrafterTarget,
        drafter: &DrafterMaterial,
    ) -> Option<(DrafterVerdict, bool)> {
        self.assess(&target.material, Some(drafter), ModelMethod::Auto)
            .await
            .map(|assessed| (verdict_of(&assessed.profile), fits(&assessed.profile)))
            .map_err(|error| tracing::warn!(%error, "drafter assessment failed"))
            .ok()
    }

    // Candidate assessment and bundles --------------------------------------------------------

    async fn candidate_drafter_material(
        &self,
        drafter: &CandidateDrafter,
    ) -> Result<Option<DrafterMaterial>, InventoryError> {
        match &drafter.reference {
            DrafterReference::Embedded => Ok(None),
            DrafterReference::Catalog { model_id } => {
                let matched = self
                    .lineage
                    .variant(model_id)
                    .ok_or_else(|| InventoryError::NotFound(model_id.as_str().to_owned()))?;
                let resolved = self
                    .assessor
                    .resolve_material(matched.model.configuration.bundle.clone())
                    .await?;
                Self::resolved_drafter(resolved)
            }
            DrafterReference::HuggingFace {
                repository,
                commit,
                path,
            } => {
                let snapshot = self.snapshot(repository.as_str(), commit).await?;
                Ok(Some(
                    self.hugging_face_drafter(&snapshot, Path::new(path))
                        .await?,
                ))
            }
            DrafterReference::Local { path } => Ok(Some(self.local_drafter(path.clone())?)),
        }
    }

    async fn hugging_face_package(
        &self,
        snapshot: &HuggingFaceRepositorySnapshot,
        primary: &Path,
        projector: Option<&Path>,
    ) -> Result<ModelPackage, InventoryError> {
        let source = ModelPreviewSource {
            repository: snapshot.repository.clone(),
            revision: snapshot.commit.clone(),
            primary_gguf: primary.to_path_buf(),
            additional_components: projector
                .map(|projector| ModelPreviewComponentSource {
                    path: projector.to_path_buf(),
                    role: ModelPreviewComponentRole::Projector,
                })
                .into_iter()
                .collect(),
        };
        Ok(self
            .models()
            .hugging_face_package(&source, snapshot)
            .await?
            .0)
    }

    /// The downloadable package of a drafter, with the method its bundle declares.
    async fn drafter_package(
        &self,
        drafter: &CandidateDrafter,
    ) -> Result<Option<(SpeculativeDraftSource, SpeculativeMethod)>, InventoryError> {
        match &drafter.reference {
            DrafterReference::Embedded => Ok(Some((
                SpeculativeDraftSource::Embedded,
                SpeculativeMethod::Mtp,
            ))),
            DrafterReference::Catalog { model_id } => {
                let matched = self
                    .lineage
                    .variant(model_id)
                    .ok_or_else(|| InventoryError::NotFound(model_id.as_str().to_owned()))?;
                match matched.model.configuration.bundle {
                    ServableModelBundle::SpeculativeDecoding {
                        draft_source,
                        method,
                        ..
                    } => Ok(Some((draft_source, method))),
                    ServableModelBundle::Standalone { .. } => Err(InventoryError::InvalidRequest(
                        format!("{} has no catalog drafter", model_id.as_str()),
                    )),
                }
            }
            DrafterReference::HuggingFace {
                repository,
                commit,
                path,
            } => {
                let snapshot = self.snapshot(repository.as_str(), commit).await?;
                let draft = self
                    .hugging_face_package(&snapshot, Path::new(path), None)
                    .await?;
                Ok(Some((
                    SpeculativeDraftSource::Separate { draft },
                    drafter.method.clone(),
                )))
            }
            DrafterReference::Local { .. } => Ok(None),
        }
    }
}

impl CatalogFingerprints for ManagedModelCandidates {
    fn catalog_fingerprints(&self) -> BoxFuture<'_, Vec<(ModelId, ModelFingerprint)>> {
        Box::pin(async move { self.catalog_facts().await.fingerprints.clone() })
    }
}

impl ModelCandidates for ManagedModelCandidates {
    fn inspect_with_progress(
        &self,
        request: CandidateInspectionRequest,
        progress: CandidateInspectionProgressSink,
    ) -> BoxFuture<'_, Result<CandidateInspection, InventoryError>> {
        Box::pin(async move {
            let progress = InspectionProgress::new(progress);
            match request {
                CandidateInspectionRequest::HuggingFace {
                    repository,
                    revision,
                } => {
                    self.inspect_hugging_face(repository, revision, &progress)
                        .await
                }
                CandidateInspectionRequest::Local { path } => {
                    self.inspect_local(path, &progress).await
                }
            }
        })
    }

    fn drafters(
        &self,
        request: DrafterCandidatesRequest,
    ) -> BoxFuture<'_, Result<DrafterCandidatesResponse, InventoryError>> {
        Box::pin(async move {
            let (target, other) = match request {
                DrafterCandidatesRequest::Candidate { candidate, other } => {
                    (self.candidate_target(&candidate).await?, other)
                }
                DrafterCandidatesRequest::Model { model_id, other } => {
                    (self.model_target(&model_id, Reach::Hub).await?, other)
                }
            };
            let list = self
                .drafter_list(&target, other.as_deref(), false, &InspectionProgress::silent())
                .await?;
            Ok(DrafterCandidatesResponse {
                candidates: list.candidates,
                entered: list.entered,
            })
        })
    }

    fn assess_candidate(
        &self,
        candidate: ModelCandidate,
    ) -> BoxFuture<'_, Result<ModelAssessment, InventoryError>> {
        Box::pin(async move {
            let target = self.candidate_target(&candidate).await?;
            let (drafter, method) = match &candidate.drafter {
                Some(drafter) => (
                    self.candidate_drafter_material(drafter).await?,
                    engine_method(&drafter.method),
                ),
                None => (None, ModelMethod::Plain),
            };
            Ok(self
                .assess(&target.material, drafter.as_ref(), method)
                .await?
                .profile)
        })
    }

    fn bundle(
        &self,
        candidate: ModelCandidate,
    ) -> BoxFuture<'_, Result<ServableModelBundle, InventoryError>> {
        Box::pin(async move {
            let drafter = match &candidate.drafter {
                Some(drafter) => self.drafter_package(drafter).await?,
                None => None,
            };
            match &candidate.source {
                CandidateSource::HuggingFace { repository, commit } => {
                    let snapshot = self.snapshot(repository.as_str(), commit).await?;
                    let primary = candidate.target.first().map(PathBuf::from).ok_or_else(|| {
                        InventoryError::InvalidRequest("a candidate names its target".to_owned())
                    })?;
                    let projector = candidate.projector.as_ref().map(|file| PathBuf::from(&file.path));
                    let target = self
                        .hugging_face_package(&snapshot, &primary, projector.as_deref())
                        .await?;
                    let expected = candidate.target.iter().map(PathBuf::from).collect::<BTreeSet<_>>();
                    let resolved = target
                        .files
                        .iter()
                        .filter(|file| file.role == ModelFileRole::Weights)
                        .map(|file| file.path.clone())
                        .collect::<BTreeSet<_>>();
                    if expected != resolved {
                        return Err(InventoryError::InvalidRequest(
                            "the candidate's target files are not one complete GGUF".to_owned(),
                        ));
                    }
                    Ok(match drafter {
                        None => ServableModelBundle::Standalone { package: target },
                        Some((draft_source, method)) => ServableModelBundle::SpeculativeDecoding {
                            target,
                            draft_source,
                            method,
                        },
                    })
                }
                CandidateSource::Local { .. } => match drafter {
                    Some((SpeculativeDraftSource::Separate { draft }, _)) => {
                        Ok(ServableModelBundle::Standalone { package: draft })
                    }
                    _ => Err(InventoryError::InvalidRequest(
                        "a local model is referenced in place; only a separate Hugging Face or catalog drafter downloads"
                            .to_owned(),
                    )),
                },
            }
        })
    }
}

// Helpers ------------------------------------------------------------------------------------

/// Run drafter verdicts with bounded concurrency, keeping their order and dropping the ones that
/// could not be assessed.
async fn collect_candidates(
    pending: Vec<impl std::future::Future<Output = Option<DrafterCandidate>>>,
    progress: &InspectionProgress,
) -> Vec<DrafterCandidate> {
    progress.drafters_planned(pending.len());
    futures_util::stream::iter(pending.into_iter().map(|check| async {
        let candidate = check.await;
        progress.drafter_checked();
        candidate
    }))
    .buffered(assessment_concurrency())
        .filter_map(|candidate| async move { candidate })
        .collect()
        .await
}

/// Runs the versions' checks with bounded concurrency, keeping their order and reporting each
/// version's verdict as soon as its own check finishes.
async fn checked_versions(
    pending: Vec<impl std::future::Future<Output = Option<CachedModelAssessment>>>,
    progress: &InspectionProgress,
) -> Vec<Option<CachedModelAssessment>> {
    futures_util::stream::iter(pending.into_iter().enumerate().map(|(index, check)| async move {
        let assessment = check.await;
        progress.version_checked(
            index,
            assessment.as_ref().map(|assessment| assessment.profile.clone()),
        );
        assessment
    }))
    .buffered(assessment_concurrency())
    .collect()
    .await
}

/// The versions an inspection returns, by label and size, before any is checked.
fn version_rows(targets: &[&ClassifiedGroup]) -> Vec<(String, u64)> {
    let labels = distinct_labels(
        &targets
            .iter()
            .map(|group| (&group.group, group.quantization.as_deref(), group.embedded_mtp))
            .collect::<Vec<_>>(),
    );
    labels
        .into_iter()
        .zip(targets)
        .map(|(label, group)| (label, group.group.size_bytes()))
        .collect()
}

/// Whether a file's name calls it a projector, which only orders and hides it before its header
/// is read.
fn is_projector_name(path: &Path) -> bool {
    path_string(path).to_lowercase().contains("mmproj")
}

fn assessment_concurrency() -> usize {
    std::thread::available_parallelism()
        .map_or(1, |cores| cores.get().min(MAX_ASSESSMENT_CONCURRENCY))
}

fn is_commit(revision: &str) -> bool {
    (40..=64).contains(&revision.len())
        && revision
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn engine_method(method: &SpeculativeMethod) -> ModelMethod {
    match method {
        SpeculativeMethod::Mtp => ModelMethod::Mtp,
        SpeculativeMethod::DFlash => ModelMethod::DFlash,
        SpeculativeMethod::DFlash2 => ModelMethod::DFlash2,
        SpeculativeMethod::DSpark => ModelMethod::DSpark,
    }
}

fn verdict_of(assessment: &ModelAssessment) -> DrafterVerdict {
    match assessment {
        ModelAssessment::Fits { drafter, .. } | ModelAssessment::DoesNotFit { drafter, .. } => {
            drafter.clone()
        }
        ModelAssessment::Unsupported {
            reasons, detail, ..
        } => DrafterVerdict::Rejected {
            reasons: reasons.clone(),
            detail: detail.clone(),
        },
    }
}

fn bound_method(verdict: &DrafterVerdict) -> Option<SpeculativeMethod> {
    match verdict {
        DrafterVerdict::Bound { method, .. } => Some(method.clone()),
        DrafterVerdict::None | DrafterVerdict::Rejected { .. } => None,
    }
}

fn lineage_kind(drafter: &BTreeSet<String>, related: &BTreeSet<String>) -> DrafterLineage {
    if drafter.is_disjoint(related) {
        DrafterLineage::None
    } else {
        DrafterLineage::Lineage
    }
}

fn is_bound(verdict: &DrafterVerdict) -> bool {
    matches!(verdict, DrafterVerdict::Bound { .. })
}

/// Whether automatic choice may take a candidate the engine binds: its source shows it was made
/// for this target. A drafter the user entered never is, nor is a catalog drafter matched by shape
/// alone: one architecture and size share every fingerprinted dimension and often the token table
/// across model generations, and a drafter trained on another generation's hidden states binds but
/// slows decoding.
fn known_to_work(candidate: &DrafterCandidate) -> bool {
    candidate.category != DrafterCategory::Other
        && candidate.lineage != DrafterLineage::Fingerprint
}

fn fits(assessment: &ModelAssessment) -> bool {
    matches!(assessment, ModelAssessment::Fits { .. })
}

/// Candidates in automatic-choice order: by category (`DrafterCategory`'s order), then by method
/// (DFlash2, DSpark, DFlash, MTP; one with no known method last), then the smaller download, and
/// otherwise in enumeration order. The first bound candidate in this order is the recommendation.
fn in_choice_order(mut candidates: Vec<DrafterCandidate>) -> Vec<DrafterCandidate> {
    candidates.sort_by_key(|candidate| {
        (
            candidate.category,
            candidate.method.as_ref().map_or(usize::MAX, method_rank),
            download_bytes(candidate),
        )
    });
    candidates
}

/// The speculation order of the model radar: DFlash2, DSpark, DFlash, then MTP.
fn method_rank(method: &SpeculativeMethod) -> usize {
    match method {
        SpeculativeMethod::DFlash2 => 0,
        SpeculativeMethod::DSpark => 1,
        SpeculativeMethod::DFlash => 2,
        SpeculativeMethod::Mtp => 3,
    }
}

fn download_bytes(candidate: &DrafterCandidate) -> u64 {
    if candidate.on_this_computer {
        0
    } else {
        candidate.size_bytes
    }
}

/// Each file of a version with its SHA-256 as the snapshot lists it; `None` when any is unknown.
fn snapshot_weights(
    snapshot: &HuggingFaceRepositorySnapshot,
    group: &GgufGroup,
) -> Option<BTreeSet<(PathBuf, String)>> {
    group
        .files
        .iter()
        .map(
            |file| match &snapshot_file(snapshot, &file.path).ok()?.content {
                ContentIdentity::Sha256 { value } => Some((file.path.clone(), value.clone())),
                _ => None,
            },
        )
        .collect()
}

/// How a catalog model serves without its catalog drafter, from its drafter candidates.
fn catalog_drafter_fit(candidates: &[DrafterCandidate]) -> CatalogDrafterFit {
    CatalogDrafterFit {
        withheld: candidates.iter().any(|candidate| {
            candidate.category == DrafterCategory::CatalogDrafter
                && is_bound(&candidate.verdict)
                && !candidate.automatic
        }),
        built_in_heads: candidates
            .iter()
            .any(|candidate| candidate.category == DrafterCategory::BuiltIn && candidate.automatic),
    }
}

/// The candidates automatic choice may take, in its order (see `DrafterCandidate::automatic`).
fn automatic(candidates: Vec<DrafterCandidate>) -> Vec<DrafterCandidate> {
    candidates
        .into_iter()
        .filter(|candidate| candidate.automatic)
        .collect()
}

fn recommended(candidates: Vec<DrafterCandidate>) -> Option<DrafterCandidate> {
    automatic(candidates).into_iter().next()
}

fn other_drafter_label(lineage: DrafterLineage) -> String {
    match lineage {
        DrafterLineage::None => "Untested".to_owned(),
        DrafterLineage::Exact | DrafterLineage::Lineage | DrafterLineage::Fingerprint => {
            "Entered by you".to_owned()
        }
    }
}

/// A target's drafter candidates, and the listed reference its different drafter resolves to.
#[derive(Default)]
struct DrafterList {
    candidates: Vec<DrafterCandidate>,
    entered: Option<DrafterReference>,
}

fn reference_key(reference: &DrafterReference) -> String {
    match reference {
        DrafterReference::HuggingFace {
            repository,
            commit,
            path,
        } => format!("hf:{}@{commit}/{path}", repository.as_str().to_lowercase()),
        DrafterReference::Local { path } => format!("local:{}", path.display()),
        DrafterReference::Catalog { model_id } => format!("catalog:{}", model_id.as_str()),
        DrafterReference::Embedded => "embedded".to_owned(),
    }
}

/// The catalog drafter's file as a Hugging Face reference.
fn catalog_draft_reference(
    model: &magnitude_service_contracts::models::RecommendableModel,
) -> Option<DrafterReference> {
    let ServableModelBundle::SpeculativeDecoding {
        draft_source: SpeculativeDraftSource::Separate { draft },
        ..
    } = &model.configuration.bundle
    else {
        return None;
    };
    let ModelPackageSource::HuggingFace {
        repository,
        revision,
    } = &draft.source
    else {
        return None;
    };
    let path = draft
        .files
        .iter()
        .filter(|file| matches!(file.role, ModelFileRole::Weights | ModelFileRole::Draft))
        .map(|file| file.path.clone())
        .min()?;
    Some(DrafterReference::HuggingFace {
        repository: hugging_face_id(repository),
        commit: revision.clone(),
        path: path_string(&path),
    })
}

/// The catalog drafter's file as a Hugging Face reference key, so the same file found elsewhere
/// is not listed twice.
fn catalog_draft_key(
    model: &magnitude_service_contracts::models::RecommendableModel,
) -> Option<String> {
    catalog_draft_reference(model).map(|reference| reference_key(&reference))
}

/// A single-file Hugging Face drafter's bytes: the header digest its material was read with, its
/// size, and its Hub SHA-256 when the Hub gives one.
fn hugging_face_drafter_content(
    file: &HuggingFaceRepositoryFile,
    material: &DrafterMaterial,
) -> DrafterContent {
    DrafterContent {
        header_digest: Some(material.identity.clone()),
        sha256: sha256_of(file).map(str::to_owned),
        size_bytes: file.size_bytes,
    }
}

/// The name Magnitude gives the drafter it ships for a catalog model, such as "Magnitude DFlash
/// for Qwen3.5 4B".
fn catalog_drafter_name(method: &SpeculativeMethod, model: &str) -> String {
    format!("Magnitude {} for {model}", method_name(method))
}

fn embedded_drafter_name(method: Option<&SpeculativeMethod>) -> String {
    match method {
        Some(SpeculativeMethod::Mtp) | None => "MTP heads".to_owned(),
        Some(method) => format!("{} heads", method_name(method)),
    }
}

fn method_name(method: &SpeculativeMethod) -> &'static str {
    match method {
        SpeculativeMethod::Mtp => "MTP",
        SpeculativeMethod::DFlash => "DFlash",
        SpeculativeMethod::DFlash2 => "DFlash2",
        SpeculativeMethod::DSpark => "DSpark",
    }
}

fn file_name(path: &Path) -> String {
    path.file_name().map_or_else(
        || path_string(path),
        |name| name.to_string_lossy().into_owned(),
    )
}

/// The identity of exact header material: each file's path and header digest, in path order.
fn header_identity(headers: impl IntoIterator<Item = (String, String)>) -> String {
    let mut headers = headers.into_iter().collect::<Vec<_>>();
    headers.sort();
    headers
        .into_iter()
        .map(|(path, digest)| format!("{path}={digest}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn hugging_face_id(repository: &str) -> HuggingFaceRepositoryId {
    HuggingFaceRepositoryId::new(repository.to_owned())
        .unwrap_or_else(|_| HuggingFaceRepositoryId::new("invalid/repository").expect("valid id"))
}

fn path_string(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn package_bytes(package: &ModelPackage) -> u64 {
    package.files.iter().map(|file| file.size_bytes).sum()
}

fn bundle_target(bundle: &ServableModelBundle) -> &ModelPackage {
    match bundle {
        ServableModelBundle::Standalone { package } => package,
        ServableModelBundle::SpeculativeDecoding { target, .. } => target,
    }
}

/// The resolved first GGUF (a split file's first shard) and projector.
fn resolved_paths(model: &ResolvedModel) -> Result<(PathBuf, Option<PathBuf>), InventoryError> {
    let target = model
        .components
        .iter()
        .filter(|component| {
            matches!(
                component.role,
                ComponentRole::Weights | ComponentRole::Shard | ComponentRole::Draft
            )
        })
        .min_by_key(|component| (component.shard_index.unwrap_or(0), component.path.clone()))
        .map(|component| component.path.clone())
        .ok_or_else(|| InventoryError::NotReady("model has no runnable weights".to_owned()))?;
    let projector = model
        .components
        .iter()
        .find(|component| component.role == ComponentRole::Projector)
        .map(|component| component.path.clone());
    Ok((target, projector))
}

fn snapshot_file<'a>(
    snapshot: &'a HuggingFaceRepositorySnapshot,
    path: &Path,
) -> Result<&'a HuggingFaceRepositoryFile, InventoryError> {
    snapshot
        .gguf_files
        .iter()
        .find(|file| file.path == path)
        .ok_or_else(|| {
            InventoryError::NotFound(format!(
                "{} is not in {} at {}",
                path.display(),
                snapshot.repository,
                snapshot.commit
            ))
        })
}

fn sha256_of(file: &HuggingFaceRepositoryFile) -> Option<&str> {
    match &file.content {
        magnitude_service_contracts::ContentIdentity::Sha256 { value } => Some(value),
        _ => None,
    }
}

/// GGUFs beneath `root`, relative to it, at most `depth` directories deep. Partial downloads and
/// hidden files are skipped.
fn classified_group(
    group: GgufGroup,
    header_digest: String,
    classification: magnitude_service_models::candidate_files::HeaderClassification,
) -> ClassifiedGroup {
    ClassifiedGroup {
        role: classification.role,
        quantization: classification.quantization,
        embedded_mtp: classification.embedded_mtp,
        header_digest,
        local_digest: None,
        group,
        unreadable: None,
    }
}

/// A file whose header is invalid, such as one with a tensor type the engine doesn't know: a
/// version of its repository that can't be read, never a reason to fail the inspection.
fn unreadable_group(group: GgufGroup, detail: String) -> ClassifiedGroup {
    ClassifiedGroup {
        role: CandidateFileRole::Target { architecture: None },
        header_digest: String::new(),
        local_digest: None,
        quantization: None,
        embedded_mtp: false,
        group,
        unreadable: Some(detail),
    }
}

/// A local folder's GGUF groups classified through the identity cache: each file is read at most
/// once, and an unchanged one not at all.
fn classify_local(
    store: &ManagedModelStore,
    root: &Path,
    files: &[CandidateGguf],
    architectures: &RecognizedArchitectures,
) -> Vec<ClassifiedGroup> {
    group_ggufs(files)
        .into_iter()
        .filter_map(|group| {
            let (gguf, classification) = store
                .classify_local_gguf(&root.join(&group.primary().path), architectures)
                .ok()?;
            Some(ClassifiedGroup {
                role: classification.role,
                quantization: classification.quantization,
                embedded_mtp: classification.embedded_mtp,
                header_digest: gguf.inspection.header_digest,
                local_digest: local_digest(
                    store,
                    group.files.iter().map(|file| root.join(&file.path)),
                ),
                group,
                unreadable: None,
            })
        })
        .collect()
}

/// The `local:` digest of a model's shards, first shard first, from the identity cache.
fn local_digest(
    store: &ManagedModelStore,
    shards: impl IntoIterator<Item = PathBuf>,
) -> Option<String> {
    let samples = shards
        .into_iter()
        .map(|shard| store.local_gguf(&shard).ok()?.inspection.content_sample)
        .collect::<Option<Vec<_>>>()?;
    Some(magnitude_service_models::gguf::local_model_digest(
        samples.iter().map(String::as_str),
    ))
}

/// The header digest of a file, from the identity cache.
fn header_digest(store: &ManagedModelStore, path: &Path) -> Option<String> {
    store
        .local_gguf(path)
        .ok()
        .map(|gguf| gguf.inspection.header_digest)
}

/// GGUF files under `root`, named by the paths the user sees: a symlinked file (a Hugging Face
/// cache snapshot entry, or any link) is listed under its own name and read through the link.
fn scan_ggufs(root: &Path, depth: usize) -> Vec<CandidateGguf> {
    let mut found = Vec::new();
    let mut pending = vec![(root.to_path_buf(), 0_usize)];
    while let Some((directory, level)) = pending.pop() {
        let Ok(entries) = std::fs::read_dir(&directory) else {
            continue;
        };
        for entry in entries.flatten() {
            if found.len() >= MAX_LOCAL_SCAN_FILES {
                return found;
            }
            let name = entry.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            if name.starts_with('.') {
                continue;
            }
            let Ok(file_type) = entry.file_type() else {
                continue;
            };
            let path = entry.path();
            if file_type.is_dir() && level + 1 < depth {
                pending.push((path, level + 1));
            } else if (file_type.is_file() || file_type.is_symlink())
                && name.to_ascii_lowercase().ends_with(".gguf")
                && let Ok(metadata) = std::fs::metadata(&path)
                && metadata.is_file()
                && let Ok(relative) = path.strip_prefix(root)
            {
                found.push(CandidateGguf {
                    path: relative.to_path_buf(),
                    size_bytes: metadata.len(),
                });
            }
        }
    }
    found.sort_by(|left, right| left.path.cmp(&right.path));
    found
}

/// The projector a vision target uses: the only one, or the smallest of several.
fn choose_projector(classified: &[ClassifiedGroup]) -> Option<&ClassifiedGroup> {
    classified
        .iter()
        .filter(|group| group.role == CandidateFileRole::Projector && group.group.files.len() == 1)
        .min_by_key(|group| (group.group.size_bytes(), group.group.primary().path.clone()))
}

/// The candidate one version of an inspection is: its shards in order and the inspection's
/// projector, with the drafter left to automatic choice.
fn version_candidate(
    source: &CandidateSource,
    group: &ClassifiedGroup,
    projector: Option<CandidateFile>,
) -> ModelCandidate {
    ModelCandidate {
        source: source.clone(),
        target: group
            .group
            .files
            .iter()
            .map(|file| path_string(&file.path))
            .collect(),
        projector,
        drafter: None,
    }
}

fn candidate_file(file: &CandidateGguf) -> CandidateFile {
    CandidateFile {
        path: path_string(&file.path),
        size_bytes: file.size_bytes,
    }
}

/// One target version with its verdict, its canonical identity, and the model already on this
/// computer with the same bytes.
struct AssessedVersion<'a> {
    group: &'a ClassifiedGroup,
    assessment: Option<CachedModelAssessment>,
    model_id: ModelId,
    catalog_model: Option<ModelId>,
    on_this_computer: Option<ModelId>,
}

fn hugging_face_model_id(repository: &str, group: &GgufGroup) -> Option<ModelId> {
    let repository = HuggingFaceRepositoryId::new(repository.to_owned()).ok()?;
    let selector = magnitude_service_contracts::models::HuggingFaceArtifactSelector::new(
        path_string(&group.primary().path),
    )
    .ok()?;
    Some(ModelId::hugging_face(&repository, &selector))
}

/// Where a model's "in this folder" drafters come from. An added model and a model whose Hub
/// repository snapshot was read list that repository's drafters instead; a model from a known
/// repository read offline is offered only that repository's flat files; any other model, its
/// own folder.
fn drafter_folder(
    origin: &CustomModelOrigin,
    repository: Option<&str>,
    snapshot_read: bool,
    parent: Option<&Path>,
) -> Option<DrafterFolder> {
    let path = parent?.to_path_buf();
    match (origin, repository) {
        (CustomModelOrigin::Added { .. }, _) => None,
        (_, Some(_)) if snapshot_read => None,
        (_, Some(repository)) => {
            let (owner, name) = repository.split_once('/')?;
            Some(DrafterFolder {
                path,
                prefix: Some(magnitude_service_models::llama_cpp_flat_prefix(owner, name)),
            })
        }
        (_, None) => Some(DrafterFolder::whole(path)),
    }
}

/// The cache key of an assessment: the environment (engine build, backend, device and its
/// bandwidth), the exact header material and the method. Material is identified by header digest,
/// so copies of the same file share one assessment.
fn assessment_evidence(
    environment: &magnitude_service_contracts::models::AssessmentEnvironmentId,
    target: &str,
    drafter: Option<&str>,
    method: &ModelMethod,
) -> Result<String, InventoryError> {
    serde_json::to_string(&(
        "candidate-assessment-v3",
        &environment.0,
        target,
        drafter,
        format!("{method:?}"),
    ))
    .map_err(|error| InventoryError::Internal(error.to_string()))
}

fn local_model_id(header_digest: &str) -> Option<ModelId> {
    magnitude_service_contracts::models::LocalModelDigest::new(header_digest.to_owned())
        .ok()
        .map(|digest| ModelId::local(&digest))
}

/// The custom model already on this computer with the same bytes: the same header digest, the
/// same Hugging Face identity and weights size (a projector it was found with, such as an Ollama
/// pull's, is not part of the version), the same files, or files with the same header and sizes
/// wherever they are.
fn on_this_computer(
    store: &ManagedModelStore,
    custom: &[CustomModel],
    hugging_face: Option<(&ModelId, &str)>,
    group: &GgufGroup,
    header_digest: &str,
    local_digest: Option<&str>,
    root: Option<&Path>,
) -> Option<ModelId> {
    let local = local_digest.and_then(local_model_id);
    let paths = root
        .map(|root| {
            group
                .files
                .iter()
                .map(|file| root.join(&file.path))
                .collect::<BTreeSet<_>>()
        })
        .unwrap_or_default();
    let listed = || {
        custom
            .iter()
            .filter(|model| !matches!(model.state, CustomModelState::NotInstalled))
    };
    // A model this version is by identity or location wins over one that merely has its bytes,
    // so a file is never answered with a same-header twin.
    listed()
        .find(|model| {
            Some(&model.model_id) == local.as_ref()
                || hugging_face.is_some_and(|(model_id, commit)| {
                    &model.model_id == model_id
                        && known_commit(model).is_none_or(|known| known == commit)
                        && model.files.iter().map(|file| file.size_bytes).sum::<u64>()
                            == group.size_bytes()
                })
                || (!paths.is_empty() && model.files.iter().any(|file| paths.contains(&file.path)))
        })
        .or_else(|| {
            listed().find(|model| same_bytes(store, model, group, header_digest, local_digest))
        })
        .map(|model| model.model_id.clone())
}

/// The repository commit a listed model is known to be at: an added model's.
fn known_commit(model: &CustomModel) -> Option<&str> {
    match &model.origin {
        CustomModelOrigin::Added { commit, .. } => Some(commit),
        CustomModelOrigin::Imported { .. } | CustomModelOrigin::Found { .. } => None,
    }
}

/// The listed model found at exactly these files.
fn listed_at(custom: &[CustomModel], group: &GgufGroup, root: &Path) -> Option<ModelId> {
    let paths = group
        .files
        .iter()
        .map(|file| root.join(&file.path))
        .collect::<BTreeSet<_>>();
    custom
        .iter()
        .filter(|model| !matches!(model.state, CustomModelState::NotInstalled))
        .find(|model| model.files.iter().any(|file| paths.contains(&file.path)))
        .map(|model| model.model_id.clone())
}

/// Whether a model's weights are the version's bytes: the same file sizes, then the same `local:`
/// digest, which samples the weights, for a local version, or the same header for a remote one
/// whose weights can't be sampled. Files are read only for a model whose sizes already match.
fn same_bytes(
    store: &ManagedModelStore,
    model: &CustomModel,
    group: &GgufGroup,
    header_digest: &str,
    local_digest: Option<&str>,
) -> bool {
    model.files.len() == group.files.len()
        && model
            .files
            .iter()
            .zip(&group.files)
            .all(|(file, version)| file.size_bytes == version.size_bytes)
        && match local_digest {
            Some(local_digest) => {
                self::local_digest(store, model.files.iter().map(|file| file.path.clone()))
                    .is_some_and(|digest| digest == local_digest)
            }
            None => model.files.first().is_some_and(|primary| {
                self::header_digest(store, &primary.path).is_some_and(|digest| digest == header_digest)
            }),
        }
}

/// The verdict of a local file shorter than its header declares, which the engine never reads.
fn unreadable_local_group(
    store: &ManagedModelStore,
    root: &Path,
    group: &GgufGroup,
) -> Option<CachedModelAssessment> {
    let detail = group
        .files
        .iter()
        .find_map(|file| match store.local_gguf(&root.join(&file.path)) {
            Ok(gguf) => gguf.truncated,
            Err(error) => Some(error.to_string()),
        })?;
    Some(super::assessor::unreadable_assessment(detail))
}

/// Each preview header's digest, keyed by the path its file was materialized at.
fn preview_headers(prepared: &magnitude_service_models::PreparedPreview) -> BTreeMap<PathBuf, String> {
    prepared
        .model
        .components
        .iter()
        .filter_map(|component| {
            let header = prepared
                .headers
                .iter()
                .find(|header| component.path.ends_with(&header.path))?;
            Some((component.path.clone(), header.digest.clone()))
        })
        .collect()
}

/// Versions with their verdicts. The recommended version is the largest fitting Q4_K-class
/// version, else the largest fitting version.
/// `projector_bytes` is the projector a download adds, or `None` for local files used in place.
fn candidate_versions(
    versions: Vec<AssessedVersion<'_>>,
    projector_bytes: Option<u64>,
) -> Vec<CandidateVersion> {
    let recommended = recommended_version(
        versions
            .iter()
            .map(|version| (version.group, version.assessment.as_ref())),
    );
    let labels = distinct_labels(
        &versions
            .iter()
            .map(|version| (&version.group.group, version.group.quantization.as_deref(), version.group.embedded_mtp))
            .collect::<Vec<_>>(),
    );
    versions
        .into_iter()
        .zip(labels)
        .enumerate()
        .map(|(index, (version, label))| {
            let group = version.group;
            CandidateVersion {
                label,
                model_id: version.model_id,
                files: group.group.files.iter().map(candidate_file).collect(),
                size_bytes: group.group.size_bytes(),
                download_bytes: match (projector_bytes, &version.on_this_computer) {
                    (Some(projector_bytes), None) => group.group.size_bytes() + projector_bytes,
                    _ => 0,
                },
                assessment: version.assessment.map(|assessment| assessment.profile),
                recommended: Some(index) == recommended,
                catalog_model: version.catalog_model,
                on_this_computer: version.on_this_computer,
            }
        })
        .collect()
}

/// The recommended version: the largest fitting Q4_K-class version, else the largest fitting one.
fn recommended_version<'a>(
    versions: impl Iterator<Item = (&'a ClassifiedGroup, Option<&'a CachedModelAssessment>)>,
) -> Option<usize> {
    versions
        .enumerate()
        .filter(|(_, (_, assessment))| {
            matches!(
                assessment.map(|assessment| &assessment.profile),
                Some(ModelAssessment::Fits { .. })
            )
        })
        .max_by_key(|(_, (group, _))| {
            (
                group.quantization.as_deref().is_some_and(is_q4_k_class),
                group.group.size_bytes(),
            )
        })
        .map(|(index, _)| index)
}

/// Each version's label, distinct among its candidate's: its quantization label (see
/// `version_label`); among versions sharing one, `· MTP` for those whose header carries built-in
/// prediction heads; then the quantization each file's name spells, where those differ; then the
/// name words that tell them apart (those not every one of them has, `· QAD` for `…-QAD-Q4_0`
/// beside `…-Q4_0`); and the file path only when even those don't.
fn distinct_labels(versions: &[(&GgufGroup, Option<&str>, bool)]) -> Vec<String> {
    fn duplicated(labels: &[String]) -> BTreeMap<String, Vec<usize>> {
        let mut by_label = BTreeMap::<String, Vec<usize>>::new();
        for (index, label) in labels.iter().enumerate() {
            by_label.entry(label.clone()).or_default().push(index);
        }
        by_label.retain(|_, indices| indices.len() > 1);
        by_label
    }
    let mut labels = versions
        .iter()
        .map(|(group, quantization, _)| version_label(&group.primary().path, *quantization))
        .collect::<Vec<_>>();
    for indices in duplicated(&labels).into_values() {
        if indices.iter().any(|&index| versions[index].2 != versions[indices[0]].2) {
            for &index in &indices {
                if versions[index].2 {
                    labels[index].push_str(" · MTP");
                }
            }
        }
    }
    // A header label shared by files whose names spell different quantizations (`UD-Q8_K_XL` and
    // `Q8_0`, both Q8_0 by their headers) takes each file's own spelling.
    for indices in duplicated(&labels).into_values() {
        let named = indices
            .iter()
            .map(|&index| version_label(&versions[index].0.primary().path, None))
            .collect::<Vec<_>>();
        for (position, &index) in indices.iter().enumerate() {
            if named.iter().filter(|other| **other == named[position]).count() == 1 {
                let heads = if labels[index].ends_with(" · MTP") { " · MTP" } else { "" };
                labels[index] = format!("{}{heads}", named[position]);
            }
        }
    }
    let words = |index: usize| {
        let name = versions[index].0.primary().path.file_stem().and_then(|stem| stem.to_str()).unwrap_or_default().to_owned();
        name.split(['-', '.'])
            .filter(|word| !word.is_empty())
            .map(str::to_owned)
            .collect::<Vec<_>>()
    };
    for indices in duplicated(&labels).into_values() {
        let shared = indices
            .iter()
            .map(|&index| words(index).into_iter().collect::<BTreeSet<_>>())
            .reduce(|left, right| left.intersection(&right).cloned().collect())
            .unwrap_or_default();
        for &index in &indices {
            let distinct = words(index)
                .into_iter()
                .filter(|word| !shared.contains(word) && !(versions[index].2 && word.eq_ignore_ascii_case("mtp")))
                .collect::<Vec<_>>();
            if !distinct.is_empty() {
                labels[index] = format!("{} · {}", labels[index], distinct.join("-"));
            }
        }
    }
    for indices in duplicated(&labels).into_values() {
        for index in indices {
            labels[index] = path_string(&versions[index].0.primary().path);
        }
    }
    labels
}

/// `(repository, revision, path)` from `owner/repo`, `owner/repo:path`, `owner/repo/path`,
/// `owner/repo@rev:path`, any of those after `hf:`, or a Hugging Face URL (`…/blob/<rev>/<path>`
/// or `…/resolve/<rev>/<path>` included).
fn parse_hugging_face_reference(
    reference: &str,
) -> Result<(String, String, Option<String>), InventoryError> {
    let original_reference = reference;
    let reference = reference.strip_prefix("hf:").unwrap_or(reference);
    let invalid = || {
        InventoryError::InvalidRequest(format!(
            "{original_reference} is not a Hugging Face reference or an absolute file path"
        ))
    };
    if let Some(repository) = hugging_face_repository_from_url(reference) {
        let rest = reference
            .split_once("huggingface.co/")
            .or_else(|| reference.split_once("hf.co/"))
            .map(|(_, rest)| rest)
            .unwrap_or_default();
        let parts = rest.split('/').collect::<Vec<_>>();
        let original = parts.iter().take(2).copied().collect::<Vec<_>>().join("/");
        let _ = repository;
        return match parts.get(2).copied() {
            Some("blob" | "resolve") if parts.len() > 4 => {
                Ok((original, parts[3].to_owned(), Some(parts[4..].join("/"))))
            }
            _ => Ok((original, "main".to_owned(), None)),
        };
    }
    let (repository, path) = match reference.split_once(':') {
        Some((repository, path)) => (repository, Some(path.to_owned())),
        None => match reference.match_indices('/').nth(1) {
            Some((split, _)) if split + 1 < reference.len() => {
                (&reference[..split], Some(reference[split + 1..].to_owned()))
            }
            _ => (reference, None),
        },
    };
    let (repository, revision) = match repository.split_once('@') {
        Some((repository, revision)) => (repository, revision.to_owned()),
        None => (repository, "main".to_owned()),
    };
    HuggingFaceRepositoryId::new(repository.to_owned()).map_err(|_| invalid())?;
    Ok((repository.to_owned(), revision, path))
}

#[cfg(test)]
#[path = "candidate_tests.rs"]
mod hub_tests;

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_gguf() -> Vec<u8> {
        let string = |bytes: &mut Vec<u8>, value: &str| {
            bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
            bytes.extend_from_slice(value.as_bytes());
        };
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(&2_u64.to_le_bytes());
        string(&mut bytes, "general.architecture");
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        string(&mut bytes, "llama");
        string(&mut bytes, "llama.context_length");
        bytes.extend_from_slice(&4_u32.to_le_bytes());
        bytes.extend_from_slice(&4096_u32.to_le_bytes());
        string(&mut bytes, "token_embd.weight");
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&32_u64.to_le_bytes());
        bytes.extend_from_slice(&0_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.resize(bytes.len().next_multiple_of(32), 0);
        bytes.resize(bytes.len() + 128, 0);
        bytes
    }

    #[test]
    fn assessments_are_cached_per_environment_and_shared_by_copies_of_the_same_file() {
        use magnitude_service_contracts::models::AssessmentEnvironmentId;
        let root = tempfile::tempdir().unwrap();
        let first = root.path().join("first/model.gguf");
        let copy = root.path().join("second/copy.gguf");
        for path in [&first, &copy] {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, minimal_gguf()).unwrap();
        }
        let identity = |path: &Path| {
            magnitude_service_models::candidate_files::local_header_digest(path).unwrap()
        };
        let here = AssessmentEnvironmentId("environment_metal_m3".to_owned());
        let elsewhere = AssessmentEnvironmentId("environment_cuda_4090".to_owned());
        let key = |environment: &AssessmentEnvironmentId, path: &Path| {
            assessment_evidence(environment, &identity(path), None, &ModelMethod::Auto).unwrap()
        };
        assert_eq!(
            key(&here, &first),
            key(&here, &copy),
            "copies of the same file share one assessment"
        );
        assert_ne!(
            key(&here, &first),
            key(&elsewhere, &first),
            "another backend or device misses the cache"
        );
        assert_ne!(
            key(&here, &first),
            assessment_evidence(&here, &identity(&first), None, &ModelMethod::Plain).unwrap(),
            "the method is part of the key"
        );
    }

    #[tokio::test]
    async fn a_truncated_local_candidate_is_unreadable_and_a_whole_one_is_assessed() {
        let (_directory, store) = super::hub_tests::bare_store().await;
        let root = tempfile::tempdir().unwrap();
        let whole = minimal_gguf();
        std::fs::write(root.path().join("whole.gguf"), &whole).unwrap();
        std::fs::write(
            root.path().join("truncated.gguf"),
            &whole[..whole.len() - 1],
        )
        .unwrap();
        let group = |name: &str| GgufGroup {
            files: vec![CandidateGguf {
                path: PathBuf::from(name),
                size_bytes: 0,
            }],
        };

        assert!(unreadable_local_group(&store, root.path(), &group("whole.gguf")).is_none());
        let truncated =
            unreadable_local_group(&store, root.path(), &group("truncated.gguf")).unwrap();
        let ModelAssessment::Unsupported {
            profile, reasons, ..
        } = truncated.profile
        else {
            panic!("a truncated file is unsupported");
        };
        assert_eq!(profile, None);
        assert_eq!(
            reasons,
            [
                magnitude_service_contracts::models::UnsupportedReason::Known(
                    magnitude_service_contracts::models::KnownUnsupportedReason::Unreadable
                )
            ]
        );
    }

    #[test]
    fn duplicate_quantization_labels_are_told_apart_by_header_then_by_the_words_that_differ() {
        let group = |path: &str| GgufGroup {
            files: vec![CandidateGguf {
                path: PathBuf::from(path),
                size_bytes: 1,
            }],
        };
        let lfm = [group("LFM2.5-1.2B-Instruct-Q4_0.gguf"), group("LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf")];
        assert_eq!(
            distinct_labels(&[(&lfm[0], Some("Q4_0"), false), (&lfm[1], Some("Q4_0"), false)]),
            ["Q4_0", "Q4_0 · QAD"]
        );
        let fable = [
            group("Fable-NEO-MAX-Q8_0.gguf"),
            group("Fable-NEO-MAX-MTP-Q8_0.gguf"),
            group("Fable-plusIQ-NEO-MAX-MTP-Q8_0.gguf"),
            group("Fable-plusIQ-TOOLS-NEO-MAX-MTP-Q8_0.gguf"),
            group("Fable-NEO-MAX-Q6_K.gguf"),
        ];
        assert_eq!(
            distinct_labels(&[
                (&fable[0], Some("Q8_0"), false),
                (&fable[1], Some("Q8_0"), true),
                (&fable[2], Some("Q8_0"), true),
                (&fable[3], Some("Q8_0"), true),
                (&fable[4], Some("Q6_K"), false),
            ]),
            ["Q8_0", "Q8_0 · MTP", "Q8_0 · MTP · plusIQ", "Q8_0 · MTP · plusIQ-TOOLS", "Q6_K"],
            "built-in heads come from the header; only words that differ are added"
        );
        let same = [group("a/model-Q4_K_M.gguf"), group("b/model-Q4_K_M.gguf")];
        assert_eq!(
            distinct_labels(&[(&same[0], Some("Q4_K_M"), false), (&same[1], Some("Q4_K_M"), false)]),
            ["a/model-Q4_K_M.gguf", "b/model-Q4_K_M.gguf"],
            "files whose names differ in no word are told apart by their paths"
        );
        let gemma = [group("gemma-4-12b-it-Q8_0.gguf"), group("gemma-4-12b-it-UD-Q8_K_XL.gguf")];
        assert_eq!(
            distinct_labels(&[(&gemma[0], Some("Q8_0"), false), (&gemma[1], Some("Q8_0"), false)]),
            ["Q8_0", "UD-Q8_K_XL"],
            "one header label over two spelled quantizations takes each name's spelling"
        );
    }

    #[test]
    fn hugging_face_references_parse_every_accepted_form() {
        assert_eq!(
            parse_hugging_face_reference("z-lab/Qwen3.6-35B-A3B-DFlash").unwrap(),
            (
                "z-lab/Qwen3.6-35B-A3B-DFlash".to_owned(),
                "main".to_owned(),
                None
            )
        );
        assert_eq!(
            parse_hugging_face_reference("owner/repo@abc:MTP/mtp.gguf").unwrap(),
            (
                "owner/repo".to_owned(),
                "abc".to_owned(),
                Some("MTP/mtp.gguf".to_owned())
            )
        );
        assert_eq!(
            parse_hugging_face_reference("https://huggingface.co/owner/repo/blob/main/draft.gguf")
                .unwrap(),
            (
                "owner/repo".to_owned(),
                "main".to_owned(),
                Some("draft.gguf".to_owned())
            )
        );
        let path_form = (
            "magnitudedev/Qwen3.5-4B-DFlash-GGUF".to_owned(),
            "main".to_owned(),
            Some("Qwen3.5-4B-DFlash-Q8_0.gguf".to_owned()),
        );
        assert_eq!(
            parse_hugging_face_reference(
                "magnitudedev/Qwen3.5-4B-DFlash-GGUF/Qwen3.5-4B-DFlash-Q8_0.gguf"
            )
            .unwrap(),
            path_form
        );
        assert_eq!(
            parse_hugging_face_reference(
                "hf:magnitudedev/Qwen3.5-4B-DFlash-GGUF/Qwen3.5-4B-DFlash-Q8_0.gguf"
            )
            .unwrap(),
            path_form
        );
        assert_eq!(
            parse_hugging_face_reference("hf:owner/repo/MTP/mtp.gguf").unwrap(),
            (
                "owner/repo".to_owned(),
                "main".to_owned(),
                Some("MTP/mtp.gguf".to_owned())
            )
        );
        assert_eq!(
            parse_hugging_face_reference("hf:owner/repo").unwrap(),
            ("owner/repo".to_owned(), "main".to_owned(), None)
        );
        assert!(parse_hugging_face_reference("not a reference").is_err());
    }

    fn candidate(
        category: DrafterCategory,
        method: Option<SpeculativeMethod>,
        size_bytes: u64,
        name: &str,
    ) -> DrafterCandidate {
        DrafterCandidate {
            category,
            reference: DrafterReference::Local {
                path: PathBuf::from(format!("/{name}")),
            },
            name: name.to_owned(),
            label: name.to_owned(),
            size_bytes,
            lineage: DrafterLineage::Lineage,
            verdict: match &method {
                Some(method) => DrafterVerdict::Bound {
                    method: method.clone(),
                    memory_bytes: 1,
                },
                None => DrafterVerdict::Rejected {
                    reasons: Vec::new(),
                    detail: "rejected".to_owned(),
                },
            },
            automatic: category != DrafterCategory::Other && method.is_some(),
            on_this_computer: false,
            method,
        }
    }

    fn names(candidates: &[DrafterCandidate]) -> Vec<&str> {
        candidates
            .iter()
            .map(|candidate| candidate.name.as_str())
            .collect()
    }

    #[test]
    fn repository_drafters_order_by_method_then_the_smaller_download() {
        use SpeculativeMethod::{DFlash, DFlash2, DSpark, Mtp};
        let ordered = in_choice_order(vec![
            candidate(DrafterCategory::NextToModel, Some(Mtp), 1, "mtp"),
            candidate(
                DrafterCategory::NextToModel,
                Some(DFlash),
                9,
                "dflash-large",
            ),
            candidate(DrafterCategory::NextToModel, None, 1, "rejected"),
            candidate(DrafterCategory::NextToModel, Some(DSpark), 5, "dspark"),
            candidate(
                DrafterCategory::NextToModel,
                Some(DFlash),
                3,
                "dflash-small",
            ),
            candidate(DrafterCategory::NextToModel, Some(DFlash2), 7, "dflash2"),
        ]);
        assert_eq!(
            names(&ordered),
            [
                "dflash2",
                "dspark",
                "dflash-small",
                "dflash-large",
                "mtp",
                "rejected"
            ]
        );
        assert_eq!(recommended(ordered).unwrap().name, "dflash2");
    }

    #[test]
    fn a_repository_drafter_of_any_method_precedes_the_catalog_drafter_and_built_in_heads() {
        use SpeculativeMethod::{DFlash2, Mtp};
        let ordered = in_choice_order(vec![
            candidate(DrafterCategory::BuiltIn, Some(Mtp), 0, "heads"),
            candidate(DrafterCategory::CatalogDrafter, Some(DFlash2), 1, "catalog"),
            candidate(DrafterCategory::NextToModel, Some(Mtp), 9, "mtp-file"),
        ]);
        assert_eq!(names(&ordered), ["mtp-file", "catalog", "heads"]);
        assert_eq!(recommended(ordered).unwrap().name, "mtp-file");
    }

    #[test]
    fn the_catalog_drafter_precedes_built_in_heads_and_any_separate_drafter_beats_heads() {
        use SpeculativeMethod::{DFlash, Mtp};
        let catalog = in_choice_order(vec![
            candidate(DrafterCategory::BuiltIn, Some(Mtp), 0, "heads"),
            candidate(DrafterCategory::CatalogDrafter, Some(DFlash), 9, "catalog"),
        ]);
        assert_eq!(recommended(catalog).unwrap().name, "catalog");
        let built_in = in_choice_order(vec![candidate(
            DrafterCategory::BuiltIn,
            Some(Mtp),
            0,
            "heads",
        )]);
        assert_eq!(recommended(built_in).unwrap().name, "heads");
    }

    #[test]
    fn rejected_and_user_entered_drafters_are_never_recommended() {
        use SpeculativeMethod::Mtp;
        let ordered = in_choice_order(vec![
            candidate(DrafterCategory::NextToModel, None, 1, "rejected-repository"),
            candidate(DrafterCategory::CatalogDrafter, None, 1, "rejected-catalog"),
            candidate(DrafterCategory::Other, Some(Mtp), 1, "entered"),
        ]);
        assert_eq!(recommended(ordered.clone()), None);
        let with_heads = in_choice_order(
            ordered
                .into_iter()
                .chain([candidate(DrafterCategory::BuiltIn, Some(Mtp), 0, "heads")])
                .collect(),
        );
        assert_eq!(recommended(with_heads).unwrap().name, "heads");
    }

    #[test]
    fn a_drafter_already_here_counts_as_no_download() {
        use SpeculativeMethod::DFlash;
        let mut here = candidate(DrafterCategory::NextToModel, Some(DFlash), 9, "here");
        here.on_this_computer = true;
        let ordered = in_choice_order(vec![
            candidate(DrafterCategory::NextToModel, Some(DFlash), 3, "remote"),
            here,
        ]);
        assert_eq!(names(&ordered), ["here", "remote"]);
    }
}
