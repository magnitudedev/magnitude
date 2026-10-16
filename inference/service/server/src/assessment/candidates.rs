//! Candidates: files not yet added (a Hugging Face repository at a commit, or a local path), and
//! the drafters any model can use. Inspection groups a candidate's GGUFs into versions and asks
//! the engine for each version's verdict from its headers alone; drafter enumeration lists every
//! drafter known for a target in the product's category order, each with the engine's binding
//! verdict. Assessments here are cached by the exact header material they read, like every
//! assessment.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Weak};

use futures_util::StreamExt;
use futures_util::future::BoxFuture;
use magnitude_engine::assessment::{
    ModelPackagePaths, PreparedModelAssessment, finish_model_assessment, prepare_model_assessment,
};
use magnitude_engine::fingerprint::ModelFingerprint as EngineModelFingerprint;
use magnitude_engine::options::ModelMethod;
use magnitude_service_contracts::models::{
    CandidateDrafter, CandidateFile, CandidateInspection, CandidateInspectionRequest,
    CandidateSource, CandidateVersion, CatalogModelState, CatalogModels, CustomModel,
    CustomModelOrigin, CustomModelState, DrafterCandidate, DrafterCandidatesRequest,
    DrafterCandidatesResponse, DrafterCategory, DrafterLineage, DrafterReference, DrafterVerdict,
    HuggingFaceRepositoryId, InstalledModelPackages, LocalModels, ModelAssessment,
    ModelAssessmentProfile, ModelAssessmentSubject, ModelBundleInput, ModelCandidate,
    ModelCandidates, ModelFileRole, ModelFingerprint, ModelId, ModelPackage, ModelPackageOperand,
    ModelPackageSource, ParsedModelId, ServableModelBundle, ServingProfile, SpeculativeDraftSource,
    SpeculativeMethod,
};
use magnitude_service_contracts::{
    ComponentRole, HuggingFaceRepositoryFile, HuggingFaceRepositoryRequest,
    HuggingFaceRepositorySnapshot, InventoryError, ModelPreviewComponentRole,
    ModelPreviewComponentSource, ModelPreviewSource, ResolvedModel,
};
use magnitude_service_models::candidate_files::{
    CandidateFileRole, CandidateGguf, GgufGroup, RecognizedArchitectures, display_name,
    group_ggufs, header_base_repositories, header_inspection, hugging_face_repository_from_url,
    is_q4_k_class, local_directory, local_header_digest, version_label,
};
use magnitude_service_models::lineage::{CatalogLineage, CatalogLineageMatch};
use magnitude_service_models::{
    CachedModelAssessment, CustomModelEnrichment, CustomModelEnrichmentResult, CustomModelFacts,
    ManagedModelStore, ModelDomainResolver, ServableModelBundleKey,
    refresh_hugging_face_repository,
};
use sha2::{Digest, Sha256};

use super::assessor::{ModelAssessor, model_assessment, performance_depths, preparation_error};

/// Concurrent header requests while classifying a repository's files.
const HEADER_CONCURRENCY: usize = 6;
/// Concurrent engine assessments of one request's versions or drafters.
const ASSESSMENT_CONCURRENCY: usize = 3;
const MAX_LOCAL_SCAN_DEPTH: usize = 2;
const MAX_LOCAL_SCAN_FILES: usize = 512;
const MAX_ON_THIS_COMPUTER_DRAFTERS: usize = 8;
const MAX_HUB_DRAFTER_BASES: usize = 3;
const MAX_HUB_DRAFTER_REPOSITORIES: usize = 4;
const MAX_HUB_DRAFTER_FILES: usize = 2;
/// How long a custom model whose enrichment failed waits before it is tried again.
const ENRICHMENT_RETRY_DELAY: std::time::Duration = std::time::Duration::from_secs(300);

type Guard = Arc<dyn Send + Sync>;

/// A target's header material: its first GGUF (shards beside it), its projector, and the identity
/// of exactly those bytes.
struct TargetMaterial {
    target: PathBuf,
    projector: Option<PathBuf>,
    identity: String,
    _guards: Vec<Guard>,
}

/// A separate drafter's header material.
struct DrafterMaterial {
    path: PathBuf,
    identity: String,
    _guards: Vec<Guard>,
}

/// Everything drafter enumeration knows about one target.
struct DrafterTarget {
    material: TargetMaterial,
    repository: Option<(String, String)>,
    primary: Option<PathBuf>,
    folder: Option<PathBuf>,
    snapshot: Option<HuggingFaceRepositorySnapshot>,
    lineage: BTreeSet<String>,
    catalog: Option<ModelId>,
}

/// One classified complete GGUF of a candidate.
struct ClassifiedGroup {
    group: GgufGroup,
    role: CandidateFileRole,
    header_digest: String,
    base_repositories: BTreeSet<String>,
    quantization: Option<String>,
}

/// Computed lineage hints and recommended drafters, keyed by model and exact target package.
#[derive(Default)]
struct CandidateEnrichment {
    results: std::sync::RwLock<BTreeMap<(ModelId, String), CustomModelEnrichmentResult>>,
}

impl CandidateEnrichment {
    fn knows(&self, key: &(ModelId, String)) -> bool {
        self.results
            .read()
            .is_ok_and(|results| results.contains_key(key))
    }

    /// Record a result; whether it changed what the projection sees.
    fn record(&self, key: (ModelId, String), result: CustomModelEnrichmentResult) -> bool {
        let Ok(mut results) = self.results.write() else {
            return false;
        };
        let previous = results
            .iter()
            .find(|((model_id, _), _)| model_id == &key.0)
            .map(|(_, previous)| previous.clone())
            .unwrap_or_default();
        results.retain(|(model_id, _), _| model_id != &key.0);
        results.insert(key, result.clone());
        previous != result
    }
}

impl CustomModelEnrichment for CandidateEnrichment {
    fn enrich(&self, model: &CustomModelFacts<'_>) -> CustomModelEnrichmentResult {
        self.results
            .read()
            .ok()
            .and_then(|results| {
                results
                    .get(&(model.model_id.clone(), model.target.id.0.clone()))
                    .cloned()
            })
            .unwrap_or_default()
    }
}

pub struct ManagedModelCandidates {
    assessor: Arc<ModelAssessor>,
    lineage: Arc<CatalogLineage>,
    architectures: RecognizedArchitectures,
    catalog: Arc<dyn CatalogModels>,
    local: Arc<dyn LocalModels>,
    catalog_fingerprints: tokio::sync::OnceCell<Vec<(ModelId, ModelFingerprint)>>,
    gates: tokio::sync::Mutex<BTreeMap<String, Weak<tokio::sync::Mutex<()>>>>,
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
            catalog_fingerprints: tokio::sync::OnceCell::new(),
            gates: tokio::sync::Mutex::new(BTreeMap::new()),
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
        let target = self.model_target(model_id).await?;
        Ok(self
            .enumerate_drafters(target, None, true)
            .await?
            .into_iter()
            .find(|candidate| matches!(candidate.verdict, DrafterVerdict::Bound { .. })))
    }

    /// A custom model's lineage hint and recommended drafter.
    async fn enrich_model(
        &self,
        model_id: &ModelId,
    ) -> Result<CustomModelEnrichmentResult, InventoryError> {
        let target = self.model_target(model_id).await?;
        let plain = self.assess(&target.material, None, ModelMethod::Auto).await?;
        let exact = match (&target.repository, &target.primary) {
            (Some((repository, commit)), Some(primary)) => {
                Some((repository.as_str(), commit.as_str(), primary.as_path()))
            }
            _ => None,
        };
        let variant_of = self
            .catalog_match(exact, &target.lineage, plain.fingerprint.as_ref())
            .await
            .map(|(matched, _)| matched.model_id);
        let recommended_drafter = self
            .enumerate_drafters(target, None, true)
            .await?
            .into_iter()
            .find(|candidate| matches!(candidate.verdict, DrafterVerdict::Bound { .. }));
        Ok(CustomModelEnrichmentResult {
            variant_of,
            recommended_drafter,
        })
    }

    /// Fill every custom model's lineage hint and recommended drafter. The projection asks the
    /// enrichment synchronously, so results are computed here in the background, one model at a
    /// time, keyed by the model and its exact target package, and the list is republished when
    /// they change.
    pub fn start_enrichment(self: &Arc<Self>, resolver: Arc<ModelDomainResolver>) {
        let enrichment = Arc::new(CandidateEnrichment::default());
        resolver.set_custom_model_enrichment(enrichment.clone());
        let candidates = Arc::clone(self);
        tokio::spawn(async move {
            let mut changes = candidates.local.watch_local();
            let mut failed = BTreeMap::<(ModelId, String), std::time::Instant>::new();
            loop {
                let models = candidates
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
                let mut changed = false;
                for model_id in models {
                    let Ok(configuration) =
                        candidates
                            .assessor
                            .configuration_for(&ModelAssessmentSubject::Custom {
                                model_id: model_id.clone(),
                            })
                    else {
                        continue;
                    };
                    let key = (model_id.clone(), bundle_target(&configuration.bundle).id.0.clone());
                    if enrichment.knows(&key)
                        || failed
                            .get(&key)
                            .is_some_and(|at| at.elapsed() < ENRICHMENT_RETRY_DELAY)
                    {
                        continue;
                    }
                    match candidates.enrich_model(&model_id).await {
                        Ok(result) => {
                            failed.remove(&key);
                            changed |= enrichment.record(key, result);
                        }
                        Err(error) => {
                            tracing::warn!(model.id = %model_id.as_str(), %error, "custom model enrichment failed");
                            failed.insert(key, std::time::Instant::now());
                        }
                    }
                }
                if changed {
                    resolver.custom_model_enrichment_changed();
                }
                tokio::select! {
                    change = changes.next() => if change.is_none() { return },
                    () = tokio::time::sleep(ENRICHMENT_RETRY_DELAY), if !failed.is_empty() => {}
                }
            }
        });
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
        let evidence = serde_json::to_string(&(
            "candidate-assessment-v1",
            &environment.id.0,
            &target.identity,
            drafter.map(|drafter| drafter.identity.as_str()),
            format!("{method:?}"),
        ))
        .map_err(|error| InventoryError::Internal(error.to_string()))?;
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
        let setup_environment = Arc::clone(&environment);
        let (prepared, fallback_context) = crate::spawn_blocking_traced(move || {
            let fallback_context = magnitude_service_models::gguf::inspect(&paths.target)
                .ok()
                .and_then(|inspection| inspection.training_context_length)
                .unwrap_or(0);
            (
                prepare_model_assessment(&paths, &setup_environment.setup),
                fallback_context,
            )
        })
        .await
        .map_err(|error| InventoryError::Internal(format!("candidate preparation failed: {error}")))?;
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
        let identity = prepared
            .headers
            .iter()
            .map(|header| format!("{}={}", header.path.display(), header.digest))
            .collect::<Vec<_>>()
            .join(",");
        Ok(TargetMaterial {
            target,
            projector,
            identity,
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
            _guards: vec![Arc::new(prepared)],
        })
    }

    fn local_target(
        &self,
        target: PathBuf,
        projector: Option<PathBuf>,
    ) -> Result<TargetMaterial, InventoryError> {
        let mut identity = Vec::new();
        for path in std::iter::once(&target).chain(projector.iter()) {
            identity.push(local_header_digest(path)?);
        }
        Ok(TargetMaterial {
            target,
            projector,
            identity: identity.join(","),
            _guards: Vec::new(),
        })
    }

    fn local_drafter(&self, path: PathBuf) -> Result<DrafterMaterial, InventoryError> {
        Ok(DrafterMaterial {
            identity: local_header_digest(&path)?,
            path,
            _guards: Vec::new(),
        })
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
                        matches!(component.role, ComponentRole::Weights | ComponentRole::Draft)
                    })
                    .map(|component| component.path.clone())
                    .min()
            })
            .ok_or_else(|| InventoryError::Integrity("the draft resolved without weights".to_owned()))?;
        Ok(Some(DrafterMaterial {
            path,
            identity,
            _guards: vec![Arc::new(resolved)],
        }))
    }

    async fn installed_package_material(
        &self,
        package: &ModelPackage,
    ) -> Result<magnitude_service_contracts::models::ResolvedServableModelBundle, InventoryError>
    {
        self.models()
            .resolve_bundle(ModelBundleInput::Standalone {
                package: ModelPackageOperand::SourceBacked {
                    package: package.clone(),
                },
            })
            .await
    }

    // Classification -------------------------------------------------------------------------

    async fn classify_hugging_face(
        &self,
        snapshot: &HuggingFaceRepositorySnapshot,
    ) -> Result<Vec<ClassifiedGroup>, InventoryError> {
        let files = snapshot
            .gguf_files
            .iter()
            .map(|file| CandidateGguf {
                path: file.path.clone(),
                size_bytes: file.size_bytes,
            })
            .collect::<Vec<_>>();
        let mut pending = Vec::new();
        for group in group_ggufs(&files) {
            pending.push(self.classify_hugging_face_group(snapshot, group));
        }
        let classified = futures_util::stream::iter(pending)
            .buffered(HEADER_CONCURRENCY)
            .collect::<Vec<_>>()
            .await;
        classified.into_iter().collect()
    }

    async fn classify_hugging_face_group(
        &self,
        snapshot: &HuggingFaceRepositorySnapshot,
        group: GgufGroup,
    ) -> Result<ClassifiedGroup, InventoryError> {
        let file = snapshot_file(snapshot, &group.primary().path)?;
        let header = self
            .models()
            .hugging_face_gguf_header(&snapshot.repository, &snapshot.commit, file)
            .await?;
        let (directory, inspection) = header_inspection(&header.bytes)?;
        Ok(ClassifiedGroup {
            role: CandidateFileRole::of(&directory, &self.architectures),
            base_repositories: header_base_repositories(&directory),
            quantization: inspection.quantization,
            header_digest: header.digest,
            group,
        })
    }

    /// A version's verdict, when its architecture is one the engine runs. A failure to read its
    /// material leaves the version without a verdict rather than failing the inspection.
    async fn version_assessment(
        &self,
        recognized: bool,
        material: impl std::future::Future<Output = Result<TargetMaterial, InventoryError>>,
    ) -> Option<CachedModelAssessment> {
        if !recognized {
            return None;
        }
        let material = material
            .await
            .map_err(|error| tracing::warn!(%error, "candidate version material is unavailable"))
            .ok()?;
        self.assess(&material, None, ModelMethod::Auto)
            .await
            .map_err(|error| tracing::warn!(%error, "candidate version assessment failed"))
            .ok()
    }

    fn classify_local(&self, root: &Path, files: &[CandidateGguf]) -> Vec<ClassifiedGroup> {
        group_ggufs(files)
            .into_iter()
            .filter_map(|group| {
                let path = root.join(&group.primary().path);
                let directory = local_directory(&path).ok()?;
                let inspection = magnitude_service_models::gguf::inspect(&path).ok()?;
                Some(ClassifiedGroup {
                    role: CandidateFileRole::of(&directory, &self.architectures),
                    base_repositories: header_base_repositories(&directory),
                    quantization: inspection.quantization,
                    header_digest: local_header_digest(&path).ok()?,
                    group,
                })
            })
            .collect()
    }

    // Inspection -----------------------------------------------------------------------------

    async fn inspect_hugging_face(
        &self,
        repository: HuggingFaceRepositoryId,
        revision: String,
    ) -> Result<CandidateInspection, InventoryError> {
        let snapshot = self.snapshot(repository.as_str(), &revision).await?;
        let classified = self.classify_hugging_face(&snapshot).await?;
        let projector = choose_projector(&classified);
        let targets = classified
            .iter()
            .filter(|group| matches!(group.role, CandidateFileRole::Target { .. }))
            .collect::<Vec<_>>();
        let unsupported_architecture = self.unsupported_architecture(&targets);
        let custom = self.custom_models().await;
        let catalog_installed = self.installed_catalog_models().await;
        let projector_path = projector.map(|projector| projector.group.primary().path.clone());
        let mut pending = Vec::new();
        for group in &targets {
            pending.push(self.version_assessment(
                self.recognized(group),
                self.hugging_face_target(
                    &snapshot,
                    &group.group.primary().path,
                    projector_path.as_deref(),
                ),
            ));
        }
        let assessments = futures_util::stream::iter(pending)
            .buffered(ASSESSMENT_CONCURRENCY)
            .collect::<Vec<_>>()
            .await;
        let versions = targets
            .iter()
            .copied()
            .zip(assessments)
            .filter_map(|(group, assessment)| {
                let model_id = hugging_face_model_id(&snapshot.repository, &group.group)?;
                let on_this_computer = self
                    .lineage
                    .exact_target(&snapshot.repository, &snapshot.commit, &group.group.primary().path)
                    .map(|matched| matched.model_id)
                    .filter(|model_id| catalog_installed.contains(model_id))
                    .or_else(|| {
                        on_this_computer(
                            &custom,
                            Some(&model_id),
                            &group.group,
                            &group.header_digest,
                            None,
                        )
                    });
                Some(AssessedVersion {
                    group,
                    assessment,
                    model_id,
                    on_this_computer,
                })
            })
            .collect::<Vec<_>>();
        let mut lineage = self
            .lineage
            .hub()
            .lineage(&snapshot.repository, Some(snapshot.base_models.clone()))
            .await;
        lineage.extend(
            targets
                .iter()
                .flat_map(|group| group.base_repositories.iter().cloned()),
        );
        let fingerprint = versions
            .iter()
            .find_map(|version| version.assessment.as_ref()?.fingerprint.clone());
        let variant_of = self
            .catalog_match(None, &lineage, fingerprint.as_ref())
            .await
            .map(|(matched, _)| matched.model_id);
        Ok(CandidateInspection {
            source: CandidateSource::HuggingFace {
                repository: HuggingFaceRepositoryId::new(snapshot.repository.clone())
                    .map_err(|error| InventoryError::InvalidRequest(error.to_string()))?,
                commit: snapshot.commit.clone(),
            },
            display_name: display_name(&snapshot.repository),
            license: snapshot.license.clone(),
            variant_of,
            versions: candidate_versions(versions),
            projector: projector.map(|projector| candidate_file(projector.group.primary())),
            unsupported_architecture,
        })
    }

    async fn inspect_local(&self, path: PathBuf) -> Result<CandidateInspection, InventoryError> {
        let metadata = std::fs::metadata(&path)
            .map_err(|error| InventoryError::NotFound(format!("{}: {error}", path.display())))?;
        let (root, selected) = if metadata.is_file() {
            let root = path
                .parent()
                .map(Path::to_path_buf)
                .ok_or_else(|| InventoryError::InvalidRequest("a file needs a folder".to_owned()))?;
            let name = path
                .file_name()
                .map(PathBuf::from)
                .ok_or_else(|| InventoryError::InvalidRequest("invalid file path".to_owned()))?;
            (root, Some(name))
        } else {
            (path.clone(), None)
        };
        let depth = if selected.is_some() { 1 } else { MAX_LOCAL_SCAN_DEPTH };
        let files = scan_ggufs(&root, depth);
        let classified = self.classify_local(&root, &files);
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
            return Err(InventoryError::InvalidRequest(format!(
                "{} is not a model the engine could run on its own (it is a projector, a drafter or an incomplete split file)",
                path.display()
            )));
        }
        let unsupported_architecture = self.unsupported_architecture(&targets);
        let custom = self.custom_models().await;
        let projector_path = projector.map(|projector| root.join(&projector.group.primary().path));
        let mut pending = Vec::new();
        for group in &targets {
            let material = self.local_target(
                root.join(&group.group.primary().path),
                projector_path.clone(),
            );
            pending.push(self.version_assessment(self.recognized(group), async { material }));
        }
        let assessments = futures_util::stream::iter(pending)
            .buffered(ASSESSMENT_CONCURRENCY)
            .collect::<Vec<_>>()
            .await;
        let versions = targets
            .iter()
            .copied()
            .zip(assessments)
            .filter_map(|(group, assessment)| {
                let on_this_computer =
                    on_this_computer(&custom, None, &group.group, &group.header_digest, Some(&root));
                // A file already listed keeps that model's identity (a Hugging Face one when its
                // location gives it); any other file is identified by its header digest.
                let model_id = on_this_computer
                    .clone()
                    .or_else(|| local_model_id(&group.header_digest))?;
                Some(AssessedVersion {
                    group,
                    assessment,
                    model_id,
                    on_this_computer,
                })
            })
            .collect::<Vec<_>>();
        let lineage = targets
            .iter()
            .flat_map(|group| group.base_repositories.iter().cloned())
            .collect::<BTreeSet<_>>();
        let fingerprint = versions
            .iter()
            .find_map(|version| version.assessment.as_ref()?.fingerprint.clone());
        let variant_of = self
            .catalog_match(None, &lineage, fingerprint.as_ref())
            .await
            .map(|(matched, _)| matched.model_id);
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
            versions: candidate_versions(versions),
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
        if targets.is_empty() || targets.iter().any(|group| self.recognized(group)) {
            return None;
        }
        Some(
            targets
                .iter()
                .filter_map(|group| match &group.role {
                    CandidateFileRole::Target { architecture } => {
                        Some(architecture.clone().unwrap_or_default())
                    }
                    _ => None,
                })
                .collect::<BTreeSet<_>>()
                .into_iter()
                .map(|architecture| {
                    magnitude_service_contracts::models::UnsupportedReason::Architecture {
                        architecture,
                    }
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
                    .filter(|model| matches!(model.local_state, CatalogModelState::Installed { .. }))
                    .map(|model| model.id)
                    .collect()
            })
            .unwrap_or_default()
    }

    // Catalog matching ----------------------------------------------------------------------

    /// Each catalog model's shape fingerprint, from its release header material, computed once.
    async fn catalog_fingerprints(&self) -> &[(ModelId, ModelFingerprint)] {
        self.catalog_fingerprints
            .get_or_init(|| async {
                let mut fingerprints = Vec::new();
                for candidate in self.lineage.representatives().collect::<Vec<_>>() {
                    let resolved = match self
                        .assessor
                        .resolve_material(candidate.model.configuration.bundle.clone())
                        .await
                    {
                        Ok(resolved) => resolved,
                        Err(error) => {
                            tracing::warn!(%error, model.id = %candidate.model_id.as_str(), "catalog fingerprint material is unavailable");
                            continue;
                        }
                    };
                    let fingerprint = crate::spawn_blocking_traced(move || {
                        let target = resolved_paths(&resolved.target_model).ok()?.0;
                        EngineModelFingerprint::from_path(&target)
                            .ok()
                            .map(super::assessor::model_fingerprint)
                    })
                    .await
                    .ok()
                    .flatten();
                    if let Some(fingerprint) = fingerprint {
                        fingerprints.push((candidate.model_id, fingerprint));
                    }
                }
                fingerprints
            })
            .await
    }

    /// The catalog model a target derives from: an exact catalog target, then lineage, then an
    /// equal shape fingerprint.
    async fn catalog_match(
        &self,
        exact: Option<(&str, &str, &Path)>,
        lineage: &BTreeSet<String>,
        fingerprint: Option<&ModelFingerprint>,
    ) -> Option<(CatalogLineageMatch, DrafterLineage)> {
        if let Some((repository, commit, primary)) = exact
            && let Some(matched) = self.lineage.exact_target(repository, commit, primary)
        {
            return Some((matched, DrafterLineage::Exact));
        }
        if let Some(matched) = self.lineage.match_lineage(lineage).await {
            return Some((matched, DrafterLineage::Lineage));
        }
        let fingerprint = fingerprint?;
        let model_id = self
            .catalog_fingerprints()
            .await
            .iter()
            .find(|(_, candidate)| candidate == fingerprint)
            .map(|(model_id, _)| model_id.clone())?;
        self.lineage
            .variant(&model_id)
            .map(|matched| (matched, DrafterLineage::Fingerprint))
    }

    // Drafter targets ------------------------------------------------------------------------

    async fn candidate_target(&self, candidate: &ModelCandidate) -> Result<DrafterTarget, InventoryError> {
        let primary = candidate
            .target
            .first()
            .map(PathBuf::from)
            .ok_or_else(|| InventoryError::InvalidRequest("a candidate names its target".to_owned()))?;
        match &candidate.source {
            CandidateSource::HuggingFace { repository, commit } => {
                let snapshot = self.snapshot(repository.as_str(), commit).await?;
                let projector = candidate.projector.as_ref().map(|file| PathBuf::from(&file.path));
                let material = self
                    .hugging_face_target(&snapshot, &primary, projector.as_deref())
                    .await?;
                let mut lineage = self
                    .lineage
                    .hub()
                    .lineage(&snapshot.repository, Some(snapshot.base_models.clone()))
                    .await;
                lineage.extend(header_lineage(&material.target));
                Ok(DrafterTarget {
                    material,
                    repository: Some((snapshot.repository.clone(), snapshot.commit.clone())),
                    primary: Some(primary),
                    folder: None,
                    snapshot: Some(snapshot),
                    lineage,
                    catalog: None,
                })
            }
            CandidateSource::Local { path } => {
                let target = path.join(&primary);
                let projector = candidate.projector.as_ref().map(|file| path.join(&file.path));
                let lineage = header_lineage(&target);
                let folder = target.parent().map(Path::to_path_buf);
                Ok(DrafterTarget {
                    material: self.local_target(target, projector)?,
                    repository: None,
                    primary: None,
                    folder,
                    snapshot: None,
                    lineage,
                    catalog: None,
                })
            }
        }
    }

    /// A model's target: a catalog model's release material, or a custom model's installed
    /// target package.
    async fn model_target(&self, model_id: &ModelId) -> Result<DrafterTarget, InventoryError> {
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
                folder: None,
                snapshot: None,
                lineage: BTreeSet::new(),
                catalog: Some(model_id.clone()),
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
            (_, ModelPackageSource::HuggingFace { repository, revision }) => {
                Some((repository.clone(), revision.clone()))
            }
            _ => match model_id.parsed() {
                ParsedModelId::HuggingFace { repository_id, .. } => {
                    Some((repository_id.as_str().to_owned(), String::new()))
                }
                _ => None,
            },
        };
        let primary = target
            .files
            .iter()
            .filter(|file| file.role == ModelFileRole::Weights)
            .map(|file| file.path.clone())
            .min();
        let folder = match &model.origin {
            CustomModelOrigin::Added { .. } => None,
            CustomModelOrigin::Imported { .. } | CustomModelOrigin::Found { .. } => {
                material.target.parent().map(Path::to_path_buf)
            }
        };
        let mut lineage = header_lineage(&material.target);
        let snapshot = match &repository {
            Some((repository, commit)) => {
                lineage.extend(self.lineage.hub().lineage(repository, None).await);
                if is_commit(commit) {
                    self.snapshot(repository, commit).await.ok()
                } else {
                    None
                }
            }
            None => None,
        };
        Ok(DrafterTarget {
            material,
            repository,
            primary,
            folder,
            snapshot,
            lineage,
            catalog: None,
        })
    }

    // Drafter enumeration --------------------------------------------------------------------

    async fn enumerate_drafters(
        &self,
        target: DrafterTarget,
        other: Option<&str>,
        stop_at_first_bound: bool,
    ) -> Result<Vec<DrafterCandidate>, InventoryError> {
        let plain = self.assess(&target.material, None, ModelMethod::Auto).await?;
        if matches!(plain.profile, ModelAssessment::Unsupported { .. }) {
            return Ok(Vec::new());
        }
        let exact = match (&target.repository, &target.primary) {
            (Some((repository, commit)), Some(primary)) => {
                Some((repository.as_str(), commit.as_str(), primary.as_path()))
            }
            _ => None,
        };
        let matched = match &target.catalog {
            Some(model_id) => self
                .lineage
                .variant(model_id)
                .map(|matched| (matched, DrafterLineage::Exact)),
            None => {
                self.catalog_match(exact, &target.lineage, plain.fingerprint.as_ref())
                    .await
            }
        };
        let mut related = target.lineage.clone();
        if let Some((matched, _)) = &matched {
            related.extend(self.lineage.lineage_of(&matched.model_id).await);
        }
        let mut candidates = Vec::new();
        let mut seen = BTreeSet::new();
        let done = |candidates: &[DrafterCandidate]| {
            stop_at_first_bound
                && candidates
                    .iter()
                    .any(|candidate| matches!(candidate.verdict, DrafterVerdict::Bound { .. }))
        };

        if let Some((matched, lineage)) = &matched {
            if let Some(candidate) = self.catalog_drafter(&target, matched, *lineage).await {
                seen.insert(reference_key(&candidate.reference));
                if let Some(path) = catalog_draft_key(&matched.model) {
                    seen.insert(path);
                }
                candidates.push(candidate);
            }
        }
        if target.catalog.is_some() {
            return Ok(candidates);
        }
        let embedded_catalog = matched.as_ref().is_some_and(|(matched, _)| {
            matches!(
                &matched.model.configuration.bundle,
                ServableModelBundle::SpeculativeDecoding {
                    draft_source: SpeculativeDraftSource::Embedded,
                    ..
                }
            )
        });
        if !embedded_catalog && !done(&candidates) {
            let verdict = verdict_of(&plain.profile);
            if !matches!(verdict, DrafterVerdict::None) {
                candidates.push(DrafterCandidate {
                    category: DrafterCategory::BuiltIn,
                    reference: DrafterReference::Embedded,
                    label: "Built into this model".to_owned(),
                    size_bytes: 0,
                    lineage: DrafterLineage::Exact,
                    method: bound_method(&verdict),
                    verdict,
                    on_this_computer: true,
                });
            }
        }
        if !done(&candidates) {
            let next_to = self.next_to_model(&target, &mut seen).await;
            candidates.extend(sorted(next_to));
        }
        if !done(&candidates) {
            let on_computer = self.on_this_computer(&target, &related, &mut seen).await;
            candidates.extend(sorted(on_computer));
        }
        if !done(&candidates) {
            let hub = self.hub_drafters(&target, &related, &mut seen).await;
            candidates.extend(sorted(hub));
        }
        if let Some(other) = other {
            match self.other_drafter(&target, &related, other).await {
                Ok(candidate) if !seen.contains(&reference_key(&candidate.reference)) => {
                    candidates.push(candidate);
                }
                Ok(_) => {}
                Err(error) => return Err(error),
            }
        }
        Ok(candidates)
    }

    async fn catalog_drafter(
        &self,
        target: &DrafterTarget,
        matched: &CatalogLineageMatch,
        lineage: DrafterLineage,
    ) -> Option<DrafterCandidate> {
        let ServableModelBundle::SpeculativeDecoding {
            draft_source,
            method,
            ..
        } = &matched.model.configuration.bundle
        else {
            return None;
        };
        let (drafter, size_bytes, on_this_computer) = match draft_source {
            SpeculativeDraftSource::Embedded => (None, 0, true),
            SpeculativeDraftSource::Separate { draft } => {
                let resolved = self
                    .assessor
                    .resolve_material(matched.model.configuration.bundle.clone())
                    .await
                    .map_err(|error| tracing::warn!(%error, "catalog drafter material is unavailable"))
                    .ok()?;
                let material = Self::resolved_drafter(resolved).ok().flatten()?;
                (
                    Some(material),
                    package_bytes(draft),
                    self.models().has_installed_package(&draft.id),
                )
            }
        };
        let verdict = self
            .assess(&target.material, drafter.as_ref(), engine_method(method))
            .await
            .map(|assessed| verdict_of(&assessed.profile))
            .map_err(|error| tracing::warn!(%error, "catalog drafter assessment failed"))
            .ok()?;
        Some(DrafterCandidate {
            category: DrafterCategory::CatalogDrafter,
            reference: DrafterReference::Catalog {
                model_id: matched.model_id.clone(),
            },
            label: format!("Recommended · Trained for {}", matched.model.display_name),
            size_bytes,
            lineage,
            method: bound_method(&verdict).or(Some(method.clone())),
            verdict,
            on_this_computer,
        })
    }

    /// Drafter files in the target's repository or folder.
    async fn next_to_model(
        &self,
        target: &DrafterTarget,
        seen: &mut BTreeSet<String>,
    ) -> Vec<DrafterCandidate> {
        let mut found = Vec::new();
        if let Some(snapshot) = &target.snapshot {
            let Ok(classified) = self.classify_hugging_face(snapshot).await else {
                return found;
            };
            let drafters = classified
                .into_iter()
                .filter(|group| matches!(group.role, CandidateFileRole::Drafter { .. }))
                .filter(|group| group.group.files.len() == 1)
                .filter(|group| {
                    seen.insert(reference_key(&DrafterReference::HuggingFace {
                        repository: hugging_face_id(&snapshot.repository),
                        commit: snapshot.commit.clone(),
                        path: path_string(&group.group.primary().path),
                    }))
                })
                .collect::<Vec<_>>();
            let mut pending = Vec::new();
            for group in drafters {
                pending.push(self.hugging_face_drafter_candidate(
                    target,
                    snapshot,
                    group.group.primary().path.clone(),
                    DrafterCategory::NextToModel,
                    "In this repository".to_owned(),
                ));
            }
            found = collect_candidates(pending).await;
        }
        if let Some(folder) = &target.folder {
            let files = scan_ggufs(folder, MAX_LOCAL_SCAN_DEPTH);
            let drafters = self
                .classify_local(folder, &files)
                .into_iter()
                .filter(|group| matches!(group.role, CandidateFileRole::Drafter { .. }))
                .filter(|group| group.group.files.len() == 1)
                .map(|group| (folder.join(&group.group.primary().path), group.group.size_bytes()))
                .filter(|(path, _)| {
                    seen.insert(reference_key(&DrafterReference::Local { path: path.clone() }))
                })
                .collect::<Vec<_>>();
            let mut pending = Vec::new();
            for (path, size_bytes) in drafters {
                pending.push(self.local_drafter_candidate(target, path, size_bytes));
            }
            let local = collect_candidates(pending).await;
            found.extend(local);
        }
        found
    }

    /// Installed drafters in any location whose lineage matches the target.
    async fn on_this_computer(
        &self,
        target: &DrafterTarget,
        related: &BTreeSet<String>,
        seen: &mut BTreeSet<String>,
    ) -> Vec<DrafterCandidate> {
        let Ok(installed) = self.models().list_installed().await else {
            return Vec::new();
        };
        let mut packages = Vec::new();
        for installed in installed.packages {
            let package = installed.package;
            let architecture = package.properties.architecture.as_str();
            let drafter = (self.architectures.drafters.contains(architecture)
                && !self.architectures.targets.contains(architecture))
                || package
                    .files
                    .iter()
                    .any(|file| matches!(file.role, ModelFileRole::Draft | ModelFileRole::Mtp));
            if !drafter || !seen.insert(format!("package:{}", package.id.0)) {
                continue;
            }
            let Some(reference) = installed_reference(&package, &installed.path) else {
                continue;
            };
            if !seen.insert(reference_key(&reference)) {
                continue;
            }
            packages.push((package, reference));
        }
        let mut pending = Vec::new();
        for (package, reference) in packages {
            pending.push(self.installed_drafter_candidate(target, related, package, reference));
        }
        let mut found = collect_candidates(pending).await;
        found.truncate(MAX_ON_THIS_COMPUTER_DRAFTERS);
        found
    }

    async fn installed_drafter_candidate(
        &self,
        target: &DrafterTarget,
        related: &BTreeSet<String>,
        package: ModelPackage,
        reference: DrafterReference,
    ) -> Option<DrafterCandidate> {
        let resolved = self.installed_package_material(&package).await.ok()?;
        let path = resolved_paths(&resolved.target_model).ok()?.0;
        let mut lineage = header_lineage(&path);
        if let ModelPackageSource::HuggingFace { repository, .. } = &package.source {
            lineage.extend(self.lineage.hub().lineage(repository, None).await);
        }
        if lineage.is_disjoint(related) {
            return None;
        }
        let material = DrafterMaterial {
            path,
            identity: format!("package:{}", package.id.0),
            _guards: vec![Arc::new(resolved)],
        };
        let verdict = self.drafter_verdict(target, &material).await?;
        Some(DrafterCandidate {
            category: DrafterCategory::OnThisComputer,
            reference,
            label: "On this computer".to_owned(),
            size_bytes: package_bytes(&package),
            lineage: DrafterLineage::Lineage,
            method: bound_method(&verdict),
            verdict,
            on_this_computer: true,
        })
    }

    /// A drafter file of a Hugging Face snapshot, with the engine's verdict for the target.
    async fn hugging_face_drafter_candidate(
        &self,
        target: &DrafterTarget,
        snapshot: &HuggingFaceRepositorySnapshot,
        path: PathBuf,
        category: DrafterCategory,
        label: String,
    ) -> Option<DrafterCandidate> {
        let file = snapshot_file(snapshot, &path).ok()?;
        let material = self.hugging_face_drafter(snapshot, &path).await.ok()?;
        let verdict = self.drafter_verdict(target, &material).await?;
        Some(DrafterCandidate {
            category,
            reference: DrafterReference::HuggingFace {
                repository: hugging_face_id(&snapshot.repository),
                commit: snapshot.commit.clone(),
                path: path_string(&path),
            },
            label,
            size_bytes: file.size_bytes,
            lineage: DrafterLineage::Lineage,
            method: bound_method(&verdict),
            verdict,
            on_this_computer: sha256_of(file)
                .is_some_and(|sha256| self.models().has_installed_file(sha256)),
        })
    }

    async fn local_drafter_candidate(
        &self,
        target: &DrafterTarget,
        path: PathBuf,
        size_bytes: u64,
    ) -> Option<DrafterCandidate> {
        let material = self.local_drafter(path.clone()).ok()?;
        let verdict = self.drafter_verdict(target, &material).await?;
        Some(DrafterCandidate {
            category: DrafterCategory::NextToModel,
            reference: DrafterReference::Local { path },
            label: "In this folder".to_owned(),
            size_bytes,
            lineage: DrafterLineage::Lineage,
            method: bound_method(&verdict),
            verdict,
            on_this_computer: true,
        })
    }

    /// Third-party drafters on Hugging Face for the target's bases, found by `base_model` and the
    /// `speculative-decoding` tag.
    async fn hub_drafters(
        &self,
        target: &DrafterTarget,
        related: &BTreeSet<String>,
        seen: &mut BTreeSet<String>,
    ) -> Vec<DrafterCandidate> {
        let catalog_repositories = self.lineage.repositories();
        let own = target
            .repository
            .as_ref()
            .map(|(repository, _)| repository.to_lowercase());
        let bases = related
            .iter()
            .filter(|base| Some(*base) != own.as_ref() && !catalog_repositories.contains(*base))
            .take(MAX_HUB_DRAFTER_BASES)
            .cloned()
            .collect::<Vec<_>>();
        let mut repositories = Vec::new();
        for base in bases {
            match self.models().search_speculative_drafters(&base).await {
                Ok(found) => repositories.extend(found),
                Err(error) => tracing::warn!(%error, base = %base, "Hugging Face drafter search failed"),
            }
        }
        let mut unique = BTreeSet::new();
        let repositories = repositories
            .into_iter()
            .filter(|repository| {
                let key = repository.to_lowercase();
                Some(&key) != own.as_ref()
                    && !catalog_repositories.contains(&key)
                    && unique.insert(key)
            })
            .take(MAX_HUB_DRAFTER_REPOSITORIES)
            .collect::<Vec<_>>();
        let mut files = Vec::new();
        for repository in repositories {
            let Ok(snapshot) = self.snapshot(&repository, "main").await else {
                continue;
            };
            let Ok(classified) = self.classify_hugging_face(&snapshot).await else {
                continue;
            };
            let mut drafters = classified
                .into_iter()
                .filter(|group| matches!(group.role, CandidateFileRole::Drafter { .. }))
                .filter(|group| group.group.files.len() == 1)
                .collect::<Vec<_>>();
            drafters.sort_by_key(|group| group.group.size_bytes());
            for group in drafters.into_iter().take(MAX_HUB_DRAFTER_FILES) {
                let reference = DrafterReference::HuggingFace {
                    repository: hugging_face_id(&snapshot.repository),
                    commit: snapshot.commit.clone(),
                    path: path_string(&group.group.primary().path),
                };
                if seen.insert(reference_key(&reference)) {
                    files.push((snapshot.clone(), group.group.primary().path.clone()));
                }
            }
        }
        let mut pending = Vec::new();
        for (snapshot, path) in &files {
            let owner = snapshot.repository.split('/').next().unwrap_or_default();
            pending.push(self.hugging_face_drafter_candidate(
                target,
                snapshot,
                path.clone(),
                DrafterCategory::HuggingFace,
                format!("By {owner}"),
            ));
        }
        collect_candidates(pending).await
    }

    /// An explicit drafter: a Hugging Face reference (`owner/repo`, `owner/repo:path`,
    /// `owner/repo@revision:path` or a model URL) or a local file.
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
            let lineage = header_lineage(&material.path);
            let verdict = self
                .drafter_verdict(target, &material)
                .await
                .ok_or_else(|| InventoryError::Internal("drafter assessment failed".to_owned()))?;
            return Ok(DrafterCandidate {
                category: DrafterCategory::Other,
                reference: DrafterReference::Local {
                    path: local.to_path_buf(),
                },
                label: local.display().to_string(),
                size_bytes: std::fs::metadata(local).map(|metadata| metadata.len()).unwrap_or(0),
                lineage: lineage_kind(&lineage, related),
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
                let classified = self.classify_hugging_face(&snapshot).await?;
                classified
                    .into_iter()
                    .filter(|group| matches!(group.role, CandidateFileRole::Drafter { .. }))
                    .filter(|group| group.group.files.len() == 1)
                    .min_by_key(|group| group.group.size_bytes())
                    .map(|group| group.group.primary().path.clone())
                    .ok_or_else(|| {
                        InventoryError::InvalidRequest(format!(
                            "{repository} has no drafter GGUF"
                        ))
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
        lineage.extend(header_lineage(&material.path));
        let verdict = self
            .drafter_verdict(target, &material)
            .await
            .ok_or_else(|| InventoryError::Internal("drafter assessment failed".to_owned()))?;
        Ok(DrafterCandidate {
            category: DrafterCategory::Other,
            reference: DrafterReference::HuggingFace {
                repository: hugging_face_id(&snapshot.repository),
                commit: snapshot.commit.clone(),
                path: path_string(&path),
            },
            label: format!("{}:{}", snapshot.repository, path.display()),
            size_bytes: file.size_bytes,
            lineage: lineage_kind(&lineage, related),
            method: bound_method(&verdict),
            verdict,
            on_this_computer: sha256_of(file)
                .is_some_and(|sha256| self.models().has_installed_file(sha256)),
        })
    }

    async fn drafter_verdict(
        &self,
        target: &DrafterTarget,
        drafter: &DrafterMaterial,
    ) -> Option<DrafterVerdict> {
        self.assess(&target.material, Some(drafter), ModelMethod::Auto)
            .await
            .map(|assessed| verdict_of(&assessed.profile))
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
                    self.hugging_face_drafter(&snapshot, Path::new(path)).await?,
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
        Ok(self.models().hugging_face_package(&source, snapshot).await?.0)
    }

    /// The downloadable package of a drafter, with the method its bundle declares.
    async fn drafter_package(
        &self,
        drafter: &CandidateDrafter,
    ) -> Result<Option<(SpeculativeDraftSource, SpeculativeMethod)>, InventoryError> {
        match &drafter.reference {
            DrafterReference::Embedded => {
                Ok(Some((SpeculativeDraftSource::Embedded, SpeculativeMethod::Mtp)))
            }
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

impl ModelCandidates for ManagedModelCandidates {
    fn inspect(
        &self,
        request: CandidateInspectionRequest,
    ) -> BoxFuture<'_, Result<CandidateInspection, InventoryError>> {
        Box::pin(async move {
            match request {
                CandidateInspectionRequest::HuggingFace {
                    repository,
                    revision,
                } => self.inspect_hugging_face(repository, revision).await,
                CandidateInspectionRequest::Local { path } => self.inspect_local(path).await,
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
                    (self.model_target(&model_id).await?, other)
                }
            };
            Ok(DrafterCandidatesResponse {
                candidates: self
                    .enumerate_drafters(target, other.as_deref(), false)
                    .await?,
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
) -> Vec<DrafterCandidate> {
    futures_util::stream::iter(pending)
        .buffered(ASSESSMENT_CONCURRENCY)
        .filter_map(|candidate| async move { candidate })
        .collect()
        .await
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
        ModelAssessment::Unsupported { reasons, detail, .. } => DrafterVerdict::Rejected {
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

/// Within one category: a lineage match before a fingerprint-only match, then the smaller
/// download, then the label for a deterministic order.
fn sorted(mut candidates: Vec<DrafterCandidate>) -> Vec<DrafterCandidate> {
    candidates.sort_by(|left, right| {
        (left.category, left.lineage, left.size_bytes, &left.label).cmp(&(
            right.category,
            right.lineage,
            right.size_bytes,
            &right.label,
        ))
    });
    candidates
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

/// The catalog drafter's file as a Hugging Face reference key, so the same file found elsewhere
/// is not listed twice.
fn catalog_draft_key(model: &magnitude_service_contracts::models::RecommendableModel) -> Option<String> {
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
    Some(reference_key(&DrafterReference::HuggingFace {
        repository: hugging_face_id(repository),
        commit: revision.clone(),
        path: path_string(&path),
    }))
}

fn hugging_face_id(repository: &str) -> HuggingFaceRepositoryId {
    HuggingFaceRepositoryId::new(repository.to_owned())
        .unwrap_or_else(|_| HuggingFaceRepositoryId::new("invalid/repository").expect("valid id"))
}

fn path_string(path: &Path) -> String {
    path.to_string_lossy().replace('\\', "/")
}

fn installed_reference(package: &ModelPackage, root: &Path) -> Option<DrafterReference> {
    let path = package
        .files
        .iter()
        .filter(|file| {
            matches!(
                file.role,
                ModelFileRole::Weights | ModelFileRole::Draft | ModelFileRole::Mtp
            )
        })
        .map(|file| file.path.clone())
        .min()?;
    Some(match &package.source {
        ModelPackageSource::HuggingFace {
            repository,
            revision,
        } => DrafterReference::HuggingFace {
            repository: HuggingFaceRepositoryId::new(repository.clone()).ok()?,
            commit: revision.clone(),
            path: path_string(&path),
        },
        ModelPackageSource::Local { .. } => DrafterReference::Local {
            path: if path.is_absolute() { path } else { root.join(path) },
        },
    })
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

/// The repositories a local GGUF names as its base models.
fn header_lineage(path: &Path) -> BTreeSet<String> {
    local_directory(path)
        .map(|directory| header_base_repositories(&directory))
        .unwrap_or_default()
}

/// GGUFs beneath `root`, relative to it, at most `depth` directories deep. Partial downloads and
/// hidden files are skipped.
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
            } else if file_type.is_file()
                && name.to_ascii_lowercase().ends_with(".gguf")
                && let (Ok(relative), Ok(metadata)) = (path.strip_prefix(root), entry.metadata())
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

fn local_model_id(header_digest: &str) -> Option<ModelId> {
    magnitude_service_contracts::models::LocalModelDigest::new(header_digest.to_owned())
        .ok()
        .map(|digest| ModelId::local(&digest))
}

/// The custom model already on this computer with the same bytes: the same header digest, the
/// same Hugging Face identity and size, or the same files.
fn on_this_computer(
    custom: &[CustomModel],
    hugging_face: Option<&ModelId>,
    group: &GgufGroup,
    header_digest: &str,
    root: Option<&Path>,
) -> Option<ModelId> {
    let local = local_model_id(header_digest);
    let paths = root
        .map(|root| {
            group
                .files
                .iter()
                .map(|file| root.join(&file.path))
                .collect::<BTreeSet<_>>()
        })
        .unwrap_or_default();
    custom
        .iter()
        .filter(|model| !matches!(model.state, CustomModelState::NotInstalled))
        .find(|model| {
            Some(&model.model_id) == local.as_ref()
                || (Some(&model.model_id) == hugging_face && model.size_bytes == group.size_bytes())
                || (!paths.is_empty() && model.files.iter().any(|file| paths.contains(&file.path)))
        })
        .map(|model| model.model_id.clone())
}

/// Versions with their verdicts. The recommended version is the largest fitting Q4_K-class
/// version, else the largest fitting version.
fn candidate_versions(versions: Vec<AssessedVersion<'_>>) -> Vec<CandidateVersion> {
    let fits = |assessment: &Option<CachedModelAssessment>| {
        matches!(
            assessment.as_ref().map(|assessment| &assessment.profile),
            Some(ModelAssessment::Fits { .. })
        )
    };
    let recommended = versions
        .iter()
        .enumerate()
        .filter(|(_, version)| fits(&version.assessment))
        .max_by_key(|(_, version)| {
            (
                version.group.quantization.as_deref().is_some_and(is_q4_k_class),
                version.group.group.size_bytes(),
            )
        })
        .map(|(index, _)| index);
    let mut labels = BTreeMap::<String, usize>::new();
    for version in &versions {
        *labels.entry(version_label(&version.group.group.primary().path)).or_default() += 1;
    }
    versions
        .into_iter()
        .enumerate()
        .map(|(index, version)| {
            let group = version.group;
            let label = version_label(&group.group.primary().path);
            let label = if labels.get(&label).copied().unwrap_or(0) > 1 {
                path_string(&group.group.primary().path)
            } else {
                label
            };
            CandidateVersion {
                label,
                model_id: version.model_id,
                files: group.group.files.iter().map(candidate_file).collect(),
                size_bytes: group.group.size_bytes(),
                assessment: version.assessment.map(|assessment| assessment.profile),
                recommended: Some(index) == recommended,
                on_this_computer: version.on_this_computer,
            }
        })
        .collect()
}

/// `(repository, revision, path)` from `owner/repo`, `owner/repo:path`, `owner/repo@rev:path` or a
/// Hugging Face URL (`…/blob/<rev>/<path>` or `…/resolve/<rev>/<path>` included).
fn parse_hugging_face_reference(
    reference: &str,
) -> Result<(String, String, Option<String>), InventoryError> {
    let invalid = || {
        InventoryError::InvalidRequest(format!(
            "{reference} is not a Hugging Face reference or an absolute file path"
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
            Some("blob" | "resolve") if parts.len() > 4 => Ok((
                original,
                parts[3].to_owned(),
                Some(parts[4..].join("/")),
            )),
            _ => Ok((original, "main".to_owned(), None)),
        };
    }
    let (repository, path) = match reference.split_once(':') {
        Some((repository, path)) => (repository, Some(path.to_owned())),
        None => (reference, None),
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

    #[test]
    fn hugging_face_references_parse_every_accepted_form() {
        assert_eq!(
            parse_hugging_face_reference("z-lab/Qwen3.6-35B-A3B-DFlash").unwrap(),
            ("z-lab/Qwen3.6-35B-A3B-DFlash".to_owned(), "main".to_owned(), None)
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
            parse_hugging_face_reference(
                "https://huggingface.co/owner/repo/blob/main/draft.gguf"
            )
            .unwrap(),
            (
                "owner/repo".to_owned(),
                "main".to_owned(),
                Some("draft.gguf".to_owned())
            )
        );
        assert!(parse_hugging_face_reference("not a reference").is_err());
    }

    #[test]
    fn within_a_category_lineage_precedes_fingerprint_then_the_smaller_download() {
        let candidate = |lineage, size_bytes, label: &str| DrafterCandidate {
            category: DrafterCategory::OnThisComputer,
            reference: DrafterReference::Embedded,
            label: label.to_owned(),
            size_bytes,
            lineage,
            verdict: DrafterVerdict::None,
            on_this_computer: true,
            method: None,
        };
        let ordered = sorted(vec![
            candidate(DrafterLineage::Fingerprint, 1, "a"),
            candidate(DrafterLineage::Lineage, 9, "b"),
            candidate(DrafterLineage::Lineage, 3, "c"),
        ]);
        assert_eq!(
            ordered.iter().map(|candidate| candidate.label.as_str()).collect::<Vec<_>>(),
            ["c", "b", "a"]
        );
    }
}
