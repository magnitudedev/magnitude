use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use futures_util::future::BoxFuture;
use futures_util::{StreamExt, stream};
use icn_contracts::models::{
    CatalogDiagnostic, ModelFailure, ModelOfferingTarget, ModelOfferingTargetId,
    RecommendableModel, RecommendableModelCatalog, RecommendableModelCatalogProvider,
    RecommendableModelId, ResolvedModelTarget, ServingProfile,
};
use icn_contracts::{
    ContentId, HardwareAssessment, HuggingFaceRepositoryRequest, HuggingFaceRepositorySnapshot,
    Integrity, InventoryError, InventoryModel, InventoryProperties, ModelAvailability,
    ModelComponent, ModelId, ModelLocation, ModelPreviewSource, ModelSource, ResolvedComponent,
    ResolvedModel,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::cache::ModelBlobKind;
use crate::capabilities::model_capabilities;
use crate::inventory::ModelManager;
use crate::package_service::{offering_target_id, package_from_resolved};
use crate::preview::ModelPreviewService;

#[path = "../../../catalog/planner_bundle.rs"]
mod planner_bundle;
use planner_bundle::PlannerBundle;

const CATALOG_SOURCE: &str = include_str!("../../../catalog/models.json");
const MAX_RELEASE_CATALOG_BYTES: u64 = 64 * 1024 * 1024;
const MAX_PLANNER_BUNDLE_BYTES: u64 = 2 * 1024 * 1024 * 1024;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CatalogSource {
    models: Vec<CatalogModel>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CatalogModel {
    id: String,
    display_name: String,
    description: String,
    repository: String,
    formats: Vec<String>,
    contexts: Vec<u32>,
    license: String,
    quality_score: f64,
    quality_score_provenance: String,
    quality_evidence: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ReleaseCatalogManifest {
    source_digest: String,
    template_identity: String,
    planner_identity: String,
    planner_bundle_digest: String,
    planner_artifacts: Vec<ReleasePlannerArtifact>,
    catalog: RecommendableModelCatalog,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ReleasePlannerArtifact {
    target_id: ModelOfferingTargetId,
    model_id: ModelId,
    content_id: ContentId,
    name: String,
    supported_parameters: Vec<String>,
    properties: InventoryProperties,
    source: ModelSource,
    primary_gguf: PathBuf,
    components: Vec<ReleasePlannerComponent>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ReleasePlannerComponent {
    component: ModelComponent,
    header_digest: String,
    header_size_bytes: u64,
}

pub struct GeneratedReleaseCatalog {
    pub catalog: RecommendableModelCatalog,
    planner_artifacts: Vec<ReleasePlannerArtifact>,
    headers: BTreeMap<String, Vec<u8>>,
}

#[derive(Clone)]
pub struct ReleaseCatalog {
    catalog: RecommendableModelCatalog,
    planner_artifacts: Arc<BTreeMap<ModelOfferingTargetId, ReleasePlannerArtifact>>,
    planner_bundle: Arc<PlannerBundle<'static>>,
}

fn catalog_source() -> Result<CatalogSource, InventoryError> {
    let source: CatalogSource = serde_json::from_str(CATALOG_SOURCE)
        .map_err(|error| InventoryError::Integrity(format!("invalid catalog source: {error}")))?;
    if source.models.is_empty() {
        return Err(InventoryError::Integrity(
            "catalog source must contain at least one model".to_owned(),
        ));
    }
    let mut ids = BTreeSet::new();
    for model in &source.models {
        let formats = model.formats.iter().collect::<BTreeSet<_>>();
        let contexts = model.contexts.iter().collect::<BTreeSet<_>>();
        if model.id.is_empty()
            || model.display_name.is_empty()
            || model.description.is_empty()
            || model.repository.is_empty()
            || model.formats.is_empty()
            || formats.len() != model.formats.len()
            || model.contexts.is_empty()
            || contexts.len() != model.contexts.len()
            || model.contexts.contains(&0)
            || model.license.is_empty()
            || !model.quality_score.is_finite()
            || model.quality_score < 0.0
            || model.quality_score_provenance.is_empty()
            || model.quality_evidence.is_empty()
            || !ids.insert(model.id.as_str())
        {
            return Err(InventoryError::Integrity(format!(
                "invalid or duplicate catalog declaration {}",
                model.id
            )));
        }
    }
    Ok(source)
}

pub fn catalog_source_digest() -> Result<String, InventoryError> {
    catalog_source_digest_from(CATALOG_SOURCE.as_bytes())
}

pub fn catalog_source_digest_from(bytes: &[u8]) -> Result<String, InventoryError> {
    let source: CatalogSource = serde_json::from_slice(bytes)
        .map_err(|error| InventoryError::Integrity(format!("invalid catalog source: {error}")))?;
    let canonical = serde_json::to_vec(&source)
        .map_err(|error| InventoryError::Internal(error.to_string()))?;
    Ok(format!("{:x}", Sha256::digest(canonical)))
}

pub fn encode_release_planner_bundle(
    headers: &BTreeMap<String, Vec<u8>>,
) -> Result<Vec<u8>, InventoryError> {
    encode_release_planner_bundle_with_progress(headers, |_, _| {})
}

pub fn encode_release_planner_bundle_with_progress(
    headers: &BTreeMap<String, Vec<u8>>,
    progress: impl FnMut(usize, usize),
) -> Result<Vec<u8>, InventoryError> {
    planner_bundle::encode(headers, progress).map_err(InventoryError::Integrity)
}

pub fn release_catalog_manifest(
    generated: &GeneratedReleaseCatalog,
    template_identity: impl Into<String>,
    planner_identity: impl Into<String>,
    planner_bundle: &[u8],
) -> Result<ReleaseCatalogManifest, InventoryError> {
    let source = catalog_source()?;
    validate_resolved_catalog(&generated.catalog, &source)?;
    validate_planner_artifacts(&generated.catalog, &generated.planner_artifacts)?;
    Ok(ReleaseCatalogManifest {
        source_digest: catalog_source_digest()?,
        template_identity: template_identity.into(),
        planner_identity: planner_identity.into(),
        planner_bundle_digest: format!("{:x}", Sha256::digest(planner_bundle)),
        planner_artifacts: generated.planner_artifacts.clone(),
        catalog: generated.catalog.clone(),
    })
}

pub fn load_release_catalog(
    catalog_path: &Path,
    planner_bundle_path: &Path,
    template_identity: &str,
    planner_identity: &str,
) -> Result<ReleaseCatalog, InventoryError> {
    let catalog_bytes = read_bounded_regular_file(catalog_path, MAX_RELEASE_CATALOG_BYTES)?;
    let manifest: ReleaseCatalogManifest =
        serde_json::from_slice(&catalog_bytes).map_err(|error| {
            InventoryError::Integrity(format!("invalid release catalog: {error}"))
        })?;
    if manifest.template_identity != template_identity {
        return Err(InventoryError::Integrity(
            "release catalog was generated with a different template assessor".to_owned(),
        ));
    }
    if manifest.planner_identity != planner_identity {
        return Err(InventoryError::Integrity(
            "release catalog was generated for a different native planner".to_owned(),
        ));
    }
    let planner_bundle_bytes =
        read_bounded_regular_file(planner_bundle_path, MAX_PLANNER_BUNDLE_BYTES)?;
    if manifest.planner_bundle_digest != planner_bundle::sha256(&planner_bundle_bytes) {
        return Err(InventoryError::Integrity(
            "model planner input bundle does not match the release catalog".to_owned(),
        ));
    }
    validate_runtime_catalog(&manifest.catalog)?;
    validate_planner_artifacts(&manifest.catalog, &manifest.planner_artifacts)?;
    // The catalog is loaded once for the process lifetime. Retaining its immutable bytes lets the
    // indexed bundle lazily decompress individual headers without copying the complete bundle.
    let planner_bundle_bytes: &'static [u8] =
        Box::leak(planner_bundle_bytes.into_boxed_slice());
    let planner_bundle =
        PlannerBundle::parse(planner_bundle_bytes).map_err(InventoryError::Integrity)?;
    for artifact in &manifest.planner_artifacts {
        for component in &artifact.components {
            if !planner_bundle.contains(&component.header_digest) {
                return Err(InventoryError::Integrity(format!(
                    "model planner input bundle is missing header {}",
                    component.header_digest
                )));
            }
        }
    }
    Ok(ReleaseCatalog {
        catalog: manifest.catalog,
        planner_artifacts: Arc::new(
            manifest
                .planner_artifacts
                .into_iter()
                .map(|artifact| (artifact.target_id.clone(), artifact))
                .collect(),
        ),
        planner_bundle: Arc::new(planner_bundle),
    })
}

fn read_bounded_regular_file(path: &Path, maximum: u64) -> Result<Vec<u8>, InventoryError> {
    let metadata =
        fs::symlink_metadata(path).map_err(|error| InventoryError::Io(error.to_string()))?;
    if !metadata.is_file() || metadata.file_type().is_symlink() || metadata.len() > maximum {
        return Err(InventoryError::Integrity(format!(
            "{} is not a bounded regular release file",
            path.display()
        )));
    }
    fs::read(path).map_err(|error| InventoryError::Io(error.to_string()))
}

fn validate_runtime_catalog(catalog: &RecommendableModelCatalog) -> Result<(), InventoryError> {
    if !catalog.diagnostics.is_empty() {
        return Err(InventoryError::Integrity(format!(
            "release catalog contains {} unresolved entries",
            catalog.diagnostics.len()
        )));
    }
    let model_ids = catalog
        .models
        .iter()
        .map(|model| model.id.clone())
        .collect::<BTreeSet<_>>();
    let target_ids = catalog
        .models
        .iter()
        .map(|model| model.target_id.clone())
        .collect::<BTreeSet<_>>();
    if catalog.models.is_empty()
        || model_ids.len() != catalog.models.len()
        || target_ids.len() != catalog.models.len()
    {
        return Err(InventoryError::Integrity(
            "release catalog has missing or duplicate model identities".to_owned(),
        ));
    }
    Ok(())
}

fn validate_planner_artifacts(
    catalog: &RecommendableModelCatalog,
    artifacts: &[ReleasePlannerArtifact],
) -> Result<(), InventoryError> {
    let catalog_targets = catalog
        .models
        .iter()
        .map(|model| model.target_id.clone())
        .collect::<BTreeSet<_>>();
    let artifact_targets = artifacts
        .iter()
        .map(|artifact| artifact.target_id.clone())
        .collect::<BTreeSet<_>>();
    if artifact_targets.len() != artifacts.len() || artifact_targets != catalog_targets {
        return Err(InventoryError::Integrity(
            "release planner artifacts do not exactly cover the catalog targets".to_owned(),
        ));
    }
    for artifact in artifacts {
        if artifact.components.is_empty()
            || !artifact
                .components
                .iter()
                .any(|component| component.component.path == artifact.primary_gguf)
            || artifact.components.iter().any(|component| {
                component.header_digest.len() != 64
                    || component.header_size_bytes == 0
                    || !component
                        .header_digest
                        .bytes()
                        .all(|byte| byte.is_ascii_hexdigit())
            })
        {
            return Err(InventoryError::Integrity(format!(
                "invalid release planner artifact {}",
                artifact.target_id.0
            )));
        }
    }
    Ok(())
}

impl GeneratedReleaseCatalog {
    pub fn encode_planner_bundle(&self) -> Result<Vec<u8>, InventoryError> {
        encode_release_planner_bundle(&self.headers)
    }
}

impl ReleaseCatalog {
    #[must_use]
    pub fn catalog(&self) -> &RecommendableModelCatalog {
        &self.catalog
    }

    pub fn resolve_target(
        &self,
        target_id: &ModelOfferingTargetId,
    ) -> Result<Option<ResolvedModelTarget>, InventoryError> {
        let Some(artifact) = self.planner_artifacts.get(target_id) else {
            return Ok(None);
        };
        let target = self
            .catalog
            .models
            .iter()
            .find(|model| &model.target_id == target_id)
            .map(|model| model.target.clone())
            .ok_or_else(|| {
                InventoryError::Integrity(format!(
                    "catalog target {} has no model declaration",
                    target_id.0
                ))
            })?;
        let workspace =
            tempfile::tempdir().map_err(|error| InventoryError::Io(error.to_string()))?;
        for component in &artifact.components {
            let header = self
                .planner_bundle
                .header(&component.header_digest)
                .map_err(InventoryError::Integrity)?;
            let path = workspace.path().join(&component.component.path);
            let parent = path.parent().ok_or_else(|| {
                InventoryError::Integrity("planner component has no parent".to_owned())
            })?;
            fs::create_dir_all(parent).map_err(|error| InventoryError::Io(error.to_string()))?;
            let mut file = OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&path)
                .map_err(|error| InventoryError::Io(error.to_string()))?;
            file.write_all(&header)
                .and_then(|()| file.set_len(component.component.size_bytes))
                .map_err(|error| InventoryError::Io(error.to_string()))?;
        }
        let components = artifact
            .components
            .iter()
            .map(|component| component.component.clone())
            .collect::<Vec<_>>();
        let location = ModelLocation::Directory {
            source_id: "release_catalog".to_owned(),
            root: workspace.path().to_path_buf(),
            components: components.clone(),
            total_bytes: components
                .iter()
                .map(|component| component.size_bytes)
                .sum(),
            integrity: Integrity::Verified {
                method: "release_catalog_header_digest".to_owned(),
            },
        };
        let model = InventoryModel {
            id: artifact.model_id.clone(),
            content_id: artifact.content_id.clone(),
            created: 0,
            name: artifact.name.clone(),
            supported_parameters: artifact.supported_parameters.clone(),
            serving_configuration: None,
            availability: ModelAvailability::Available { ready_at: 0 },
            source: artifact.source.clone(),
            location,
            properties: artifact.properties.clone(),
            hardware: HardwareAssessment::NotAssessed {
                reason: "release planner input".to_owned(),
            },
            operations: Vec::new(),
            updated_at: 0,
        };
        let resolved = ResolvedModel {
            model,
            components: artifact
                .components
                .iter()
                .map(|component| ResolvedComponent {
                    path: workspace.path().join(&component.component.path),
                    role: component.component.role.clone(),
                    shard_index: component.component.shard_index,
                    relationship: component.component.relationship.clone(),
                })
                .collect(),
        };
        Ok(Some(
            ResolvedModelTarget::new(target_id.clone(), target, resolved, None)
                .retain_resolution_guard(workspace),
        ))
    }
}

fn validate_resolved_catalog(
    catalog: &RecommendableModelCatalog,
    source: &CatalogSource,
) -> Result<(), InventoryError> {
    if !catalog.diagnostics.is_empty() {
        return Err(InventoryError::Integrity(format!(
            "release catalog contains {} unresolved entries",
            catalog.diagnostics.len()
        )));
    }
    let actual = catalog
        .models
        .iter()
        .map(|model| model.id.0.as_str())
        .collect::<BTreeSet<_>>();
    let expected = source
        .models
        .iter()
        .flat_map(|model| {
            model
                .formats
                .iter()
                .map(|format| format!("{}:{format}", model.id))
        })
        .collect::<BTreeSet<_>>();
    if actual.len() != catalog.models.len()
        || actual.len() != expected.len()
        || !expected.iter().all(|id| actual.contains(id.as_str()))
    {
        return Err(InventoryError::Integrity(
            "release catalog does not exactly cover its source declarations".to_owned(),
        ));
    }
    Ok(())
}

fn fidelity(declaration_id: &str, format: &str) -> (u32, bool) {
    if declaration_id.starts_with("gemma-4-") {
        return (58, true);
    }
    if declaration_id == "nemotron-3-super-120b-a12b"
        || declaration_id == "nemotron-3-ultra-550b-a55b"
    {
        return (58, true);
    }
    if declaration_id == "glm-5.2" {
        return (100, false);
    }
    let rank = if format.contains("Q8") {
        80
    } else if format.contains("Q6") {
        60
    } else if format.contains("Q5") {
        50
    } else {
        40
    };
    (rank, false)
}

pub struct ResolvingRecommendableCatalog {
    models: Arc<ModelManager>,
    hugging_face: Arc<ModelPreviewService>,
}

impl ResolvingRecommendableCatalog {
    #[must_use]
    pub fn new(models: Arc<ModelManager>, hugging_face: Arc<ModelPreviewService>) -> Self {
        Self {
            models,
            hugging_face,
        }
    }

    async fn resolve_model(
        &self,
        declaration: &CatalogModel,
        format: &str,
        snapshot: &HuggingFaceRepositorySnapshot,
    ) -> Result<
        (
            RecommendableModel,
            ReleasePlannerArtifact,
            BTreeMap<String, Vec<u8>>,
        ),
        InventoryError,
    > {
        let selector = format.to_ascii_lowercase();
        let mut matches = snapshot
            .gguf_files
            .iter()
            .filter(|file| {
                let path = file.path.to_string_lossy().to_ascii_lowercase();
                let basename = path.rsplit('/').next().unwrap_or(path.as_str());
                path.contains(&selector)
                    && !basename.starts_with("mmproj-")
                    && !basename.contains("imatrix")
                    && (!is_later_shard(basename) || is_first_shard(basename))
            })
            .collect::<Vec<_>>();
        if matches.len() != 1 {
            return Err(InventoryError::Integrity(format!(
                "{} format {format} resolved to {} primary files",
                declaration.repository,
                matches.len()
            )));
        }
        let primary = matches.remove(0);
        let prepared = self
            .models
            .prepare_preview_from_repository_snapshot(
                &ModelPreviewSource {
                    repository: snapshot.repository.clone(),
                    revision: snapshot.commit.clone(),
                    primary_gguf: primary.path.clone(),
                    additional_components: Vec::new(),
                },
                snapshot,
            )
            .await?;
        let package = package_from_resolved(&prepared.model)?;
        let capabilities = model_capabilities(&prepared.model.model.properties);
        let (fidelity_rank, quantization_aware) = fidelity(&declaration.id, format);
        let target_id = offering_target_id(&[&package.id]);
        let model = RecommendableModel {
            id: RecommendableModelId(format!("{}:{format}", declaration.id)),
            checkpoint_id: declaration.id.clone(),
            target_id,
            target: ModelOfferingTarget::Package { package },
            eligible_serving_profiles: declaration
                .contexts
                .iter()
                .map(|context_length| ServingProfile {
                    context_length: *context_length,
                })
                .collect(),
            display_name: declaration.display_name.clone(),
            description: declaration.description.clone(),
            license: snapshot
                .license
                .clone()
                .filter(|license| license != "other")
                .unwrap_or_else(|| declaration.license.clone()),
            capabilities,
            quality_score: declaration.quality_score,
            quality_score_provenance: declaration.quality_score_provenance.clone(),
            fidelity_rank,
            quantization_aware,
            quality_evidence: declaration.quality_evidence.clone(),
        };
        let headers = prepared
            .headers
            .iter()
            .map(|header| {
                self.models
                    .cache
                    .read_blob(ModelBlobKind::GgufHeader, &header.digest)
                    .map(|bytes| (header.digest.clone(), bytes))
                    .ok_or_else(|| {
                        InventoryError::Integrity(format!(
                            "catalog generation lost planner header {}",
                            header.digest
                        ))
                    })
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        let planner = ReleasePlannerArtifact {
            target_id: model.target_id.clone(),
            model_id: prepared.model.model.id.clone(),
            content_id: prepared.model.model.content_id.clone(),
            name: prepared.model.model.name.clone(),
            supported_parameters: prepared.model.model.supported_parameters.clone(),
            properties: prepared.model.model.properties.clone(),
            source: prepared.model.model.source.clone(),
            primary_gguf: primary.path.clone(),
            components: prepared
                .components
                .iter()
                .map(|component| {
                    let header = prepared
                        .headers
                        .iter()
                        .find(|header| header.path == component.path)
                        .ok_or_else(|| {
                            InventoryError::Integrity(format!(
                                "catalog component {} has no planner header",
                                component.path.display()
                            ))
                        })?;
                    Ok(ReleasePlannerComponent {
                        component: component.clone(),
                        header_digest: header.digest.clone(),
                        header_size_bytes: u64::try_from(
                            headers
                                .get(&header.digest)
                                .ok_or_else(|| {
                                    InventoryError::Integrity(format!(
                                        "catalog generation lost planner header {}",
                                        header.digest
                                    ))
                                })?
                                .len(),
                        )
                        .map_err(|_| {
                            InventoryError::Integrity("planner header is too large".to_owned())
                        })?,
                    })
                })
                .collect::<Result<Vec<_>, InventoryError>>()?,
        };
        Ok((model, planner, headers))
    }
}

fn is_first_shard(name: &str) -> bool {
    name.rsplit_once("-00001-of-")
        .is_some_and(|(_, suffix)| suffix.ends_with(".gguf"))
}

fn is_later_shard(name: &str) -> bool {
    let Some(stem) = name.strip_suffix(".gguf") else {
        return false;
    };
    stem.rsplit_once("-of-")
        .and_then(|(prefix, count)| prefix.rsplit_once('-').map(|(_, index)| (index, count)))
        .is_some_and(|(index, count)| {
            index.len() == 5
                && count.len() == 5
                && index.bytes().all(|byte| byte.is_ascii_digit())
                && count.bytes().all(|byte| byte.is_ascii_digit())
                && index != "00001"
        })
}

impl ResolvingRecommendableCatalog {
    pub fn resolve_release_catalog(
        &self,
    ) -> BoxFuture<'_, Result<GeneratedReleaseCatalog, InventoryError>> {
        Box::pin(async move {
            let source = catalog_source()?;
            let repositories = source
                .models
                .iter()
                .map(|declaration| declaration.repository.clone())
                .fold(Vec::new(), |mut unique, repository| {
                    if !unique.contains(&repository) {
                        unique.push(repository);
                    }
                    unique
                });
            let snapshots = stream::iter(repositories)
                .map(|repository| async move {
                    let result = self
                        .hugging_face
                        .refresh_repository(HuggingFaceRepositoryRequest {
                            repository: repository.clone(),
                            revision: "main".to_owned(),
                        })
                        .await;
                    (repository, result)
                })
                .buffer_unordered(12)
                .collect::<Vec<_>>()
                .await;
            let mut resolved_snapshots = BTreeMap::new();
            let mut snapshot_failures = BTreeMap::new();
            for (repository, result) in snapshots {
                match result {
                    Ok(snapshot) => {
                        resolved_snapshots.insert(repository, snapshot);
                    }
                    Err(error) => {
                        snapshot_failures.insert(repository, error.to_string());
                    }
                }
            }
            let resolved_snapshots = &resolved_snapshots;
            let snapshot_failures = &snapshot_failures;
            let resolved = stream::iter(source.models.into_iter().enumerate())
                .map(|(declaration_index, declaration)| async move {
                    let mut formats = Vec::with_capacity(declaration.formats.len());
                    for (format_index, format) in declaration.formats.iter().enumerate() {
                        let result = match resolved_snapshots.get(&declaration.repository) {
                            Some(snapshot) => {
                                self.resolve_model(&declaration, format, snapshot).await
                            }
                            None => Err(InventoryError::Io(
                                snapshot_failures
                                    .get(&declaration.repository)
                                    .cloned()
                                    .unwrap_or_else(|| {
                                        format!(
                                            "repository {} was not resolved",
                                            declaration.repository
                                        )
                                    }),
                            )),
                        };
                        formats.push((
                            declaration_index,
                            format_index,
                            declaration.clone(),
                            format.clone(),
                            result,
                        ));
                    }
                    formats
                })
                .buffer_unordered(6)
                .flat_map(stream::iter)
                .collect::<Vec<_>>()
                .await;
            let mut resolved = resolved;
            resolved.sort_by_key(|(declaration_index, format_index, ..)| {
                (*declaration_index, *format_index)
            });
            let mut models = Vec::new();
            let mut planner_artifacts = Vec::new();
            let mut headers = BTreeMap::new();
            let mut diagnostics = Vec::new();
            for (_, _, declaration, format, result) in resolved {
                match result {
                    Ok((model, planner, model_headers)) => {
                        models.push(model);
                        planner_artifacts.push(planner);
                        for (digest, header) in model_headers {
                            if let Some(previous) = headers.insert(digest.clone(), header.clone())
                                && previous != header
                            {
                                return Err(InventoryError::Integrity(format!(
                                    "planner header digest collision {digest}"
                                )));
                            }
                        }
                    }
                    Err(error) => diagnostics.push(CatalogDiagnostic {
                        entry_id: Some(RecommendableModelId(format!(
                            "{}:{format}",
                            declaration.id
                        ))),
                        failure: ModelFailure {
                            code: "catalog_resolution_failed".to_owned(),
                            message: error.to_string(),
                            retryable: true,
                        },
                    }),
                }
            }
            Ok(GeneratedReleaseCatalog {
                catalog: RecommendableModelCatalog {
                    models,
                    diagnostics,
                },
                planner_artifacts,
                headers,
            })
        })
    }
}

impl RecommendableModelCatalogProvider for ResolvingRecommendableCatalog {
    fn catalog(&self) -> BoxFuture<'_, Result<RecommendableModelCatalog, InventoryError>> {
        Box::pin(async move { Ok(self.resolve_release_catalog().await?.catalog) })
    }
}

pub struct ReleaseRecommendableCatalog {
    catalog: RecommendableModelCatalog,
}

impl ReleaseRecommendableCatalog {
    #[must_use]
    pub fn new(catalog: RecommendableModelCatalog) -> Self {
        Self { catalog }
    }
}

impl RecommendableModelCatalogProvider for ReleaseRecommendableCatalog {
    fn catalog(&self) -> BoxFuture<'_, Result<RecommendableModelCatalog, InventoryError>> {
        Box::pin(async { Ok(self.catalog.clone()) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_selector_distinguishes_first_and_later_shards() {
        assert!(is_first_shard("model-00001-of-00003.gguf"));
        assert!(!is_later_shard("model-00001-of-00003.gguf"));
        assert!(is_later_shard("model-00002-of-00003.gguf"));
        assert!(!is_later_shard("model.gguf"));
    }

    #[test]
    fn workstation_catalog_uses_published_gguf_format_names() {
        let formats = |id: &str| {
            catalog_source()
                .expect("catalog source")
                .models
                .iter()
                .find(|model| model.id == id)
                .expect("catalog model")
                .formats
                .clone()
        };
        assert_eq!(
            formats("laguna-s-2.1"),
            ["UD-Q4_K_XL", "UD-Q6_K_XL", "UD-Q8_K_XL"]
        );
        assert_eq!(formats("qwen3.5-122b-a10b"), ["Q4_K_M"]);
        assert_eq!(
            formats("nemotron-3-super-120b-a12b"),
            ["UD-Q4_K_XL", "MXFP4_MOE"]
        );
        assert_eq!(formats("deepseek-v4-flash"), ["UD-Q3_K_M"]);
        assert_eq!(formats("glm-5.2"), ["BF16"]);
    }

    #[test]
    fn source_digest_is_stable_and_nonempty() {
        assert_eq!(catalog_source_digest().expect("source digest").len(), 64);
        assert!(!catalog_source().expect("catalog source").models.is_empty());
    }

}
