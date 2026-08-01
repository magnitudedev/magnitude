use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use futures_util::future::BoxFuture;
use futures_util::{StreamExt, stream};
use icn_contracts::models::{
    CatalogDiagnostic, ModelFailure, ModelOfferingTarget, ModelOfferingTargetId, ModelPackage,
    ModelPackageSource, RecommendableModel, RecommendableModelCatalog,
    RecommendableModelCatalogProvider, RecommendableModelId, ResolvedModelTarget, ServingProfile,
};
use icn_contracts::{
    ContentId, HardwareAssessment, HuggingFaceRepositoryRequest, HuggingFaceRepositorySnapshot,
    Integrity, InventoryError, InventoryModel, InventoryProperties, ModelAvailability,
    ModelComponent, ModelId, ModelLocation, ModelPreviewSource, ModelSource, ResolvedComponent,
    ResolvedModel,
};
use serde::{Deserialize, Serialize};

use crate::cache::ModelBlobKind;
use crate::capabilities::model_capabilities;
use crate::inventory::ModelManager;
use crate::package_service::{offering_target_id, package_from_resolved};
use crate::planner_stub::{PlannerStubComponent, compact_planner_stub, planner_stub_context};
use crate::refresh_hugging_face_repository;

#[path = "../../../catalog/planner_bundle.rs"]
mod planner_bundle;
use planner_bundle::PlannerBundle;

const CATALOG_SOURCE: &str = include_str!("../../../catalog/models.json");
const CATALOG_LOCK: &str = include_str!("../../../catalog/models.lock.json");
const MAX_PLANNER_BUNDLE_BYTES: u64 = 2 * 1024 * 1024 * 1024;

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CatalogSource {
    models: Vec<CatalogModel>,
}

#[derive(Clone, Deserialize)]
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
struct ReleasePlannerManifest {
    planner_inputs: BTreeMap<ModelOfferingTargetId, ReleasePlannerInput>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ReleasePlannerInput {
    model_id: RecommendableModelId,
    package: ModelPackage,
    properties: InventoryProperties,
    primary_gguf: PathBuf,
    components: Vec<ReleasePlannerComponent>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ReleasePlannerComponent {
    component: ModelComponent,
    source_header_digest: String,
    source_header_size_bytes: u64,
    planner_stub_digest: String,
    planner_stub_size_bytes: u64,
}

pub struct GeneratedReleaseCatalog {
    pub catalog: RecommendableModelCatalog,
    planner_inputs: BTreeMap<ModelOfferingTargetId, ReleasePlannerInput>,
    source_headers: BTreeMap<String, Vec<u8>>,
    planner_stubs: BTreeMap<String, Vec<u8>>,
}

#[derive(Clone)]
pub struct ReleaseCatalog {
    catalog: RecommendableModelCatalog,
    planner_inputs: Arc<BTreeMap<ModelOfferingTargetId, ReleasePlannerInput>>,
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

pub fn model_catalog_lock() -> Result<BTreeMap<String, String>, InventoryError> {
    model_catalog_lock_from(CATALOG_LOCK.as_bytes(), &catalog_source()?)
}

pub async fn advance_model_catalog_lock(
    models: Arc<ModelManager>,
) -> Result<BTreeMap<String, String>, InventoryError> {
    let source = catalog_source()?;
    let resolved = stream::iter(source.models)
        .map(|declaration| {
            let models = Arc::clone(&models);
            async move {
                let snapshot = refresh_hugging_face_repository(
                    &models,
                    HuggingFaceRepositoryRequest {
                        repository: declaration.repository,
                        revision: "main".to_owned(),
                    },
                )
                .await?;
                Ok::<_, InventoryError>((declaration.id, snapshot.commit))
            }
        })
        .buffer_unordered(12)
        .collect::<Vec<_>>()
        .await;
    resolved.into_iter().collect()
}

fn model_catalog_lock_from(
    bytes: &[u8],
    source: &CatalogSource,
) -> Result<BTreeMap<String, String>, InventoryError> {
    let lock: BTreeMap<String, String> = serde_json::from_slice(bytes).map_err(|error| {
        InventoryError::Integrity(format!("invalid model catalog lock: {error}"))
    })?;
    let expected = source
        .models
        .iter()
        .map(|model| model.id.clone())
        .collect::<BTreeSet<_>>();
    if lock.keys().cloned().collect::<BTreeSet<_>>() != expected {
        return Err(InventoryError::Integrity(
            "model catalog lock does not exactly cover models.json".to_owned(),
        ));
    }
    if lock.values().any(|commit| {
        commit.len() != 40
            || !commit
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    }) {
        return Err(InventoryError::Integrity(
            "model catalog lock contains a non-commit revision".to_owned(),
        ));
    }
    Ok(lock)
}

pub fn load_release_catalog(planner_bundle_path: &Path) -> Result<ReleaseCatalog, InventoryError> {
    let planner_bundle_bytes =
        read_bounded_regular_file(planner_bundle_path, MAX_PLANNER_BUNDLE_BYTES)?;
    // The catalog is loaded once for the process lifetime. Retaining its immutable bytes lets the
    // indexed bundle lazily decompress individual inputs without copying the complete bundle.
    let planner_bundle_bytes: &'static [u8] = Box::leak(planner_bundle_bytes.into_boxed_slice());
    let planner_bundle =
        PlannerBundle::parse(planner_bundle_bytes).map_err(InventoryError::Integrity)?;
    let manifest: ReleasePlannerManifest = serde_json::from_slice(planner_bundle.manifest())
        .map_err(|error| InventoryError::Integrity(format!("invalid planner manifest: {error}")))?;
    let source = catalog_source()?;
    let catalog = catalog_from_planner_inputs(&source, &manifest.planner_inputs)?;
    validate_runtime_catalog(&catalog)?;
    validate_resolved_catalog(&catalog, &source)?;
    validate_planner_inputs(&catalog, &manifest.planner_inputs)?;
    let mut expected_inputs = BTreeMap::new();
    for input in manifest.planner_inputs.values() {
        for component in &input.components {
            if let Some(previous_size) = expected_inputs.insert(
                component.planner_stub_digest.as_str(),
                component.planner_stub_size_bytes,
            ) && previous_size != component.planner_stub_size_bytes
            {
                return Err(InventoryError::Integrity(format!(
                    "planner input {} has inconsistent declared sizes",
                    component.planner_stub_digest
                )));
            }
        }
    }
    let bundled_inputs = planner_bundle.digests().collect::<BTreeSet<_>>();
    if bundled_inputs != expected_inputs.keys().copied().collect() {
        return Err(InventoryError::Integrity(
            "model planner input bundle does not exactly cover its manifest".to_owned(),
        ));
    }
    for (digest, expected_size) in expected_inputs {
        let input = planner_bundle
            .input(digest)
            .map_err(InventoryError::Integrity)?;
        if u64::try_from(input.len()).ok() != Some(expected_size) {
            return Err(InventoryError::Integrity(format!(
                "planner input {digest} does not match its manifest size"
            )));
        }
    }
    Ok(ReleaseCatalog {
        catalog,
        planner_inputs: Arc::new(manifest.planner_inputs),
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
        || catalog.models.iter().any(|model| match &model.target {
            ModelOfferingTarget::Package { package } => match &package.source {
                ModelPackageSource::HuggingFace { revision, .. } => {
                    revision.len() != 40
                        || !revision
                            .bytes()
                            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
                }
                ModelPackageSource::Local { .. } => true,
            },
            ModelOfferingTarget::SpeculativeDecodingPair { .. } => true,
        })
    {
        return Err(InventoryError::Integrity(
            "release catalog has missing or duplicate model identities".to_owned(),
        ));
    }
    Ok(())
}

fn validate_planner_inputs(
    catalog: &RecommendableModelCatalog,
    artifacts: &BTreeMap<ModelOfferingTargetId, ReleasePlannerInput>,
) -> Result<(), InventoryError> {
    let catalog_targets = catalog
        .models
        .iter()
        .map(|model| model.target_id.clone())
        .collect::<BTreeSet<_>>();
    let artifact_targets = artifacts.keys().cloned().collect::<BTreeSet<_>>();
    if artifact_targets != catalog_targets {
        return Err(InventoryError::Integrity(
            "release planner inputs do not exactly cover the catalog targets".to_owned(),
        ));
    }
    for (target_id, artifact) in artifacts {
        let package_files = catalog
            .models
            .iter()
            .find(|model| &model.target_id == target_id)
            .and_then(|model| match &model.target {
                ModelOfferingTarget::Package { package } => Some(
                    package
                        .files
                        .iter()
                        .map(|file| (file.path.clone(), file.size_bytes))
                        .collect::<BTreeSet<_>>(),
                ),
                ModelOfferingTarget::SpeculativeDecodingPair { .. } => None,
            });
        let planner_files = artifact
            .components
            .iter()
            .map(|component| {
                (
                    component.component.path.clone(),
                    component.component.size_bytes,
                )
            })
            .collect::<BTreeSet<_>>();
        if package_files.as_ref() != Some(&planner_files)
            || artifact.components.is_empty()
            || !artifact
                .components
                .iter()
                .any(|component| component.component.path == artifact.primary_gguf)
            || artifact.components.iter().any(|component| {
                component.source_header_digest.len() != 64
                    || component.source_header_size_bytes == 0
                    || component.planner_stub_digest.len() != 64
                    || component.planner_stub_size_bytes == 0
                    || !valid_hex_digest(&component.source_header_digest)
                    || !valid_hex_digest(&component.planner_stub_digest)
            })
        {
            return Err(InventoryError::Integrity(format!(
                "invalid release planner input {}",
                target_id.0
            )));
        }
    }
    Ok(())
}

fn valid_hex_digest(value: &str) -> bool {
    value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

impl GeneratedReleaseCatalog {
    pub fn encode_planner_bundle(
        &self,
        progress: impl FnMut(usize, usize),
    ) -> Result<Vec<u8>, InventoryError> {
        let source = catalog_source()?;
        validate_resolved_catalog(&self.catalog, &source)?;
        validate_planner_inputs(&self.catalog, &self.planner_inputs)?;
        let manifest = serde_json::to_vec(&ReleasePlannerManifest {
            planner_inputs: self.planner_inputs.clone(),
        })
        .map_err(|error| InventoryError::Internal(error.to_string()))?;
        planner_bundle::encode(&manifest, &self.planner_stubs, progress)
            .map_err(InventoryError::Integrity)
    }

    pub fn resolve_source_planner_target(
        &self,
        target_id: &ModelOfferingTargetId,
    ) -> Result<ResolvedModelTarget, InventoryError> {
        self.resolve_generated_planner_target(target_id, false)
    }

    pub fn resolve_compact_planner_target(
        &self,
        target_id: &ModelOfferingTargetId,
    ) -> Result<ResolvedModelTarget, InventoryError> {
        self.resolve_generated_planner_target(target_id, true)
    }

    fn resolve_generated_planner_target(
        &self,
        target_id: &ModelOfferingTargetId,
        compact: bool,
    ) -> Result<ResolvedModelTarget, InventoryError> {
        let (artifact, target) =
            planner_input_and_target(&self.catalog, &self.planner_inputs, target_id)?;
        materialize_planner_target(
            target_id,
            artifact,
            target,
            |component| {
                let (digest, expected_size) = if compact {
                    (
                        &component.planner_stub_digest,
                        component.planner_stub_size_bytes,
                    )
                } else {
                    (
                        &component.source_header_digest,
                        component.source_header_size_bytes,
                    )
                };
                let inputs = if compact {
                    &self.planner_stubs
                } else {
                    &self.source_headers
                };
                let input = inputs.get(digest).ok_or_else(|| {
                    InventoryError::Integrity(format!("missing planner input {digest}"))
                })?;
                Ok((Cow::Borrowed(input.as_slice()), expected_size))
            },
            if compact {
                "generated_compact_planner_stub"
            } else {
                "generated_source_planner_header"
            },
        )
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
        let Some(artifact) = self.planner_inputs.get(target_id) else {
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
        Ok(Some(materialize_planner_target(
            target_id,
            artifact,
            target,
            |component| {
                let stub = self
                    .planner_bundle
                    .input(&component.planner_stub_digest)
                    .map_err(InventoryError::Integrity)?;
                Ok((Cow::Owned(stub), component.planner_stub_size_bytes))
            },
            "release_catalog_planner_stub_digest",
        )?))
    }
}

fn planner_input_and_target<'a>(
    catalog: &'a RecommendableModelCatalog,
    artifacts: &'a BTreeMap<ModelOfferingTargetId, ReleasePlannerInput>,
    target_id: &ModelOfferingTargetId,
) -> Result<(&'a ReleasePlannerInput, ModelOfferingTarget), InventoryError> {
    let artifact = artifacts.get(target_id).ok_or_else(|| {
        InventoryError::Integrity(format!(
            "catalog target {} has no planner input",
            target_id.0
        ))
    })?;
    let target = catalog
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
    Ok((artifact, target))
}

fn materialize_planner_target<'a>(
    target_id: &ModelOfferingTargetId,
    artifact: &ReleasePlannerInput,
    target: ModelOfferingTarget,
    mut input_for: impl FnMut(&ReleasePlannerComponent) -> Result<(Cow<'a, [u8]>, u64), InventoryError>,
    integrity_method: &str,
) -> Result<ResolvedModelTarget, InventoryError> {
    let package = match &target {
        ModelOfferingTarget::Package { package } => package,
        ModelOfferingTarget::SpeculativeDecodingPair { .. } => {
            return Err(InventoryError::Integrity(
                "release planner input requires a package target".to_owned(),
            ));
        }
    };
    let source = match &package.source {
        ModelPackageSource::HuggingFace {
            repository,
            revision,
        } => ModelSource::HuggingFace {
            repository: repository.clone(),
            requested_revision: revision.clone(),
            commit: revision.clone(),
            metadata: None,
        },
        ModelPackageSource::Local { .. } => {
            return Err(InventoryError::Integrity(
                "release catalog package must have an immutable Hugging Face source".to_owned(),
            ));
        }
    };
    let package_identity = package.id.0.clone();
    let workspace = tempfile::tempdir().map_err(|error| InventoryError::Io(error.to_string()))?;
    for component in &artifact.components {
        let (input, expected_size) = input_for(component)?;
        if input.len()
            != usize::try_from(expected_size)
                .map_err(|_| InventoryError::Integrity("planner input is too large".to_owned()))?
        {
            return Err(InventoryError::Integrity(format!(
                "planner input for {} has the wrong length",
                component.component.path.display()
            )));
        }
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
        file.write_all(&input)
            .and_then(|()| file.set_len(component.component.size_bytes))
            .map_err(|error| InventoryError::Io(error.to_string()))?;
    }
    let components = artifact
        .components
        .iter()
        .map(|component| component.component.clone())
        .collect::<Vec<_>>();
    let model = InventoryModel {
        id: ModelId(package_identity.clone()),
        content_id: ContentId(package_identity.clone()),
        created: 0,
        name: package_identity,
        supported_parameters: Vec::new(),
        serving_configuration: None,
        availability: ModelAvailability::Available { ready_at: 0 },
        source,
        location: ModelLocation::Directory {
            source_id: "release_catalog".to_owned(),
            root: workspace.path().to_path_buf(),
            components: components.clone(),
            total_bytes: components
                .iter()
                .map(|component| component.size_bytes)
                .sum(),
            integrity: Integrity::Verified {
                method: integrity_method.to_owned(),
            },
        },
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
    Ok(
        ResolvedModelTarget::new(target_id.clone(), target, resolved, None)
            .retain_resolution_guard(workspace),
    )
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
    let lock = model_catalog_lock()?;
    for model in &catalog.models {
        let declaration = source
            .models
            .iter()
            .find(|declaration| declaration.id == model.checkpoint_id)
            .ok_or_else(|| {
                InventoryError::Integrity(format!(
                    "release catalog target {} has no source declaration",
                    model.id.0
                ))
            })?;
        let expected_commit = lock.get(&model.checkpoint_id).ok_or_else(|| {
            InventoryError::Integrity(format!(
                "model catalog lock is missing {}",
                model.checkpoint_id
            ))
        })?;
        let package_source = match &model.target {
            ModelOfferingTarget::Package { package } => &package.source,
            ModelOfferingTarget::SpeculativeDecodingPair { .. } => {
                return Err(InventoryError::Integrity(
                    "release catalog contains an unsupported speculative target".to_owned(),
                ));
            }
        };
        if !package_source_matches(package_source, declaration, expected_commit) {
            return Err(InventoryError::Integrity(format!(
                "release catalog target {} does not match models.json and models.lock.json",
                model.id.0
            )));
        }
    }
    Ok(())
}

fn package_source_matches(
    source: &ModelPackageSource,
    declaration: &CatalogModel,
    expected_commit: &str,
) -> bool {
    matches!(
        source,
        ModelPackageSource::HuggingFace {
            repository,
            revision,
        } if repository == &declaration.repository && revision == expected_commit
    )
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

fn recommendable_model(
    declaration: &CatalogModel,
    format: &str,
    package: ModelPackage,
    properties: &InventoryProperties,
) -> RecommendableModel {
    let (fidelity_rank, quantization_aware) = fidelity(&declaration.id, format);
    RecommendableModel {
        id: RecommendableModelId(format!("{}:{format}", declaration.id)),
        checkpoint_id: declaration.id.clone(),
        target_id: offering_target_id(&[&package.id]),
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
        license: declaration.license.clone(),
        capabilities: model_capabilities(properties),
        quality_score: declaration.quality_score,
        quality_score_provenance: declaration.quality_score_provenance.clone(),
        fidelity_rank,
        quantization_aware,
        quality_evidence: declaration.quality_evidence.clone(),
    }
}

fn catalog_from_planner_inputs(
    source: &CatalogSource,
    inputs: &BTreeMap<ModelOfferingTargetId, ReleasePlannerInput>,
) -> Result<RecommendableModelCatalog, InventoryError> {
    let by_model_id = inputs
        .values()
        .map(|input| (input.model_id.clone(), input))
        .collect::<BTreeMap<_, _>>();
    if by_model_id.len() != inputs.len() {
        return Err(InventoryError::Integrity(
            "planner bundle contains duplicate catalog model identities".to_owned(),
        ));
    }
    let mut models = Vec::new();
    for declaration in &source.models {
        for format in &declaration.formats {
            let model_id = RecommendableModelId(format!("{}:{format}", declaration.id));
            let input = by_model_id.get(&model_id).ok_or_else(|| {
                InventoryError::Integrity(format!(
                    "planner bundle is missing catalog model {}",
                    model_id.0
                ))
            })?;
            let model = recommendable_model(
                declaration,
                format,
                input.package.clone(),
                &input.properties,
            );
            if !inputs.contains_key(&model.target_id) {
                return Err(InventoryError::Integrity(format!(
                    "planner bundle target does not match catalog model {}",
                    model_id.0
                )));
            }
            models.push(model);
        }
    }
    if models.len() != inputs.len() {
        return Err(InventoryError::Integrity(
            "planner bundle contains models absent from the source catalog".to_owned(),
        ));
    }
    Ok(RecommendableModelCatalog {
        models,
        diagnostics: Vec::new(),
    })
}

pub struct ResolvingRecommendableCatalog {
    models: Arc<ModelManager>,
}

impl ResolvingRecommendableCatalog {
    #[must_use]
    pub fn new(models: Arc<ModelManager>) -> Self {
        Self { models }
    }

    async fn resolve_model(
        &self,
        declaration: &CatalogModel,
        format: &str,
        snapshot: &HuggingFaceRepositorySnapshot,
    ) -> Result<
        (
            RecommendableModel,
            ReleasePlannerInput,
            BTreeMap<String, Vec<u8>>,
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
        let model = recommendable_model(
            declaration,
            format,
            package.clone(),
            &prepared.model.model.properties,
        );
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
                            "bundle construction lost source header {}",
                            header.digest
                        ))
                    })
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        let primary_header = prepared
            .headers
            .iter()
            .find(|header| header.path == primary.path)
            .ok_or_else(|| {
                InventoryError::Integrity("catalog primary has no planner header".to_owned())
            })?;
        let context = planner_stub_context(
            headers
                .get(&primary_header.digest)
                .ok_or_else(|| {
                    InventoryError::Integrity(format!(
                        "bundle construction lost primary header {}",
                        primary_header.digest
                    ))
                })?
                .as_slice(),
        )
        .map_err(|error| InventoryError::Integrity(error.to_string()))?;
        let mut planner_stubs = BTreeMap::new();
        let components = prepared
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
                let source = headers.get(&header.digest).ok_or_else(|| {
                    InventoryError::Integrity(format!(
                        "bundle construction lost source header {}",
                        header.digest
                    ))
                })?;
                if planner_bundle::sha256(source) != header.digest {
                    return Err(InventoryError::Integrity(format!(
                        "catalog source header {} failed integrity validation",
                        header.digest
                    )));
                }
                let kind = if component.path == primary.path {
                    PlannerStubComponent::Primary
                } else {
                    PlannerStubComponent::Auxiliary
                };
                let stub = compact_planner_stub(source, &context, kind)
                    .map_err(|error| InventoryError::Integrity(error.to_string()))?;
                let planner_stub_digest = planner_bundle::sha256(&stub);
                let planner_stub_size_bytes = u64::try_from(stub.len()).map_err(|_| {
                    InventoryError::Integrity("planner stub is too large".to_owned())
                })?;
                if let Some(previous) =
                    planner_stubs.insert(planner_stub_digest.clone(), stub.clone())
                    && previous != stub
                {
                    return Err(InventoryError::Integrity(format!(
                        "planner stub digest collision {planner_stub_digest}"
                    )));
                }
                Ok(ReleasePlannerComponent {
                    component: component.clone(),
                    source_header_digest: header.digest.clone(),
                    source_header_size_bytes: u64::try_from(source.len()).map_err(|_| {
                        InventoryError::Integrity("planner source header is too large".to_owned())
                    })?,
                    planner_stub_digest,
                    planner_stub_size_bytes,
                })
            })
            .collect::<Result<Vec<_>, InventoryError>>()?;
        let planner = ReleasePlannerInput {
            model_id: model.id.clone(),
            package,
            properties: prepared.model.model.properties.clone(),
            primary_gguf: primary.path.clone(),
            components,
        };
        Ok((model, planner, headers, planner_stubs))
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
    pub fn resolve_release_catalog<F>(
        &self,
        progress: F,
    ) -> BoxFuture<'_, Result<GeneratedReleaseCatalog, InventoryError>>
    where
        F: Fn(&str, usize, usize) + Send + Sync + 'static,
    {
        Box::pin(async move {
            let source = catalog_source()?;
            let lock = model_catalog_lock()?;
            let repositories = source
                .models
                .iter()
                .map(|declaration| {
                    Ok::<_, InventoryError>((
                        declaration.repository.clone(),
                        lock.get(&declaration.id).cloned().ok_or_else(|| {
                            InventoryError::Integrity(format!(
                                "model catalog lock is missing {}",
                                declaration.id
                            ))
                        })?,
                    ))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let repository_total = repositories.len();
            let mut repository_completed = 0;
            let snapshots = stream::iter(repositories)
                .map(|(repository, revision)| async move {
                    let result = refresh_hugging_face_repository(
                        &self.models,
                        HuggingFaceRepositoryRequest {
                            repository: repository.clone(),
                            revision,
                        },
                    )
                    .await;
                    (repository, result)
                })
                .buffer_unordered(12)
                .inspect(|_| {
                    repository_completed += 1;
                    progress(
                        "Resolved catalog repositories",
                        repository_completed,
                        repository_total,
                    );
                })
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
            let model_total = source.models.len();
            let mut model_completed = 0;
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
                .inspect(|_| {
                    model_completed += 1;
                    progress("Prepared catalog models", model_completed, model_total);
                })
                .flat_map(stream::iter)
                .collect::<Vec<_>>()
                .await;
            let mut resolved = resolved;
            resolved.sort_by_key(|(declaration_index, format_index, ..)| {
                (*declaration_index, *format_index)
            });
            let mut models = Vec::new();
            let mut planner_inputs = BTreeMap::new();
            let mut source_headers = BTreeMap::new();
            let mut planner_stubs = BTreeMap::new();
            let mut diagnostics = Vec::new();
            for (_, _, declaration, format, result) in resolved {
                match result {
                    Ok((model, planner, model_headers, model_stubs)) => {
                        planner_inputs.insert(model.target_id.clone(), planner);
                        models.push(model);
                        for (digest, header) in model_headers {
                            if let Some(previous) =
                                source_headers.insert(digest.clone(), header.clone())
                                && previous != header
                            {
                                return Err(InventoryError::Integrity(format!(
                                    "planner source header digest collision {digest}"
                                )));
                            }
                        }
                        for (digest, stub) in model_stubs {
                            if let Some(previous) =
                                planner_stubs.insert(digest.clone(), stub.clone())
                                && previous != stub
                            {
                                return Err(InventoryError::Integrity(format!(
                                    "planner stub digest collision {digest}"
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
                planner_inputs,
                source_headers,
                planner_stubs,
            })
        })
    }
}

impl RecommendableModelCatalogProvider for ResolvingRecommendableCatalog {
    fn catalog(&self) -> BoxFuture<'_, Result<RecommendableModelCatalog, InventoryError>> {
        Box::pin(async move { Ok(self.resolve_release_catalog(|_, _, _| {}).await?.catalog) })
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
        assert_eq!(
            formats("qwen3.5-122b-a10b"),
            ["UD-Q4_K_XL", "UD-Q5_K_XL", "UD-Q6_K_XL", "UD-Q8_K_XL"]
        );
        assert_eq!(
            formats("nemotron-3-super-120b-a12b"),
            ["UD-Q4_K_XL", "MXFP4_MOE"]
        );
        assert_eq!(formats("deepseek-v4-flash"), ["UD-Q4_K_XL", "UD-Q8_K_XL"]);
        assert_eq!(formats("glm-5.2"), ["BF16"]);
    }

    #[test]
    fn model_lock_exactly_covers_the_authored_catalog() {
        let source = catalog_source().expect("catalog source");
        let lock = model_catalog_lock().expect("model catalog lock");
        assert_eq!(lock.len(), source.models.len());
        assert!(
            source
                .models
                .iter()
                .all(|model| lock.contains_key(&model.id))
        );
    }

    #[test]
    fn package_source_must_match_the_authored_repository_and_locked_commit() {
        let source = catalog_source().expect("catalog source");
        let declaration = &source.models[0];
        let commit = model_catalog_lock().expect("model catalog lock")[&declaration.id].clone();
        let package_source =
            |repository: String, revision: String| ModelPackageSource::HuggingFace {
                repository,
                revision,
            };
        assert!(package_source_matches(
            &package_source(declaration.repository.clone(), commit.clone()),
            declaration,
            &commit,
        ));
        assert!(!package_source_matches(
            &package_source("other/repository".to_owned(), commit.clone()),
            declaration,
            &commit,
        ));
        assert!(!package_source_matches(
            &package_source(declaration.repository.clone(), "0".repeat(40)),
            declaration,
            &commit,
        ));
    }
}
