use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use futures_util::future::BoxFuture;
use futures_util::{StreamExt, stream};
use icn_contracts::models::{
    CatalogDiagnostic, ModelFailure, ModelPackage, ModelPackageSource, ModelServingConfiguration,
    RecommendableModel, RecommendableModelCatalog, RecommendableModelCatalogProvider,
    RecommendableModelId, ResolvedServableModelBundle, ServableModelBundle, ServableModelBundleKey,
    ServingProfile, SpeculativeDraftSource, SpeculativeMethod,
};
use icn_contracts::{
    ComponentRole, ContentId, HuggingFaceRepositoryRequest, HuggingFaceRepositorySnapshot,
    Integrity, InventoryError, InventoryModel, InventoryProperties, ModelAvailability,
    ModelComponent, ModelId, ModelLocation, ModelPreviewSource, ModelSource, ResolvedComponent,
    ResolvedModel,
};
use serde::{Deserialize, Serialize};

use crate::cache::ModelBlobKind;
use crate::capabilities::model_capabilities;
use crate::inventory::ModelManager;
use crate::package_service::{
    package_from_resolved, servable_model_bundle_key_for_bundle, serving_configuration_id,
    serving_configuration_identity_is_valid,
};
use crate::planner_stub::{PlannerStubComponent, compact_planner_stub, planner_stub_context};
use crate::preview::PreparedPreview;
use crate::refresh_hugging_face_repository;

#[path = "../../../catalog/planner_bundle.rs"]
mod planner_bundle;
use planner_bundle::PlannerBundle;

const CATALOG_SOURCE: &str = include_str!("../../../catalog/models.json");
const CATALOG_LOCK: &str = include_str!("../../../catalog/models.lock.json");
const MIN_CATALOG_CONTEXT_LENGTH: u32 = 4_096;
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
    variants: Vec<CatalogVariant>,
    context_length: u32,
    #[serde(default)]
    speculative_decoding: Option<CatalogSpeculativeDecoding>,
    license: String,
    quality_score: f64,
    quality_score_provenance: String,
    quality_evidence: Vec<String>,
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CatalogVariant {
    format: String,
    variant_label: String,
    fidelity_rank: u32,
    quantization_aware: bool,
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CatalogSpeculativeDecoding {
    method: CatalogSpeculativeMethod,
    draft: CatalogSpeculativeDraftSource,
}

#[derive(Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum CatalogSpeculativeDraftSource {
    Embedded,
    File {
        #[serde(default)]
        repository: Option<String>,
        path: PathBuf,
    },
}

#[derive(Clone, Copy, Deserialize, PartialEq, Eq)]
enum CatalogSpeculativeMethod {
    #[serde(rename = "mtp")]
    Mtp,
    #[serde(rename = "dflash")]
    DFlash,
    #[serde(rename = "dspark")]
    DSpark,
}

impl From<CatalogSpeculativeMethod> for SpeculativeMethod {
    fn from(method: CatalogSpeculativeMethod) -> Self {
        match method {
            CatalogSpeculativeMethod::Mtp => Self::Mtp,
            CatalogSpeculativeMethod::DFlash => Self::DFlash,
            CatalogSpeculativeMethod::DSpark => Self::DSpark,
        }
    }
}

impl CatalogSpeculativeDraftSource {
    fn file<'a>(&'a self, target_repository: &'a str) -> Option<(&'a str, &'a Path)> {
        match self {
            Self::Embedded => None,
            Self::File { repository, path } => Some((
                repository.as_deref().unwrap_or(target_repository),
                path.as_path(),
            )),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ReleasePlannerManifest {
    planner_inputs: BTreeMap<ServableModelBundleKey, ReleasePlannerInput>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ReleasePlannerInput {
    model_id: RecommendableModelId,
    target: ReleasePlannerPackage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    draft: Option<ReleasePlannerPackage>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ReleasePlannerPackage {
    package: ModelPackage,
    properties: InventoryProperties,
    primary_gguf: PathBuf,
    components: Vec<ReleasePlannerComponent>,
}

impl ReleasePlannerInput {
    fn packages(&self) -> impl Iterator<Item = &ReleasePlannerPackage> {
        std::iter::once(&self.target).chain(self.draft.iter())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelCatalogLockEntry {
    pub target: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speculative_draft: Option<String>,
}

pub type ModelCatalogLock = BTreeMap<String, ModelCatalogLockEntry>;

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
    planner_inputs: BTreeMap<ServableModelBundleKey, ReleasePlannerInput>,
    source_headers: BTreeMap<String, Vec<u8>>,
    planner_stubs: BTreeMap<String, Vec<u8>>,
}

#[derive(Clone)]
pub struct ReleaseCatalog {
    catalog: RecommendableModelCatalog,
    planner_inputs: Arc<BTreeMap<ServableModelBundleKey, ReleasePlannerInput>>,
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
    let mut presentations = BTreeSet::new();
    for model in &source.models {
        let formats = model
            .variants
            .iter()
            .map(|variant| variant.format.as_str())
            .collect::<BTreeSet<_>>();
        let variant_labels = model
            .variants
            .iter()
            .map(|variant| variant.variant_label.as_str())
            .collect::<BTreeSet<_>>();
        if model.id.is_empty()
            || model.display_name.is_empty()
            || model.description.is_empty()
            || model.repository.is_empty()
            || model.variants.is_empty()
            || formats.len() != model.variants.len()
            || variant_labels.len() != model.variants.len()
            || model.variants.iter().any(|variant| {
                variant.format.is_empty()
                    || variant.format.trim() != variant.format.as_str()
                    || variant.variant_label.is_empty()
                    || variant.variant_label.trim() != variant.variant_label.as_str()
                    || variant.variant_label.contains('(')
                    || variant.variant_label.contains(')')
                    || variant.fidelity_rank == 0
            })
            || model.context_length < MIN_CATALOG_CONTEXT_LENGTH
            || model
                .speculative_decoding
                .as_ref()
                .is_some_and(|speculative| {
                    matches!(
                        (&speculative.method, &speculative.draft),
                        (
                            CatalogSpeculativeMethod::DFlash | CatalogSpeculativeMethod::DSpark,
                            CatalogSpeculativeDraftSource::Embedded
                        )
                    ) || match &speculative.draft {
                        CatalogSpeculativeDraftSource::Embedded => false,
                        CatalogSpeculativeDraftSource::File { repository, path } => {
                            repository.as_ref().is_some_and(String::is_empty)
                                || path.as_os_str().is_empty()
                        }
                    }
                })
            || model.license.is_empty()
            || !model.quality_score.is_finite()
            || model.quality_score < 0.0
            || model.quality_score_provenance.is_empty()
            || model.quality_evidence.is_empty()
            || !ids.insert(model.id.as_str())
            || model.variants.iter().any(|variant| {
                !presentations.insert((model.display_name.as_str(), variant.variant_label.as_str()))
            })
        {
            return Err(InventoryError::Integrity(format!(
                "invalid or duplicate catalog declaration {}",
                model.id
            )));
        }
    }
    Ok(source)
}

pub fn model_catalog_lock() -> Result<ModelCatalogLock, InventoryError> {
    model_catalog_lock_from(CATALOG_LOCK.as_bytes(), &catalog_source()?)
}

pub async fn advance_model_catalog_lock(
    models: Arc<ModelManager>,
) -> Result<ModelCatalogLock, InventoryError> {
    let source = catalog_source()?;
    let resolved = stream::iter(source.models)
        .map(|declaration| {
            let models = Arc::clone(&models);
            async move {
                let entry_id = declaration.id;
                let target_repository = declaration.repository;
                let target = refresh_hugging_face_repository(
                    &models,
                    HuggingFaceRepositoryRequest {
                        repository: target_repository.clone(),
                        revision: "main".to_owned(),
                    },
                )
                .await
                .map_err(|error| {
                    InventoryError::Upstream(format!(
                        "failed to resolve catalog entry {entry_id}: {error}"
                    ))
                })?;
                let speculative_draft = match declaration
                    .speculative_decoding
                    .and_then(|speculative| match speculative.draft {
                        CatalogSpeculativeDraftSource::Embedded => None,
                        CatalogSpeculativeDraftSource::File { repository, .. } => {
                            Some(repository.unwrap_or_else(|| target_repository.clone()))
                        }
                    }) {
                    Some(repository) if repository == target_repository => {
                        Some(target.commit.clone())
                    }
                    Some(repository) => Some(
                        refresh_hugging_face_repository(
                            &models,
                            HuggingFaceRepositoryRequest {
                                repository,
                                revision: "main".to_owned(),
                            },
                        )
                        .await
                        .map_err(|error| {
                            InventoryError::Upstream(format!(
                                "failed to resolve catalog entry {entry_id} speculative draft: {error}"
                            ))
                        })?
                        .commit,
                    ),
                    None => None,
                };
                Ok::<_, InventoryError>((
                    entry_id,
                    ModelCatalogLockEntry {
                        target: target.commit,
                        speculative_draft,
                    },
                ))
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
) -> Result<ModelCatalogLock, InventoryError> {
    let lock: ModelCatalogLock = serde_json::from_slice(bytes).map_err(|error| {
        InventoryError::Integrity(format!("invalid model catalog lock: {error}"))
    })?;
    validate_model_catalog_lock(&lock, source)?;
    Ok(lock)
}

fn validate_model_catalog_lock(
    lock: &ModelCatalogLock,
    source: &CatalogSource,
) -> Result<(), InventoryError> {
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
    if source.models.iter().any(|model| {
        let Some(entry) = lock.get(&model.id) else {
            return true;
        };
        !valid_commit(&entry.target)
            || entry.speculative_draft.is_some()
                != model
                    .speculative_decoding
                    .as_ref()
                    .is_some_and(|speculative| {
                        matches!(
                            speculative.draft,
                            CatalogSpeculativeDraftSource::File { .. }
                        )
                    })
            || entry
                .speculative_draft
                .as_ref()
                .is_some_and(|commit| !valid_commit(commit))
    }) {
        return Err(InventoryError::Integrity(
            "model catalog lock contains invalid or mismatched package revisions".to_owned(),
        ));
    }
    Ok(())
}

fn valid_commit(commit: &str) -> bool {
    commit.len() == 40
        && commit
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
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
        for component in input
            .packages()
            .flat_map(|package| package.components.iter())
        {
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
    let configuration_ids = catalog
        .models
        .iter()
        .map(|model| model.configuration.id.clone())
        .collect::<BTreeSet<_>>();
    if catalog.models.is_empty()
        || model_ids.len() != catalog.models.len()
        || configuration_ids.len() != catalog.models.len()
        || catalog.models.iter().any(|model| {
            !serving_configuration_identity_is_valid(&model.configuration)
                || model.configuration.profile.context_length < MIN_CATALOG_CONTEXT_LENGTH
                || servable_bundle_packages(&model.configuration.bundle).any(|package| {
                    model.configuration.profile.context_length
                        > package.properties.maximum_context_length
                        || !matches!(
                            &package.source,
                            ModelPackageSource::HuggingFace { revision, .. }
                                if valid_commit(revision)
                        )
                })
        })
    {
        return Err(InventoryError::Integrity(
            "release catalog has missing or duplicate model identities".to_owned(),
        ));
    }
    Ok(())
}

fn servable_bundle_packages(bundle: &ServableModelBundle) -> impl Iterator<Item = &ModelPackage> {
    let target = match bundle {
        ServableModelBundle::Standalone { package } => package,
        ServableModelBundle::SpeculativeDecoding { target, .. } => target,
    };
    let draft = match bundle {
        ServableModelBundle::SpeculativeDecoding {
            draft_source: SpeculativeDraftSource::Separate { draft },
            ..
        } => Some(draft),
        _ => None,
    };
    std::iter::once(target).chain(draft)
}

fn recommendable_model_bundle_key(model: &RecommendableModel) -> ServableModelBundleKey {
    servable_model_bundle_key_for_bundle(&model.configuration.bundle)
}

fn validate_planner_inputs(
    catalog: &RecommendableModelCatalog,
    artifacts: &BTreeMap<ServableModelBundleKey, ReleasePlannerInput>,
) -> Result<(), InventoryError> {
    let catalog_bundles = catalog
        .models
        .iter()
        .map(recommendable_model_bundle_key)
        .collect::<BTreeSet<_>>();
    let artifact_bundles = artifacts.keys().cloned().collect::<BTreeSet<_>>();
    if artifact_bundles != catalog_bundles {
        return Err(InventoryError::Integrity(
            "release planner inputs do not exactly cover the catalog bundles".to_owned(),
        ));
    }
    for (bundle_key, artifact) in artifacts {
        let bundle = catalog
            .models
            .iter()
            .find(|model| recommendable_model_bundle_key(model) == *bundle_key)
            .map(|model| &model.configuration.bundle)
            .ok_or_else(|| {
                InventoryError::Integrity(format!(
                    "release planner input {} has no catalog bundle",
                    bundle_key.0
                ))
            })?;
        let (target, draft) = match bundle {
            ServableModelBundle::Standalone { package } => (package, None),
            ServableModelBundle::SpeculativeDecoding {
                target,
                draft_source,
                ..
            } => (
                target,
                match draft_source {
                    SpeculativeDraftSource::Embedded => None,
                    SpeculativeDraftSource::Separate { draft } => Some(draft),
                },
            ),
        };
        if !valid_release_planner_package(target, &artifact.target)
            || draft
                .zip(artifact.draft.as_ref())
                .is_some_and(|(package, planner)| !valid_release_planner_package(package, planner))
            || draft.is_some() != artifact.draft.is_some()
        {
            return Err(InventoryError::Integrity(format!(
                "invalid release planner input {}",
                bundle_key.0
            )));
        }
    }
    Ok(())
}

fn valid_release_planner_package(package: &ModelPackage, planner: &ReleasePlannerPackage) -> bool {
    let package_files = package
        .files
        .iter()
        .map(|file| (file.path.clone(), file.size_bytes))
        .collect::<BTreeSet<_>>();
    let planner_files = planner
        .components
        .iter()
        .map(|component| {
            (
                component.component.path.clone(),
                component.component.size_bytes,
            )
        })
        .collect::<BTreeSet<_>>();
    package == &planner.package
        && package_files == planner_files
        && !planner.components.is_empty()
        && planner
            .components
            .iter()
            .any(|component| component.component.path == planner.primary_gguf)
        && planner.components.iter().all(|component| {
            component.source_header_digest.len() == 64
                && component.source_header_size_bytes > 0
                && component.planner_stub_digest.len() == 64
                && component.planner_stub_size_bytes > 0
                && valid_hex_digest(&component.source_header_digest)
                && valid_hex_digest(&component.planner_stub_digest)
        })
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

    pub fn resolve_source_planner_bundle(
        &self,
        bundle_key: &ServableModelBundleKey,
    ) -> Result<ResolvedServableModelBundle, InventoryError> {
        self.resolve_generated_planner_bundle(bundle_key, false)
    }

    pub fn resolve_compact_planner_bundle(
        &self,
        bundle_key: &ServableModelBundleKey,
    ) -> Result<ResolvedServableModelBundle, InventoryError> {
        self.resolve_generated_planner_bundle(bundle_key, true)
    }

    fn resolve_generated_planner_bundle(
        &self,
        bundle_key: &ServableModelBundleKey,
        compact: bool,
    ) -> Result<ResolvedServableModelBundle, InventoryError> {
        let (artifact, bundle) =
            planner_input_and_bundle(&self.catalog, &self.planner_inputs, bundle_key)?;
        materialize_planner_bundle(
            bundle_key,
            artifact,
            bundle,
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

    pub fn resolve_bundle(
        &self,
        bundle_key: &ServableModelBundleKey,
    ) -> Result<Option<ResolvedServableModelBundle>, InventoryError> {
        let Some(artifact) = self.planner_inputs.get(bundle_key) else {
            return Ok(None);
        };
        let bundle = self
            .catalog
            .models
            .iter()
            .find(|model| recommendable_model_bundle_key(model) == *bundle_key)
            .map(|model| model.configuration.bundle.clone())
            .ok_or_else(|| {
                InventoryError::Integrity(format!(
                    "catalog bundle {} has no model declaration",
                    bundle_key.0
                ))
            })?;
        Ok(Some(materialize_planner_bundle(
            bundle_key,
            artifact,
            bundle,
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

fn planner_input_and_bundle<'a>(
    catalog: &'a RecommendableModelCatalog,
    artifacts: &'a BTreeMap<ServableModelBundleKey, ReleasePlannerInput>,
    bundle_key: &ServableModelBundleKey,
) -> Result<(&'a ReleasePlannerInput, ServableModelBundle), InventoryError> {
    let artifact = artifacts.get(bundle_key).ok_or_else(|| {
        InventoryError::Integrity(format!(
            "catalog bundle {} has no planner input",
            bundle_key.0
        ))
    })?;
    let bundle = catalog
        .models
        .iter()
        .find(|model| recommendable_model_bundle_key(model) == *bundle_key)
        .map(|model| model.configuration.bundle.clone())
        .ok_or_else(|| {
            InventoryError::Integrity(format!(
                "catalog bundle {} has no model declaration",
                bundle_key.0
            ))
        })?;
    Ok((artifact, bundle))
}

fn materialize_planner_bundle<'a>(
    bundle_key: &ServableModelBundleKey,
    artifact: &ReleasePlannerInput,
    bundle: ServableModelBundle,
    mut input_for: impl FnMut(&ReleasePlannerComponent) -> Result<(Cow<'a, [u8]>, u64), InventoryError>,
    integrity_method: &str,
) -> Result<ResolvedServableModelBundle, InventoryError> {
    let (target, draft) = match &bundle {
        ServableModelBundle::Standalone { package } => (package, None),
        ServableModelBundle::SpeculativeDecoding {
            target,
            draft_source,
            ..
        } => (
            target,
            match draft_source {
                SpeculativeDraftSource::Embedded => None,
                SpeculativeDraftSource::Separate { draft } => Some(draft),
            },
        ),
    };
    if draft.is_some() != artifact.draft.is_some() {
        return Err(InventoryError::Integrity(
            "release planner input draft does not match its bundle".to_owned(),
        ));
    }
    let (target_model, target_workspace) =
        materialize_planner_package(&artifact.target, target, &mut input_for, integrity_method)?;
    let (draft_model, draft_workspace) = match (draft, artifact.draft.as_ref()) {
        (Some(package), Some(planner)) => {
            let (model, workspace) =
                materialize_planner_package(planner, package, &mut input_for, integrity_method)?;
            (Some(model), Some(workspace))
        }
        (None, None) => (None, None),
        _ => unreachable!("draft presence was validated"),
    };
    let mut resolved =
        ResolvedServableModelBundle::new(bundle_key.clone(), bundle, target_model, draft_model)
            .retain_resolution_guard(target_workspace);
    if let Some(workspace) = draft_workspace {
        resolved = resolved.retain_resolution_guard(workspace);
    }
    Ok(resolved)
}

fn materialize_planner_package<'a>(
    artifact: &ReleasePlannerPackage,
    package: &ModelPackage,
    input_for: &mut impl FnMut(&ReleasePlannerComponent) -> Result<(Cow<'a, [u8]>, u64), InventoryError>,
    integrity_method: &str,
) -> Result<(ResolvedModel, tempfile::TempDir), InventoryError> {
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
    Ok((resolved, workspace))
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
                .variants
                .iter()
                .map(|variant| format!("{}:{}", model.id, variant.format))
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
                    "release catalog bundle {} has no source declaration",
                    model.id.0
                ))
            })?;
        let expected_lock = lock.get(&model.checkpoint_id).ok_or_else(|| {
            InventoryError::Integrity(format!(
                "model catalog lock is missing {}",
                model.checkpoint_id
            ))
        })?;
        let (target, draft_source, method) = match &model.configuration.bundle {
            ServableModelBundle::Standalone { package } => (package, None, None),
            ServableModelBundle::SpeculativeDecoding {
                target,
                draft_source,
                method,
            } => (target, Some(draft_source), Some(method)),
        };
        let speculative_matches = match (&declaration.speculative_decoding, draft_source, method) {
            (None, None, None) => true,
            (
                Some(CatalogSpeculativeDecoding {
                    method: expected_method,
                    draft: CatalogSpeculativeDraftSource::Embedded,
                }),
                Some(SpeculativeDraftSource::Embedded),
                Some(method),
            ) => expected_lock.speculative_draft.is_none() && method == &(*expected_method).into(),
            (
                Some(CatalogSpeculativeDecoding {
                    method: expected_method,
                    draft: CatalogSpeculativeDraftSource::File { .. },
                }),
                Some(SpeculativeDraftSource::Separate { draft }),
                Some(method),
            ) => {
                let Some(expected_draft_commit) = expected_lock.speculative_draft.as_deref() else {
                    return Err(InventoryError::Integrity(format!(
                        "model catalog lock is missing {} speculative draft",
                        model.checkpoint_id
                    )));
                };
                let Some((expected_repository, expected_path)) = declaration
                    .speculative_decoding
                    .as_ref()
                    .and_then(|speculative| speculative.draft.file(&declaration.repository))
                else {
                    return Err(InventoryError::Integrity(
                        "file draft declaration lost its source".to_owned(),
                    ));
                };
                method == &(*expected_method).into()
                    && package_source_matches(
                        &draft.source,
                        expected_repository,
                        expected_draft_commit,
                    )
                    && draft.files.iter().any(|file| file.path == expected_path)
            }
            _ => false,
        };
        if model.configuration.profile.context_length != declaration.context_length
            || !package_source_matches(
                &target.source,
                &declaration.repository,
                &expected_lock.target,
            )
            || !speculative_matches
        {
            return Err(InventoryError::Integrity(format!(
                "release catalog bundle {} does not match models.json and models.lock.json",
                model.id.0
            )));
        }
    }
    Ok(())
}

fn package_source_matches(
    source: &ModelPackageSource,
    expected_repository: &str,
    expected_commit: &str,
) -> bool {
    matches!(
        source,
        ModelPackageSource::HuggingFace {
            repository,
            revision,
        } if repository == expected_repository && revision == expected_commit
    )
}

fn recommendable_model(
    declaration: &CatalogModel,
    variant: &CatalogVariant,
    target: ModelPackage,
    draft: Option<ModelPackage>,
    properties: &InventoryProperties,
) -> Result<RecommendableModel, InventoryError> {
    let has_draft = draft.is_some();
    let bundle = match &declaration.speculative_decoding {
        None => ServableModelBundle::Standalone { package: target },
        Some(speculative) => {
            let draft_source = match speculative.draft {
                CatalogSpeculativeDraftSource::Embedded => {
                    let InventoryProperties::Inspected {
                        nextn_predict_layers,
                        ..
                    } = properties
                    else {
                        return Err(InventoryError::Integrity(format!(
                            "{} cannot verify its embedded speculative draft",
                            declaration.id
                        )));
                    };
                    if nextn_predict_layers.unwrap_or(0) == 0 {
                        return Err(InventoryError::Integrity(format!(
                            "{} declares an embedded speculative draft but its target GGUF has no NextN layers",
                            declaration.id
                        )));
                    }
                    SpeculativeDraftSource::Embedded
                }
                CatalogSpeculativeDraftSource::File { .. } => SpeculativeDraftSource::Separate {
                    draft: draft.ok_or_else(|| {
                        InventoryError::Integrity(format!(
                            "{} has no resolved speculative draft",
                            declaration.id
                        ))
                    })?,
                },
            };
            ServableModelBundle::SpeculativeDecoding {
                target,
                draft_source,
                method: speculative.method.into(),
            }
        }
    };
    if declaration.speculative_decoding.is_none() && has_draft {
        return Err(InventoryError::Integrity(format!(
            "{} resolved an undeclared speculative draft",
            declaration.id
        )));
    }
    let bundle_key = servable_model_bundle_key_for_bundle(&bundle);
    let profile = ServingProfile {
        context_length: declaration.context_length,
    };
    Ok(RecommendableModel {
        id: RecommendableModelId(format!("{}:{}", declaration.id, variant.format)),
        checkpoint_id: declaration.id.clone(),
        configuration: ModelServingConfiguration {
            id: serving_configuration_id(&bundle_key, &profile),
            bundle,
            profile,
        },
        display_name: declaration.display_name.clone(),
        variant_label: variant.variant_label.clone(),
        description: declaration.description.clone(),
        license: declaration.license.clone(),
        capabilities: model_capabilities(properties),
        quality_score: declaration.quality_score,
        quality_score_provenance: declaration.quality_score_provenance.clone(),
        fidelity_rank: variant.fidelity_rank,
        quantization_aware: variant.quantization_aware,
        quality_evidence: declaration.quality_evidence.clone(),
    })
}

fn catalog_from_planner_inputs(
    source: &CatalogSource,
    inputs: &BTreeMap<ServableModelBundleKey, ReleasePlannerInput>,
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
        for variant in &declaration.variants {
            let model_id = RecommendableModelId(format!("{}:{}", declaration.id, variant.format));
            let input = by_model_id.get(&model_id).ok_or_else(|| {
                InventoryError::Integrity(format!(
                    "planner bundle is missing catalog model {}",
                    model_id.0
                ))
            })?;
            let model = recommendable_model(
                declaration,
                variant,
                input.target.package.clone(),
                input.draft.as_ref().map(|draft| draft.package.clone()),
                &input.target.properties,
            )?;
            if !inputs.contains_key(&recommendable_model_bundle_key(&model)) {
                return Err(InventoryError::Integrity(format!(
                    "planner bundle key does not match catalog model {}",
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
        variant: &CatalogVariant,
        snapshots: &BTreeMap<String, HuggingFaceRepositorySnapshot>,
    ) -> Result<
        (
            RecommendableModel,
            ReleasePlannerInput,
            BTreeMap<String, Vec<u8>>,
            BTreeMap<String, Vec<u8>>,
        ),
        InventoryError,
    > {
        let snapshot = snapshots.get(&declaration.repository).ok_or_else(|| {
            InventoryError::Integrity(format!(
                "target repository {} was not resolved",
                declaration.repository
            ))
        })?;
        let selector = variant.format.to_ascii_lowercase();
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
                "{} format {} resolved to {} primary files",
                declaration.repository,
                variant.format,
                matches.len()
            )));
        }
        let primary = matches.remove(0);
        let target_prepared = self
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
        let (target, mut headers, mut planner_stubs) =
            self.release_planner_package(target_prepared, &primary.path)?;
        let draft = match declaration
            .speculative_decoding
            .as_ref()
            .and_then(|speculative| speculative.draft.file(&declaration.repository))
        {
            Some((draft_repository, draft_path)) => {
                let snapshot = snapshots.get(draft_repository).ok_or_else(|| {
                    InventoryError::Integrity(format!(
                        "draft repository {} was not resolved",
                        draft_repository
                    ))
                })?;
                if !snapshot
                    .gguf_files
                    .iter()
                    .any(|file| file.path == draft_path)
                {
                    return Err(InventoryError::Integrity(format!(
                        "{} has no draft file {}",
                        draft_repository,
                        draft_path.display()
                    )));
                }
                let prepared = self
                    .models
                    .prepare_preview_from_repository_snapshot(
                        &ModelPreviewSource {
                            repository: snapshot.repository.clone(),
                            revision: snapshot.commit.clone(),
                            primary_gguf: draft_path.to_path_buf(),
                            additional_components: Vec::new(),
                        },
                        snapshot,
                    )
                    .await?;
                let (draft, draft_headers, draft_stubs) =
                    self.release_planner_package(prepared, draft_path)?;
                merge_content_map(&mut headers, draft_headers, "source header")?;
                merge_content_map(&mut planner_stubs, draft_stubs, "planner stub")?;
                Some(draft)
            }
            None => None,
        };
        let maximum_context_length =
            draft
                .as_ref()
                .map_or(target.package.properties.maximum_context_length, |draft| {
                    target
                        .package
                        .properties
                        .maximum_context_length
                        .min(draft.package.properties.maximum_context_length)
                });
        if declaration.context_length > maximum_context_length {
            return Err(InventoryError::Integrity(format!(
                "{} configures {} context tokens above the artifact maximum of {}",
                declaration.id, declaration.context_length, maximum_context_length
            )));
        }
        let model = recommendable_model(
            declaration,
            variant,
            target.package.clone(),
            draft.as_ref().map(|draft| draft.package.clone()),
            &target.properties,
        )?;
        let planner = ReleasePlannerInput {
            model_id: model.id.clone(),
            target,
            draft,
        };
        Ok((model, planner, headers, planner_stubs))
    }

    fn release_planner_package(
        &self,
        prepared: PreparedPreview,
        primary: &Path,
    ) -> Result<
        (
            ReleasePlannerPackage,
            BTreeMap<String, Vec<u8>>,
            BTreeMap<String, Vec<u8>>,
        ),
        InventoryError,
    > {
        let package = package_from_resolved(&prepared.model)?;
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
            .find(|header| header.path == primary)
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
                let kind = if component.path == primary {
                    PlannerStubComponent::Primary
                } else if component.role == ComponentRole::Shard {
                    PlannerStubComponent::Shard
                } else {
                    PlannerStubComponent::Companion
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
        let planner = ReleasePlannerPackage {
            package,
            properties: prepared.model.model.properties.clone(),
            primary_gguf: primary.to_path_buf(),
            components,
        };
        Ok((planner, headers, planner_stubs))
    }
}

fn merge_content_map(
    target: &mut BTreeMap<String, Vec<u8>>,
    source: BTreeMap<String, Vec<u8>>,
    kind: &str,
) -> Result<(), InventoryError> {
    for (digest, content) in source {
        if let Some(previous) = target.insert(digest.clone(), content.clone())
            && previous != content
        {
            return Err(InventoryError::Integrity(format!(
                "{kind} digest collision {digest}"
            )));
        }
    }
    Ok(())
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

fn insert_locked_repository(
    repositories: &mut BTreeMap<String, String>,
    repository: &str,
    revision: &str,
) -> Result<(), InventoryError> {
    if let Some(existing) = repositories.insert(repository.to_owned(), revision.to_owned())
        && existing != revision
    {
        return Err(InventoryError::Integrity(format!(
            "catalog repository {repository} is locked to conflicting revisions"
        )));
    }
    Ok(())
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
            self.resolve_release_catalog_from_lock(source, lock, progress)
                .await
        })
    }

    pub fn resolve_release_catalog_with_lock<F>(
        &self,
        lock: ModelCatalogLock,
        progress: F,
    ) -> BoxFuture<'_, Result<GeneratedReleaseCatalog, InventoryError>>
    where
        F: Fn(&str, usize, usize) + Send + Sync + 'static,
    {
        Box::pin(async move {
            let source = catalog_source()?;
            validate_model_catalog_lock(&lock, &source)?;
            self.resolve_release_catalog_from_lock(source, lock, progress)
                .await
        })
    }

    async fn resolve_release_catalog_from_lock<F>(
        &self,
        source: CatalogSource,
        lock: ModelCatalogLock,
        progress: F,
    ) -> Result<GeneratedReleaseCatalog, InventoryError>
    where
        F: Fn(&str, usize, usize) + Send + Sync + 'static,
    {
        let mut repositories = BTreeMap::new();
        for declaration in &source.models {
            let entry = lock.get(&declaration.id).ok_or_else(|| {
                InventoryError::Integrity(format!(
                    "model catalog lock is missing {}",
                    declaration.id
                ))
            })?;
            insert_locked_repository(&mut repositories, &declaration.repository, &entry.target)?;
            if let Some(speculative) = &declaration.speculative_decoding {
                if let Some((repository, _)) = speculative.draft.file(&declaration.repository) {
                    insert_locked_repository(
                        &mut repositories,
                        repository,
                        entry.speculative_draft.as_deref().ok_or_else(|| {
                            InventoryError::Integrity(format!(
                                "model catalog lock is missing {} speculative draft",
                                declaration.id
                            ))
                        })?,
                    )?;
                } else if entry.speculative_draft.is_some() {
                    return Err(InventoryError::Integrity(format!(
                        "model catalog lock unexpectedly includes {} embedded speculative draft",
                        declaration.id
                    )));
                }
            }
        }
        let repositories = repositories.into_iter().collect::<Vec<_>>();
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
        let resolved =
            stream::iter(source.models.into_iter().enumerate())
                .map(|(declaration_index, declaration)| async move {
                    let mut variants = Vec::with_capacity(declaration.variants.len());
                    for (variant_index, variant) in declaration.variants.iter().enumerate() {
                        let missing_repository = std::iter::once(declaration.repository.as_str())
                            .chain(declaration.speculative_decoding.iter().filter_map(
                                |speculative| {
                                    speculative
                                        .draft
                                        .file(&declaration.repository)
                                        .map(|(repository, _)| repository)
                                },
                            ))
                            .find(|repository| !resolved_snapshots.contains_key(*repository));
                        let result =
                            match missing_repository {
                                None => {
                                    self.resolve_model(&declaration, variant, resolved_snapshots)
                                        .await
                                }
                                Some(repository) => Err(InventoryError::Io(
                                    snapshot_failures.get(repository).cloned().unwrap_or_else(
                                        || format!("repository {repository} was not resolved"),
                                    ),
                                )),
                            };
                        variants.push((
                            declaration_index,
                            variant_index,
                            declaration.clone(),
                            variant.clone(),
                            result,
                        ));
                    }
                    variants
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
        resolved.sort_by_key(|(declaration_index, variant_index, ..)| {
            (*declaration_index, *variant_index)
        });
        let mut models = Vec::new();
        let mut planner_inputs = BTreeMap::new();
        let mut source_headers = BTreeMap::new();
        let mut planner_stubs = BTreeMap::new();
        let mut diagnostics = Vec::new();
        for (_, _, declaration, variant, result) in resolved {
            match result {
                Ok((model, planner, model_headers, model_stubs)) => {
                    planner_inputs.insert(recommendable_model_bundle_key(&model), planner);
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
                        if let Some(previous) = planner_stubs.insert(digest.clone(), stub.clone())
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
                        "{}:{}",
                        declaration.id, variant.format
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
        let variants = |id: &str| {
            catalog_source()
                .expect("catalog source")
                .models
                .iter()
                .find(|model| model.id == id)
                .expect("catalog model")
                .variants
                .clone()
        };
        let formats = |id: &str| {
            variants(id)
                .into_iter()
                .map(|variant| variant.format)
                .collect::<Vec<_>>()
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
            formats("muse-glimmer-30b"),
            ["UD-Q4_K_XL", "UD-Q5_K_XL", "UD-Q6_K_XL", "UD-Q8_K_XL"]
        );
        assert_eq!(
            formats("nemotron-3-super-120b-a12b"),
            ["UD-Q4_K_XL", "MXFP4_MOE"]
        );
        assert_eq!(
            formats("nemotron-3.5-lightning-30b-a3b"),
            ["NVFP4", "Q4_K_M", "Q8_0"]
        );
        let nvfp4 = variants("nemotron-3.5-lightning-30b-a3b")
            .into_iter()
            .find(|variant| variant.format == "NVFP4")
            .expect("NVFP4 variant");
        assert_eq!(nvfp4.variant_label, "NVFP4");
        assert_eq!((nvfp4.fidelity_rank, nvfp4.quantization_aware), (60, false));
        assert_eq!(formats("deepseek-v4-flash"), ["UD-Q4_K_XL", "UD-Q8_K_XL"]);
        assert_eq!(
            formats("glm-5.2"),
            ["UD-Q4_K_XL", "UD-Q5_K_XL", "UD-Q6_K_XL", "UD-Q8_K_XL"]
        );
        assert_eq!(
            variants("glm-5.2")
                .into_iter()
                .map(|variant| (variant.variant_label, variant.fidelity_rank))
                .collect::<Vec<_>>(),
            [
                ("Q4".to_owned(), 40),
                ("Q5".to_owned(), 50),
                ("Q6".to_owned(), 60),
                ("Q8".to_owned(), 80)
            ]
        );
    }

    #[test]
    fn small_model_catalog_uses_curated_profiles_and_formats() {
        let source = catalog_source().expect("catalog source");
        let model = |id: &str| {
            source
                .models
                .iter()
                .find(|model| model.id == id)
                .expect("catalog model")
        };
        assert_eq!(model("qwen3.5-4b").context_length, 50_000);
        assert_eq!(model("gemma-4-e4b-it-qat").context_length, 50_000);
        assert_eq!(model("gemma-4-12b-it-qat").context_length, 100_000);
        let [gemma] = model("gemma-4-12b-it-qat").variants.as_slice() else {
            panic!("Gemma QAT model must declare one variant");
        };
        assert_eq!(gemma.variant_label, "Q4 QAT");
        assert_eq!((gemma.fidelity_rank, gemma.quantization_aware), (58, true));
        assert_eq!(
            model("lfm2.5-2.6b")
                .variants
                .iter()
                .map(|variant| variant.format.as_str())
                .collect::<Vec<_>>(),
            ["Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0"]
        );
        let [bonsai] = model("bonsai-8b-q1").variants.as_slice() else {
            panic!("bonsai must declare one variant");
        };
        assert_eq!(bonsai.format, "Q1_0");
        assert_eq!(bonsai.variant_label, "Q1");
        assert_eq!(
            (bonsai.fidelity_rank, bonsai.quantization_aware),
            (10, false)
        );
    }

    #[test]
    fn gemma_catalog_declares_same_repository_mtp_drafts() {
        let source = catalog_source().expect("catalog source");
        let lock = model_catalog_lock().expect("catalog lock");
        for model in source
            .models
            .iter()
            .filter(|model| model.id.starts_with("gemma-4-"))
        {
            let Some(speculative) = model.speculative_decoding.as_ref() else {
                panic!("{} must declare speculative decoding", model.id);
            };
            assert!(matches!(speculative.method, CatalogSpeculativeMethod::Mtp));
            let CatalogSpeculativeDraftSource::File { repository, path } = &speculative.draft
            else {
                panic!("{} must declare a file draft", model.id);
            };
            assert_eq!(repository, &None);
            assert!(
                path.file_name()
                    .is_some_and(|name| { name.to_string_lossy().starts_with("mtp-gemma-4-") })
            );
            let entry = lock.get(&model.id).expect("Gemma lock entry");
            assert_eq!(
                entry.speculative_draft.as_deref(),
                Some(entry.target.as_str())
            );
        }
    }

    #[test]
    fn catalog_declares_verified_embedded_mtp_models_explicitly() {
        let source = catalog_source().expect("catalog source");
        for id in [
            "qwen3.5-4b",
            "qwen3.5-9b",
            "qwen3.5-122b-a10b",
            "nemotron-3.5-lightning-30b-a3b",
            "glm-5.2",
        ] {
            let model = source
                .models
                .iter()
                .find(|model| model.id == id)
                .unwrap_or_else(|| panic!("missing {id} catalog model"));
            let speculative = model
                .speculative_decoding
                .as_ref()
                .unwrap_or_else(|| panic!("{id} must declare speculative decoding"));
            assert!(matches!(speculative.method, CatalogSpeculativeMethod::Mtp));
            assert!(matches!(
                speculative.draft,
                CatalogSpeculativeDraftSource::Embedded
            ));
        }
    }

    #[test]
    fn qwen_36_variants_share_their_dflash_drafts() {
        let source = catalog_source().expect("catalog source");
        let assert_dflash = |id: &str, repository: &str, path: &str| {
            let model = source
                .models
                .iter()
                .find(|model| model.id == id)
                .unwrap_or_else(|| panic!("missing {id} catalog model"));
            let speculative = model
                .speculative_decoding
                .as_ref()
                .unwrap_or_else(|| panic!("{id} must use speculative decoding"));

            assert_eq!(model.variants.len(), 4);
            assert!(matches!(
                speculative.method,
                CatalogSpeculativeMethod::DFlash
            ));
            let CatalogSpeculativeDraftSource::File {
                repository: declared_repository,
                path: declared_path,
            } = &speculative.draft
            else {
                panic!("{id} must declare a file draft");
            };
            assert_eq!(declared_repository.as_deref(), Some(repository));
            assert_eq!(declared_path, Path::new(path));
        };

        assert_dflash(
            "qwen3.6-27b",
            "magnitudedev/Qwen3.6-27B-DFlash-GGUF",
            "Qwen3.6-27B-DFlash-Q8_0.gguf",
        );
        assert_dflash(
            "qwen3.6-35b-a3b",
            "magnitudedev/Qwen3.6-35B-A3B-DFlash-GGUF",
            "Qwen3.6-35B-A3B-DFlash-Q8_0.gguf",
        );
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
        let commit = model_catalog_lock().expect("model catalog lock")[&declaration.id]
            .target
            .clone();
        let package_source =
            |repository: String, revision: String| ModelPackageSource::HuggingFace {
                repository,
                revision,
            };
        assert!(package_source_matches(
            &package_source(declaration.repository.clone(), commit.clone()),
            &declaration.repository,
            &commit,
        ));
        assert!(!package_source_matches(
            &package_source("other/repository".to_owned(), commit.clone()),
            &declaration.repository,
            &commit,
        ));
        assert!(!package_source_matches(
            &package_source(declaration.repository.clone(), "0".repeat(40)),
            &declaration.repository,
            &commit,
        ));
    }
}
