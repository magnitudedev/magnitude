use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use futures_util::future::BoxFuture;
use futures_util::{StreamExt, stream};
use magnitude_artifacts::catalog_headers::{
    CatalogComponent, ComponentRole as HeaderRole, lock_digest,
};
use magnitude_service_contracts::ComponentRole;
use magnitude_service_contracts::models::{
    CatalogBaseId, CatalogDiagnostic, CatalogQuant, CatalogSupport, CatalogVariantId, LabId,
    ModelDeveloper, ModelFailure, ModelFamily, ModelFamilyStatus, ModelFileRole, ModelId,
    ModelPackage, ModelPackageSource, ModelParameterization,
    ModelReleaseDate, ModelServingConfiguration, PackageValidation, ParsedModelId,
    RecommendableModel, RecommendableModelCatalog, RecommendableModelCatalogProvider,
    ResolvedServableModelBundle, ServableModelBundle, ServingProfile, SpeculativeDraftSource,
    SpeculativeMethod,
};
use magnitude_service_contracts::{
    ContentId, HuggingFaceRepositoryFile, HuggingFaceRepositoryRequest,
    HuggingFaceRepositorySnapshot, Integrity, InventoryEntryId, InventoryError, InventoryModel,
    InventoryProperties, ModelAvailability, ModelComponent, ModelLocation,
    ModelPreviewComponentRole, ModelPreviewComponentSource, ModelPreviewSource, ModelSource,
    ResolvedComponent, ResolvedModel,
};
use serde::{Deserialize, Serialize};

use crate::developers::ModelDevelopers;
use crate::inventory::ManagedModelStore;
use crate::package_service::{
    ServableModelBundleKey, servable_model_bundle_key_for_bundle, validated_package_from_resolved,
};
use crate::preview::PreparedPreview;
use crate::weight_lineage::WeightSample;
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
    intelligence_frontier: IntelligenceFrontier,
    labs: Vec<CatalogLab>,
    families: Vec<CatalogFamily>,
    models: Vec<CatalogModel>,
}

/// A family of models Magnitude runs: its display name, its lab and the GGUF architectures its
/// models declare. Each architecture the engine serves as a target belongs to exactly one family.
#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CatalogFamily {
    name: String,
    lab: String,
    architectures: Vec<String>,
    status: CatalogFamilyStatus,
}

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "camelCase")]
enum CatalogFamilyStatus {
    Supported,
    ComingSoon,
}

/// A lab that develops catalog models: its display name, its country (ISO 3166-1 alpha-2) and the
/// Hugging Face organizations it publishes under, which attribute models outside the catalog.
#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CatalogLab {
    id: String,
    name: String,
    country: String,
    hugging_face_organizations: Vec<String>,
}

impl CatalogLab {
    fn developer(&self) -> ModelDeveloper {
        ModelDeveloper {
            lab: LabId(self.id.clone()),
            name: self.name.clone(),
            country: self.country.clone(),
        }
    }
}

impl CatalogSource {
    /// The developer of a catalog model, whose lab `catalog_source` requires the table to hold.
    fn developer(&self, model: &CatalogModel) -> ModelDeveloper {
        self.lab_developer(&model.lab)
    }

    fn lab_developer(&self, lab: &str) -> ModelDeveloper {
        self.labs
            .iter()
            .find(|candidate| candidate.id == lab)
            .expect("catalog_source requires every lab a model or family names in the lab table")
            .developer()
    }

    fn families(&self) -> Vec<ModelFamily> {
        self.families
            .iter()
            .map(|family| ModelFamily {
                name: family.name.clone(),
                developer: self.lab_developer(&family.lab),
                architectures: family.architectures.clone(),
                status: match family.status {
                    CatalogFamilyStatus::Supported => ModelFamilyStatus::Supported,
                    CatalogFamilyStatus::ComingSoon => ModelFamilyStatus::ComingSoon,
                },
            })
            .collect()
    }

    fn developers(&self) -> ModelDevelopers {
        ModelDevelopers::new(
            self.labs
                .iter()
                .map(|lab| (lab.developer(), lab.hugging_face_organizations.clone())),
            self.models
                .iter()
                .filter_map(|model| {
                    Some((CatalogBaseId::new(model.id.clone()).ok()?, LabId(model.lab.clone())))
                })
                .collect(),
        )
    }
}

/// The developers of the compiled catalog, whose source its tests validate.
pub fn model_developers() -> &'static ModelDevelopers {
    static DEVELOPERS: std::sync::LazyLock<ModelDevelopers> = std::sync::LazyLock::new(|| {
        catalog_source()
            .expect("the compiled catalog source is valid")
            .developers()
    });
    &DEVELOPERS
}

/// The model families of the compiled catalog, whose source its tests validate.
pub fn model_families() -> &'static [ModelFamily] {
    static FAMILIES: std::sync::LazyLock<Vec<ModelFamily>> = std::sync::LazyLock::new(|| {
        catalog_source()
            .expect("the compiled catalog source is valid")
            .families()
    });
    &FAMILIES
}

/// The frontier model on Epoch AI's Capabilities Index (ECI) and the benchmarks catalog intelligence
/// is measured on: those Epoch rates that were released on or after `benchmarks_since`. A model's
/// intelligence is its expected score on them as a whole percentage of the frontier model's
/// (`expected_benchmark_score`), so the frontier is 100.
#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct IntelligenceFrontier {
    model: String,
    eci: f64,
    as_of_date: String,
    benchmarks_since: String,
    url: String,
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct IntelligenceScore {
    eci: f64,
    percent: u32,
    provenance: IntelligenceProvenance,
}

/// Where a model's ECI comes from.
#[derive(Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase", deny_unknown_fields)]
enum IntelligenceProvenance {
    /// Epoch's published ECI for the model.
    EpochPublished { url: String },
    /// Fitted with Epoch's benchmark difficulties to Epoch's own benchmark results for the model.
    EpochData { benchmarks: Vec<String>, url: String },
    /// Predicted from other evaluations of the model, calibrated on models Epoch also rates.
    #[serde(rename_all = "camelCase")]
    Bridged {
        method: String,
        evidence_urls: Vec<String>,
    },
}

/// One benchmark's fitted difficulty and slope on the ECI scale, from Epoch's `edi_scores.csv`.
struct BenchmarkDifficulty {
    difficulty: f64,
    slope: f64,
}

/// The mean expected score, from 0 to 1, of a model at `eci` on `benchmarks`: Epoch's model
/// predicts `1 / (1 + e^(-slope * (eci - difficulty)))` on each.
fn expected_benchmark_score(eci: f64, benchmarks: &[BenchmarkDifficulty]) -> f64 {
    benchmarks
        .iter()
        .map(|benchmark| 1.0 / (1.0 + (-benchmark.slope * (eci - benchmark.difficulty)).exp()))
        .sum::<f64>()
        / benchmarks.len() as f64
}

/// A catalog model's declared intelligence beside the one its ECI gives on the frontier's benchmarks.
#[derive(Debug, Clone, PartialEq)]
pub struct IntelligenceCheck {
    pub model: String,
    pub eci: f64,
    pub declared: u32,
    pub computed: u32,
}

/// Recomputes every scored catalog model's intelligence from its ECI and Epoch's benchmark
/// difficulties (`edi_scores.csv`: `benchmark_name,is_anchor,benchmark_release_date,edi,
/// estimated_slope_scaled`), keeping the benchmarks released on or after the frontier's
/// `benchmarks_since`.
pub fn catalog_intelligence_check(
    benchmark_difficulties: &str,
) -> Result<Vec<IntelligenceCheck>, InventoryError> {
    let source = catalog_source()?;
    let frontier = &source.intelligence_frontier;
    let invalid = |message: String| InventoryError::Integrity(message);
    let mut lines = benchmark_difficulties.lines();
    let header = lines.next().unwrap_or_default();
    if header.trim() != "benchmark_name,is_anchor,benchmark_release_date,edi,estimated_slope_scaled" {
        return Err(invalid(format!("unexpected benchmark difficulty header: {header}")));
    }
    let mut benchmarks = Vec::new();
    for line in lines.filter(|line| !line.trim().is_empty()) {
        let fields: Vec<&str> = line.split(',').collect();
        let [_, _, released, difficulty, slope] = fields.as_slice() else {
            return Err(invalid(format!("malformed benchmark difficulty row: {line}")));
        };
        let parse = |value: &str| {
            value
                .trim()
                .parse::<f64>()
                .map_err(|_| invalid(format!("malformed benchmark difficulty row: {line}")))
        };
        if released.trim() >= frontier.benchmarks_since.as_str() {
            benchmarks.push(BenchmarkDifficulty {
                difficulty: parse(difficulty)?,
                slope: parse(slope)?,
            });
        }
    }
    if benchmarks.is_empty() {
        return Err(invalid(format!(
            "no rated benchmark was released on or after {}",
            frontier.benchmarks_since
        )));
    }
    let frontier_score = expected_benchmark_score(frontier.eci, &benchmarks);
    Ok(source
        .models
        .iter()
        .filter_map(|model| {
            model.intelligence.as_ref().map(|intelligence| IntelligenceCheck {
                model: model.id.clone(),
                eci: intelligence.eci,
                declared: intelligence.percent,
                computed: (100.0 * expected_benchmark_score(intelligence.eci, &benchmarks)
                    / frontier_score)
                    .round() as u32,
            })
        })
        .collect())
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CatalogModel {
    id: String,
    display_name: String,
    lab: String,
    description: String,
    release_date: ModelReleaseDate,
    parameterization: ModelParameterization,
    sources: Vec<CatalogModelSource>,
    #[serde(default)]
    speculative_decoding: Option<CatalogSpeculativeDecoding>,
    /// Required of every supported model (`valid_model_score`).
    #[serde(default)]
    intelligence: Option<IntelligenceScore>,
    support: DeclaredSupport,
}

/// What the release promises for a catalog model, as the catalog declares it beside the model's
/// score; the service's `CatalogSupport` carries the two together.
#[derive(Clone, Deserialize)]
#[serde(tag = "level", rename_all = "camelCase", deny_unknown_fields)]
enum DeclaredSupport {
    Supported,
    Disabled {
        reason: String,
    },
    #[serde(rename_all = "camelCase")]
    Deprecated {
        since: String,
        replacement: ModelId,
        reason: String,
    },
}

impl DeclaredSupport {
    fn is_deprecated(&self) -> bool {
        matches!(self, Self::Deprecated { .. })
    }

    fn is_available(&self) -> bool {
        matches!(self, Self::Supported)
    }
}

impl CatalogModel {
    /// The model's support with its intelligence, which `catalog_source` requires of every supported
    /// model (`valid_model_score`).
    fn rated_support(&self) -> CatalogSupport {
        let intelligence = self.intelligence.as_ref().map(|intelligence| intelligence.percent);
        match &self.support {
            DeclaredSupport::Supported => CatalogSupport::Supported {
                intelligence: intelligence.expect("catalog_source scores every supported model"),
            },
            DeclaredSupport::Disabled { reason } => CatalogSupport::Disabled {
                reason: reason.clone(),
                intelligence,
            },
            DeclaredSupport::Deprecated {
                since,
                replacement,
                reason,
            } => CatalogSupport::Deprecated {
                since: since.clone(),
                replacement: replacement.clone(),
                reason: reason.clone(),
                intelligence,
            },
        }
    }

    /// Every variant, with the source that publishes it.
    fn variants(&self) -> impl Iterator<Item = (&CatalogModelSource, &CatalogVariant)> {
        self.sources
            .iter()
            .flat_map(|source| source.variants.iter().map(move |variant| (source, variant)))
    }

    /// Every repository a variant of the model resolves from: its sources' and a separately
    /// packaged draft's.
    fn repositories(&self) -> impl Iterator<Item = &str> {
        self.sources
            .iter()
            .map(|source| source.repository.as_str())
            .chain(self.draft_repository())
    }

    fn draft_repository(&self) -> Option<&str> {
        self.speculative_decoding
            .as_ref()
            .and_then(|speculative| speculative.draft.file())
            .map(|(repository, _)| repository)
    }
}

/// One repository's publication of a model: the variants it hosts, the projector they share and
/// the license it grants.
#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CatalogModelSource {
    repository: String,
    variants: Vec<CatalogVariant>,
    #[serde(default)]
    projector: Option<CatalogProjectorSource>,
    license: String,
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CatalogVariant {
    variant_id: String,
    format: String,
    #[serde(default)]
    path: Option<PathBuf>,
    variant_label: String,
    quantization_aware: bool,
}

#[derive(Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CatalogProjectorSource {
    path: PathBuf,
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
    File { repository: String, path: PathBuf },
}

#[derive(Clone, Copy, Deserialize, PartialEq, Eq)]
enum CatalogSpeculativeMethod {
    #[serde(rename = "mtp")]
    Mtp,
    #[serde(rename = "dflash")]
    DFlash,
    #[serde(rename = "dflash2")]
    DFlash2,
    #[serde(rename = "dspark")]
    DSpark,
    #[serde(rename = "lilicorr")]
    LiLiCorr,
}

impl From<CatalogSpeculativeMethod> for SpeculativeMethod {
    fn from(method: CatalogSpeculativeMethod) -> Self {
        match method {
            CatalogSpeculativeMethod::Mtp => Self::Mtp,
            CatalogSpeculativeMethod::DFlash => Self::DFlash,
            CatalogSpeculativeMethod::DFlash2 => Self::DFlash2,
            CatalogSpeculativeMethod::DSpark => Self::DSpark,
            CatalogSpeculativeMethod::LiLiCorr => Self::LiLiCorr,
        }
    }
}

impl CatalogSpeculativeDraftSource {
    fn file(&self) -> Option<(&str, &Path)> {
        match self {
            Self::Embedded => None,
            Self::File { repository, path } => Some((repository, path)),
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
    model_id: CatalogBaseId,
    variant_id: CatalogVariantId,
    target: ReleasePlannerPackage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    draft: Option<ReleasePlannerPackage>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ReleasePlannerPackage {
    package: ModelPackage,
    validation: PackageValidation,
    properties: InventoryProperties,
    primary_gguf: PathBuf,
    components: Vec<ReleasePlannerComponent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    weight_sample: Option<ReleasePlannerWeightSample>,
}

/// A target's weight sample (see [`WeightSample`]), stored in the bundle as an input of its own,
/// so the variants of one model, whose sampled weights are the same bytes, share it.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ReleasePlannerWeightSample {
    digest: String,
    size_bytes: u64,
}

impl ReleasePlannerPackage {
    /// Every bundle input the package names, with its declared size.
    fn inputs(&self) -> impl Iterator<Item = (&str, u64)> {
        self.components
            .iter()
            .map(|component| (component.header_digest.as_str(), component.header_size_bytes))
            .chain(
                self.weight_sample
                    .iter()
                    .map(|sample| (sample.digest.as_str(), sample.size_bytes)),
            )
    }
}

impl ReleasePlannerInput {
    fn packages(&self) -> impl Iterator<Item = &ReleasePlannerPackage> {
        std::iter::once(&self.target).chain(self.draft.iter())
    }
}

/// The immutable commit of every repository the catalog resolves from, by repository.
pub type ModelCatalogLock = BTreeMap<String, String>;

/// One package component's assessment material: its exact GGUF header (every byte through the
/// aligned tensor-data offset), which is what the engine's header-only planning and chat
/// template inspection read.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ReleasePlannerComponent {
    component: ModelComponent,
    header_digest: String,
    header_size_bytes: u64,
}

pub struct GeneratedReleaseCatalog {
    pub catalog: RecommendableModelCatalog,
    planner_inputs: BTreeMap<ServableModelBundleKey, ReleasePlannerInput>,
    headers: BTreeMap<String, Vec<u8>>,
}

#[derive(Clone)]
pub struct ReleaseCatalog {
    weight_samples: Arc<BTreeMap<ServableModelBundleKey, WeightSample>>,
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
    if !valid_intelligence_frontier(&source.intelligence_frontier) {
        return Err(InventoryError::Integrity(
            "invalid intelligence frontier declaration".to_owned(),
        ));
    }
    if !valid_labs(&source.labs) {
        return Err(InventoryError::Integrity(
            "invalid or duplicate catalog lab declaration".to_owned(),
        ));
    }
    if !valid_families(&source.families, &source.labs) {
        return Err(InventoryError::Integrity(
            "invalid or duplicate catalog family declaration".to_owned(),
        ));
    }
    let mut ids = BTreeSet::new();
    let mut presentations = BTreeSet::new();
    for model in &source.models {
        let variant_count = model.variants().count();
        let variant_ids = model
            .variants()
            .map(|(_, variant)| variant.variant_id.as_str())
            .collect::<BTreeSet<_>>();
        let variant_labels = model
            .variants()
            .map(|(_, variant)| variant.variant_label.as_str())
            .collect::<BTreeSet<_>>();
        let repositories = model
            .sources
            .iter()
            .map(|source| source.repository.as_str())
            .collect::<BTreeSet<_>>();
        if !valid_identity_component(&model.id)
            || model.display_name.is_empty()
            || model.description.is_empty()
            || !valid_parameterization(&model.parameterization)
            || model.sources.is_empty()
            || repositories.len() != model.sources.len()
            || model.sources.iter().any(|source| {
                // A format selects its variant's primary GGUF among its repository's files.
                let formats = source
                    .variants
                    .iter()
                    .map(|variant| variant.format.as_str())
                    .collect::<BTreeSet<_>>();
                source.repository.is_empty()
                    || source.variants.is_empty()
                    || formats.len() != source.variants.len()
                    || source
                        .projector
                        .as_ref()
                        .is_some_and(|projector| projector.path.as_os_str().is_empty())
                    || source.license.is_empty()
            })
            || variant_ids.len() != variant_count
            || variant_labels.len() != variant_count
            || model.variants().any(|(_, variant)| {
                !valid_variant_id(&variant.variant_id)
                    || variant.format.is_empty()
                    || variant.format.trim() != variant.format.as_str()
                    || variant
                        .path
                        .as_ref()
                        .is_some_and(|path| path.as_os_str().is_empty())
                    || variant.variant_label.is_empty()
                    || variant.variant_label.trim() != variant.variant_label.as_str()
                    || variant.variant_label.contains('(')
                    || variant.variant_label.contains(')')
            })
            || model
                .speculative_decoding
                .as_ref()
                .is_some_and(|speculative| {
                    matches!(
                        (&speculative.method, &speculative.draft),
                        (
                            CatalogSpeculativeMethod::DFlash
                                | CatalogSpeculativeMethod::DFlash2
                                | CatalogSpeculativeMethod::DSpark
                                | CatalogSpeculativeMethod::LiLiCorr,
                            CatalogSpeculativeDraftSource::Embedded
                        )
                    ) || match &speculative.draft {
                        CatalogSpeculativeDraftSource::Embedded => false,
                        CatalogSpeculativeDraftSource::File { repository, path } => {
                            repository.is_empty() || path.as_os_str().is_empty()
                        }
                    }
                })
            || !valid_model_score(model, &source.intelligence_frontier)
            || !source.labs.iter().any(|lab| lab.id == model.lab)
            || !ids.insert(model.id.as_str())
            || model.variants().any(|(_, variant)| {
                !presentations.insert((model.display_name.as_str(), variant.variant_label.as_str()))
            })
        {
            return Err(InventoryError::Integrity(format!(
                "invalid or duplicate catalog declaration {}",
                model.id
            )));
        }
    }
    if let Some(model) = source
        .models
        .iter()
        .find(|model| !valid_catalog_support(model, &source))
    {
        return Err(InventoryError::Integrity(format!(
            "invalid catalog support declaration {}",
            model.id
        )));
    }
    Ok(source)
}

/// Every lab has an identity-component ID, a name, an ISO 3166-1 alpha-2 country and at least one
/// Hugging Face organization; no two labs share an ID or an organization (ignoring case).
fn valid_labs(labs: &[CatalogLab]) -> bool {
    let mut ids = BTreeSet::new();
    let mut organizations = BTreeSet::new();
    !labs.is_empty()
        && labs.iter().all(|lab| {
            valid_identity_component(&lab.id)
                && ids.insert(lab.id.as_str())
                && valid_non_empty(&lab.name)
                && lab.country.len() == 2
                && lab.country.bytes().all(|byte| byte.is_ascii_uppercase())
                && !lab.hugging_face_organizations.is_empty()
                && lab.hugging_face_organizations.iter().all(|organization| {
                    valid_identity_component(organization)
                        && !organization.contains('/')
                        && organizations.insert(organization.to_lowercase())
                })
        })
}

/// Every family has a unique name, a lab in the lab table and at least one architecture, and no
/// architecture belongs to two families.
fn valid_families(families: &[CatalogFamily], labs: &[CatalogLab]) -> bool {
    let mut names = BTreeSet::new();
    let mut architectures = BTreeSet::new();
    !families.is_empty()
        && families.iter().all(|family| {
            valid_non_empty(&family.name)
                && names.insert(family.name.as_str())
                && labs.iter().any(|lab| lab.id == family.lab)
                && !family.architectures.is_empty()
                && family.architectures.iter().all(|architecture| {
                    valid_non_empty(architecture) && architectures.insert(architecture.as_str())
                })
        })
}

/// A deprecation names its date and reason, and its replacement is an exact supported catalog
/// configuration of another model.
fn valid_catalog_support(model: &CatalogModel, source: &CatalogSource) -> bool {
    match &model.support {
        DeclaredSupport::Supported => true,
        DeclaredSupport::Disabled { reason } => valid_non_empty(reason),
        DeclaredSupport::Deprecated {
            since,
            replacement,
            reason,
        } => {
            ModelReleaseDate::new(since.clone()).is_ok()
                && valid_non_empty(reason)
                && {
                    let ParsedModelId::Catalog {
                        base_id,
                        variant_id,
                    } = replacement.parsed()
                    else {
                        return false;
                    };
                    base_id.as_str() != model.id
                        && source.models.iter().any(|candidate| {
                            candidate.id == base_id.as_str()
                                && candidate.support.is_available()
                                && candidate
                                    .variants()
                                    .any(|(_, variant)| variant.variant_id == variant_id.as_str())
                        })
                }
        }
    }
}

/// Every supported model has a score; a disabled or deprecated one, never ranked, may have none, and
/// a score it has is valid.
fn valid_model_score(model: &CatalogModel, frontier: &IntelligenceFrontier) -> bool {
    match &model.intelligence {
        Some(intelligence) => valid_intelligence(intelligence, frontier),
        None => !model.support.is_available(),
    }
}

fn valid_intelligence_frontier(frontier: &IntelligenceFrontier) -> bool {
    frontier.eci.is_finite()
        && frontier.eci > 0.0
        && valid_non_empty(&frontier.model)
        && ModelReleaseDate::new(frontier.as_of_date.clone()).is_ok()
        && ModelReleaseDate::new(frontier.benchmarks_since.clone()).is_ok()
        && frontier.benchmarks_since < frontier.as_of_date
        && valid_https_url(&frontier.url)
}

/// A model's ECI never exceeds the frontier's, its percentage never exceeds 100, and its provenance
/// names its evidence.
fn valid_intelligence(intelligence: &IntelligenceScore, frontier: &IntelligenceFrontier) -> bool {
    if !intelligence.eci.is_finite()
        || intelligence.eci <= 0.0
        || intelligence.eci > frontier.eci
        || intelligence.percent > 100
    {
        return false;
    }
    match &intelligence.provenance {
        IntelligenceProvenance::EpochPublished { url } => valid_https_url(url),
        IntelligenceProvenance::EpochData { benchmarks, url } => {
            !benchmarks.is_empty()
                && benchmarks.iter().all(|benchmark| valid_non_empty(benchmark))
                && valid_https_url(url)
        }
        IntelligenceProvenance::Bridged {
            method,
            evidence_urls,
        } => {
            valid_non_empty(method)
                && !evidence_urls.is_empty()
                && evidence_urls.iter().all(|url| valid_https_url(url))
        }
    }
}

fn valid_non_empty(value: &str) -> bool {
    !value.is_empty() && value.trim() == value
}

fn valid_https_url(value: &str) -> bool {
    valid_non_empty(value)
        && value.starts_with("https://")
        && !value.chars().any(char::is_whitespace)
}

pub(crate) fn is_projector_path(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
        && path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name.to_ascii_lowercase().contains("mmproj"))
}

fn resolve_projector_path(
    declaration: &CatalogModel,
    source: &CatalogModelSource,
    snapshot: &HuggingFaceRepositorySnapshot,
) -> Result<Option<PathBuf>, InventoryError> {
    let projector = if let Some(projector) = &source.projector {
        if !is_projector_path(&projector.path)
            || !snapshot
                .gguf_files
                .iter()
                .any(|file| file.path == projector.path)
        {
            return Err(InventoryError::Integrity(format!(
                "{} projector {} is not a repository mmproj GGUF",
                declaration.id,
                projector.path.display()
            )));
        }
        Some(projector.path.clone())
    } else {
        let candidates = snapshot
            .gguf_files
            .iter()
            .filter(|file| is_projector_path(&file.path))
            .map(|file| file.path.clone())
            .collect::<Vec<_>>();
        match candidates.as_slice() {
            [projector] => Some(projector.clone()),
            [] => None,
            _ => {
                return Err(InventoryError::Integrity(format!(
                    "{} has {} projector candidates; declare an exact projector path",
                    declaration.id,
                    candidates.len()
                )));
            }
        }
    };
    // An embedded MTP head embeds every prompt token, so it cannot enter the
    // projector's media rows. A separately packaged assistant reads the
    // target's history, media rows included, and keeps none of its own.
    if projector.is_some()
        && declaration
            .speculative_decoding
            .as_ref()
            .is_some_and(|speculative| {
                speculative.method == CatalogSpeculativeMethod::Mtp
                    && matches!(speculative.draft, CatalogSpeculativeDraftSource::Embedded)
            })
    {
        return Err(InventoryError::Integrity(format!(
            "{} combines a projector with embedded MTP, which cannot enter projector media rows",
            declaration.id
        )));
    }
    Ok(projector)
}

fn valid_parameterization(parameterization: &ModelParameterization) -> bool {
    match parameterization {
        ModelParameterization::Dense { total_parameters } => *total_parameters > 0,
        ModelParameterization::MixtureOfExperts {
            total_parameters,
            active_parameters,
        } => *active_parameters > 0 && active_parameters < total_parameters,
    }
}

pub(crate) fn valid_identity_component(value: &str) -> bool {
    !value.is_empty() && value.trim() == value && !value.contains(':')
}

pub(crate) fn valid_variant_id(value: &str) -> bool {
    let mut components = value.split(':');
    matches!(
        (components.next(), components.next(), components.next()),
        (Some(format), Some(quality), None)
            if valid_identity_component(format) && valid_identity_component(quality)
    )
}

/// The compiled catalog lock's identity (`magnitude_artifacts::catalog_headers::lock_digest`).
pub fn catalog_lock_digest() -> String {
    lock_digest(CATALOG_LOCK.as_bytes())
}

/// The SHA-256 of the curated catalog source (`catalog/models.json`) this build holds, in hex.
pub fn catalog_source_digest() -> String {
    lock_digest(CATALOG_SOURCE.as_bytes())
}

pub fn model_catalog_lock() -> Result<ModelCatalogLock, InventoryError> {
    model_catalog_lock_from(CATALOG_LOCK.as_bytes(), &catalog_source()?)
}

/// Which catalog entries a lock update resolves to their repositories' current commits. Every
/// other repository keeps its locked revision.
pub enum CatalogLockAdvance {
    All,
    Models(BTreeSet<String>),
}

/// Advance the selected catalog entries' repositories to their current commits. A repository a
/// deprecated model resolves from is never advanced: its locked revision is what keeps existing
/// installs identifiable.
pub async fn advance_model_catalog_lock(
    models: Arc<ManagedModelStore>,
    advance: CatalogLockAdvance,
) -> Result<ModelCatalogLock, InventoryError> {
    let source = catalog_source()?;
    // The compiled lock need not cover repositories declared since it was written, so only its
    // syntax is checked here; the advanced lock is validated in full below.
    let current: ModelCatalogLock = serde_json::from_str(CATALOG_LOCK).map_err(|error| {
        InventoryError::Integrity(format!("invalid model catalog lock: {error}"))
    })?;
    if let CatalogLockAdvance::Models(ids) = &advance
        && let Some(id) = ids.iter().find(|id| {
            source
                .models
                .iter()
                .find(|model| model.id == **id)
                .is_none_or(|model| model.support.is_deprecated())
        })
    {
        return Err(InventoryError::InvalidRequest(format!(
            "{id} is not an advanceable catalog model; deprecated models keep their locked revisions"
        )));
    }
    let kept = source
        .models
        .iter()
        .filter(|model| model.support.is_deprecated())
        .flat_map(CatalogModel::repositories)
        .collect::<BTreeSet<_>>();
    let advanced = source
        .models
        .iter()
        .filter(|model| match &advance {
            CatalogLockAdvance::All => true,
            CatalogLockAdvance::Models(ids) => ids.contains(&model.id),
        })
        .flat_map(CatalogModel::repositories)
        .filter(|repository| !kept.contains(repository))
        .collect::<BTreeSet<_>>();
    let repositories = source
        .models
        .iter()
        .flat_map(CatalogModel::repositories)
        .collect::<BTreeSet<_>>();
    let resolved = stream::iter(repositories)
        .map(|repository| {
            let models = Arc::clone(&models);
            let advanced = advanced.contains(repository);
            let locked = current.get(repository).cloned();
            async move {
                if !advanced {
                    return locked
                        .map(|commit| (repository.to_owned(), commit))
                        .ok_or_else(|| {
                            InventoryError::Integrity(format!(
                                "catalog repository {repository} has no locked revision to keep"
                            ))
                        });
                }
                let snapshot = refresh_hugging_face_repository(
                    &models,
                    HuggingFaceRepositoryRequest {
                        repository: repository.to_owned(),
                        revision: "main".to_owned(),
                    },
                )
                .await
                .map_err(|error| {
                    InventoryError::Upstream(format!(
                        "failed to resolve catalog repository {repository}: {error}"
                    ))
                })?;
                Ok::<_, InventoryError>((repository.to_owned(), snapshot.commit))
            }
        })
        .buffer_unordered(12)
        .collect::<Vec<_>>()
        .await;
    let lock = resolved.into_iter().collect::<Result<ModelCatalogLock, _>>()?;
    validate_model_catalog_lock(&lock, &source)?;
    Ok(lock)
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
        .flat_map(CatalogModel::repositories)
        .collect::<BTreeSet<_>>();
    if !lock.keys().map(String::as_str).eq(expected) {
        return Err(InventoryError::Integrity(
            "model catalog lock does not exactly cover the repositories of models.json".to_owned(),
        ));
    }
    if lock.values().any(|commit| !valid_commit(commit)) {
        return Err(InventoryError::Integrity(
            "model catalog lock contains invalid package revisions".to_owned(),
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
        for (digest, size) in input.packages().flat_map(ReleasePlannerPackage::inputs) {
            if let Some(previous_size) = expected_inputs.insert(digest, size)
                && previous_size != size
            {
                return Err(InventoryError::Integrity(format!(
                    "planner input {digest} has inconsistent declared sizes"
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
        if planner_bundle
            .input_len(digest)
            .and_then(|length| u64::try_from(length).ok())
            != Some(expected_size)
        {
            return Err(InventoryError::Integrity(format!(
                "planner input {digest} does not match its manifest size"
            )));
        }
    }
    let mut weight_samples = BTreeMap::new();
    for (bundle_key, input) in &manifest.planner_inputs {
        if let Some(sample) = &input.target.weight_sample {
            let bytes = planner_bundle
                .input(&sample.digest)
                .map_err(InventoryError::Integrity)?;
            let sample = WeightSample::decode(&bytes).map_err(|error| {
                InventoryError::Integrity(format!("invalid weight sample {}: {error}", sample.digest))
            })?;
            weight_samples.insert(bundle_key.clone(), sample);
        }
    }
    Ok(ReleaseCatalog {
        catalog,
        planner_inputs: Arc::new(manifest.planner_inputs),
        planner_bundle: Arc::new(planner_bundle),
        weight_samples: Arc::new(weight_samples),
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
        .map(|model| (model.model_id.clone(), model.variant_id.clone()))
        .collect::<BTreeSet<_>>();
    let target_packages = catalog
        .models
        .iter()
        .map(|model| match &model.configuration.bundle {
            ServableModelBundle::Standalone { package } => package,
            ServableModelBundle::SpeculativeDecoding { target, .. } => target,
        })
        .collect::<Vec<_>>();
    let target_package_ids = target_packages
        .iter()
        .map(|package| package.id.clone())
        .collect::<BTreeSet<_>>();
    let complete_intrinsic_targets = target_packages
        .iter()
        .filter(|package| {
            package.properties.intrinsic_model_id.is_some()
                && package.properties.intrinsic_quality_id.is_some()
        })
        .count();
    if catalog.models.is_empty()
        || model_ids.len() != catalog.models.len()
        || target_package_ids.len() != catalog.models.len()
        || catalog.models.iter().any(|model| {
            // An embedded MTP head cannot enter the projector's media rows.
            let (target, embedded_mtp) = match &model.configuration.bundle {
                ServableModelBundle::Standalone { package } => (package, false),
                ServableModelBundle::SpeculativeDecoding {
                    target,
                    method,
                    draft_source,
                } => (
                    target,
                    *method == SpeculativeMethod::Mtp
                        && matches!(draft_source, SpeculativeDraftSource::Embedded),
                ),
            };
            let projector_count = target
                .files
                .iter()
                .filter(|file| file.role == ModelFileRole::Projector)
                .count();
            model.configuration.profile.context_length < MIN_CATALOG_CONTEXT_LENGTH
                || target.properties.maximum_context_length
                    != Some(model.configuration.profile.context_length)
                || projector_count > 1
                || (projector_count == 1 && embedded_mtp)
                || servable_bundle_packages(&model.configuration.bundle).any(|package| {
                    !matches!(
                        &package.source,
                        ModelPackageSource::HuggingFace { revision, .. }
                            if valid_commit(revision)
                    )
                })
        })
    {
        return Err(InventoryError::Integrity(format!(
            "release catalog has missing or duplicate exact identities (models={}, model_ids={}, target_packages={}, complete_intrinsic_targets={})",
            catalog.models.len(),
            model_ids.len(),
            target_package_ids.len(),
            complete_intrinsic_targets,
        )));
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
            component.header_digest.len() == 64
                && component.header_size_bytes > 0
                && valid_hex_digest(&component.header_digest)
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
        planner_bundle::encode(&manifest, &self.headers, progress)
            .map_err(InventoryError::Integrity)
    }
}

impl ReleaseCatalog {
    /// Validate all stored chunks when publishing a release artifact. Runtime assessment verifies
    /// each exact input digest when that header is read instead of scanning unrelated headers.
    pub fn verify_all_headers(&self) -> Result<(), InventoryError> {
        self.planner_bundle
            .verify()
            .map_err(InventoryError::Integrity)
    }

    #[must_use]
    pub fn catalog(&self) -> &RecommendableModelCatalog {
        &self.catalog
    }

    /// The header digest of every bundled package file, by the file's Hub SHA-256.
    #[must_use]
    pub fn file_header_digests(&self) -> BTreeMap<String, String> {
        self.planner_inputs
            .values()
            .flat_map(|input| input.packages())
            .flat_map(|package| {
                package.components.iter().filter_map(|component| {
                    package
                        .package
                        .files
                        .iter()
                        .find(|file| file.path == component.component.path)
                        .map(|file| (file.sha256.clone(), component.header_digest.clone()))
                })
            })
            .collect()
    }

    /// Write every catalog component's exact header, extended sparsely to its file's size, under
    /// `root` at `<repository>/<revision>/<path>`, and return them
    /// (`magnitude_artifacts::catalog_headers`): each variant's target, draft and projector.
    pub fn export_headers(&self, root: &Path) -> Result<Vec<CatalogComponent>, InventoryError> {
        let mut components = Vec::new();
        let mut written = BTreeSet::new();
        for input in self.planner_inputs.values() {
            let model = ModelId::catalog(&input.model_id, &input.variant_id);
            let target = (
                HeaderRole::Target,
                &input.target,
                input.target.primary_gguf.clone(),
                &[ComponentRole::Weights, ComponentRole::Shard] as &[ComponentRole],
            );
            let projector = input
                .target
                .components
                .iter()
                .find(|component| component.component.role == ComponentRole::Projector)
                .map(|projector| {
                    (
                        HeaderRole::Projector,
                        &input.target,
                        projector.component.path.clone(),
                        &[ComponentRole::Projector] as &[ComponentRole],
                    )
                });
            let draft = input.draft.as_ref().map(|draft| {
                (
                    HeaderRole::Draft,
                    draft,
                    draft.primary_gguf.clone(),
                    &[
                        ComponentRole::Weights,
                        ComponentRole::Shard,
                        ComponentRole::Draft,
                        ComponentRole::Mtp,
                    ] as &[ComponentRole],
                )
            });
            for (role, package, path, roles) in
                std::iter::once(target).chain(projector).chain(draft)
            {
                let ModelPackageSource::HuggingFace {
                    repository,
                    revision,
                } = &package.package.source
                else {
                    return Err(InventoryError::Integrity(
                        "release catalog package must have an immutable Hugging Face source"
                            .to_owned(),
                    ));
                };
                let component = CatalogComponent {
                    model: model.as_str().to_owned(),
                    role,
                    repository: repository.clone(),
                    revision: revision.clone(),
                    path,
                };
                for file in package
                    .components
                    .iter()
                    .filter(|file| roles.contains(&file.component.role))
                {
                    let path = root
                        .join(repository)
                        .join(revision)
                        .join(&file.component.path);
                    if written.insert(path.clone()) {
                        let header = self
                            .planner_bundle
                            .input(&file.header_digest)
                            .map_err(InventoryError::Integrity)?;
                        write_component_header(file, &header, &path)?;
                    }
                }
                components.push(component);
            }
        }
        components.sort_by(|left, right| (&left.model, left.role).cmp(&(&right.model, right.role)));
        Ok(components)
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
            artifact,
            bundle,
            |component| {
                self.planner_bundle
                    .input(&component.header_digest)
                    .map_err(InventoryError::Integrity)
            },
        )?))
    }
}

impl ReleaseCatalog {
    /// A release catalog of `catalog`'s declarations without release assessment material: every
    /// bundle resolves from installed files or its pinned source instead. For tests and tools
    /// that carry no planner bundle.
    pub fn without_material(catalog: RecommendableModelCatalog) -> Result<Self, InventoryError> {
        let manifest = serde_json::to_vec(&ReleasePlannerManifest {
            planner_inputs: BTreeMap::new(),
        })
        .map_err(|error| InventoryError::Internal(error.to_string()))?;
        let bytes = planner_bundle::encode(&manifest, &BTreeMap::new(), |_, _| {})
            .map_err(InventoryError::Integrity)?;
        let bytes: &'static [u8] = Box::leak(bytes.into_boxed_slice());
        Ok(Self {
            catalog,
            planner_inputs: Arc::new(BTreeMap::new()),
            planner_bundle: Arc::new(PlannerBundle::parse(bytes).map_err(InventoryError::Integrity)?),
            weight_samples: Arc::new(BTreeMap::new()),
        })
    }

    /// This catalog with `samples` as its targets' weight samples, by bundle, as a release bundle
    /// carries them. For tests and tools that carry no planner bundle.
    #[must_use]
    pub fn with_target_weight_samples(
        mut self,
        samples: BTreeMap<ServableModelBundleKey, WeightSample>,
    ) -> Self {
        self.weight_samples = Arc::new(samples);
        self
    }

    /// The weight sample of a catalog bundle's target, when the release bundle carries one.
    #[must_use]
    pub fn target_weight_sample(&self, bundle_key: &ServableModelBundleKey) -> Option<&WeightSample> {
        self.weight_samples.get(bundle_key)
    }
}

/// One component's exact header at `path`, extended sparsely to the component's logical size so
/// no tensor payload occupies disk.
fn write_component_header(
    component: &ReleasePlannerComponent,
    header: &[u8],
    path: &Path,
) -> Result<(), InventoryError> {
    if u64::try_from(header.len()).ok() != Some(component.header_size_bytes) {
        return Err(InventoryError::Integrity(format!(
            "planner input for {} has the wrong length",
            component.component.path.display()
        )));
    }
    write_header_file(header, component.component.size_bytes, path)
}

/// A GGUF file's exact `header` at `path`, extended sparsely to the file's `size` so no tensor
/// payload occupies disk: it opens as the file it was read from.
pub(crate) fn write_header_file(header: &[u8], size: u64, path: &Path) -> Result<(), InventoryError> {
    let io = |error: std::io::Error| InventoryError::Io(error.to_string());
    let parent = path
        .parent()
        .ok_or_else(|| InventoryError::Integrity("a header file has no parent".to_owned()))?;
    fs::create_dir_all(parent).map_err(io)?;
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .map_err(io)?;
    file.write_all(header)
        .and_then(|()| magnitude_service_utils::sparse_file::set_sparse_len(&file, size))
        .map_err(io)
}

/// Materialize the bundle's assessment material: each component's exact header, extended
/// sparsely to the component's logical size so no tensor payload occupies disk.
fn materialize_planner_bundle(
    artifact: &ReleasePlannerInput,
    bundle: ServableModelBundle,
    mut header_for: impl FnMut(&ReleasePlannerComponent) -> Result<Vec<u8>, InventoryError>,
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
        materialize_planner_package(&artifact.target, target, &mut header_for)?;
    let (draft_model, draft_workspace) = match (draft, artifact.draft.as_ref()) {
        (Some(package), Some(planner)) => {
            let (model, workspace) =
                materialize_planner_package(planner, package, &mut header_for)?;
            (Some(model), Some(workspace))
        }
        (None, None) => (None, None),
        _ => unreachable!("draft presence was validated"),
    };
    let mut resolved = ResolvedServableModelBundle::new(bundle, target_model, draft_model)
        .retain_resolution_guard(target_workspace);
    if let Some(workspace) = draft_workspace {
        resolved = resolved.retain_resolution_guard(workspace);
    }
    Ok(resolved)
}

fn materialize_planner_package(
    artifact: &ReleasePlannerPackage,
    package: &ModelPackage,
    header_for: &mut impl FnMut(&ReleasePlannerComponent) -> Result<Vec<u8>, InventoryError>,
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
        write_component_header(
            component,
            &header_for(component)?,
            &workspace.path().join(&component.component.path),
        )?;
    }
    let components = artifact
        .components
        .iter()
        .map(|component| component.component.clone())
        .collect::<Vec<_>>();
    let model = InventoryModel {
        id: InventoryEntryId(package_identity.clone()),
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
                method: "release_catalog_header_digest".to_owned(),
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
        .map(|model| (model.model_id.as_str(), model.variant_id.as_str()))
        .collect::<BTreeSet<_>>();
    let expected = source
        .models
        .iter()
        .flat_map(|model| {
            model
                .variants()
                .map(|(_, variant)| (model.id.as_str(), variant.variant_id.as_str()))
        })
        .collect::<BTreeSet<_>>();
    if actual.len() != catalog.models.len()
        || actual.len() != expected.len()
        || !expected.iter().all(|identity| actual.contains(identity))
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
            .find(|declaration| declaration.id == model.model_id.as_str())
            .ok_or_else(|| {
                InventoryError::Integrity(format!(
                    "release catalog model {} variant {} has no source declaration",
                    model.model_id, model.variant_id
                ))
            })?;
        let declared_source = declaration
            .sources
            .iter()
            .find(|source| {
                source
                    .variants
                    .iter()
                    .any(|variant| variant.variant_id == model.variant_id.as_str())
            })
            .ok_or_else(|| {
                InventoryError::Integrity(format!(
                    "release catalog model {} variant {} has no source declaration",
                    model.model_id, model.variant_id
                ))
            })?;
        let locked = |repository: &str| {
            lock.get(repository).ok_or_else(|| {
                InventoryError::Integrity(format!("model catalog lock is missing {repository}"))
            })
        };
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
            ) => method == &(*expected_method).into(),
            (
                Some(CatalogSpeculativeDecoding {
                    method: expected_method,
                    draft:
                        CatalogSpeculativeDraftSource::File {
                            repository: expected_repository,
                            path: expected_path,
                        },
                }),
                Some(SpeculativeDraftSource::Separate { draft }),
                Some(method),
            ) => {
                method == &(*expected_method).into()
                    && package_source_matches(
                        &draft.source,
                        expected_repository,
                        locked(expected_repository)?,
                    )
                    && draft.files.iter().any(|file| file.path == *expected_path)
            }
            _ => false,
        };
        let projector_files = target
            .files
            .iter()
            .filter(|file| file.role == ModelFileRole::Projector)
            .collect::<Vec<_>>();
        // As `resolve_projector_path` selects: the declared projector, or
        // without a declaration the repository's sole mmproj, if any.
        let projector_matches = match &declared_source.projector {
            Some(projector) => {
                projector_files.len() == 1 && projector_files[0].path == projector.path
            }
            None => projector_files.len() <= 1,
        };
        if target.properties.maximum_context_length
            != Some(model.configuration.profile.context_length)
            || !package_source_matches(
                &target.source,
                &declared_source.repository,
                locked(&declared_source.repository)?,
            )
            || !projector_matches
            || !speculative_matches
        {
            return Err(InventoryError::Integrity(format!(
                "release catalog model {} variant {} does not match models.json and models.lock.json",
                model.model_id, model.variant_id
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
    catalog: &CatalogSource,
    declaration: &CatalogModel,
    source: &CatalogModelSource,
    variant: &CatalogVariant,
    target: ModelPackage,
    draft: Option<ModelPackage>,
    properties: &InventoryProperties,
) -> Result<RecommendableModel, InventoryError> {
    // The served context is the target's supported maximum, which is the context the engine
    // resolves and assesses; a separate draft never bounds it.
    let profile = ServingProfile {
        context_length: target.properties.maximum_context_length.ok_or_else(|| {
            InventoryError::Integrity(format!(
                "{} target declares no maximum context length",
                declaration.id
            ))
        })?,
    };
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
    Ok(RecommendableModel {
        model_id: CatalogBaseId::new(declaration.id.clone())
            .map_err(|error| InventoryError::Integrity(error.to_string()))?,
        variant_id: CatalogVariantId::new(variant.variant_id.clone())
            .map_err(|error| InventoryError::Integrity(error.to_string()))?,
        configuration: ModelServingConfiguration { bundle, profile },
        display_name: declaration.display_name.clone(),
        variant_label: variant.variant_label.clone(),
        description: declaration.description.clone(),
        release_date: declaration.release_date.clone(),
        license: source.license.clone(),
        parameterization: declaration.parameterization.clone(),
        developer: catalog.developer(declaration),
        support: declaration.rated_support(),
        quant: CatalogQuant {
            repository: source.repository.clone(),
            name: variant.format.clone(),
        },
        quantization_aware: variant.quantization_aware,
    })
}

fn catalog_from_planner_inputs(
    source: &CatalogSource,
    inputs: &BTreeMap<ServableModelBundleKey, ReleasePlannerInput>,
) -> Result<RecommendableModelCatalog, InventoryError> {
    let by_identity = inputs
        .values()
        .map(|input| ((input.model_id.clone(), input.variant_id.clone()), input))
        .collect::<BTreeMap<_, _>>();
    if by_identity.len() != inputs.len() {
        return Err(InventoryError::Integrity(
            "planner bundle contains duplicate catalog model identities".to_owned(),
        ));
    }
    let mut models = Vec::new();
    for declaration in &source.models {
        for (declared_source, variant) in declaration.variants() {
            let identity = (
                CatalogBaseId::new(declaration.id.clone())
                    .map_err(|error| InventoryError::Integrity(error.to_string()))?,
                CatalogVariantId::new(variant.variant_id.clone())
                    .map_err(|error| InventoryError::Integrity(error.to_string()))?,
            );
            let input = by_identity.get(&identity).ok_or_else(|| {
                InventoryError::Integrity(format!(
                    "planner bundle is missing catalog model {} variant {}",
                    declaration.id, variant.variant_id
                ))
            })?;
            let model = recommendable_model(
                source,
                declaration,
                declared_source,
                variant,
                input.target.package.clone(),
                input.draft.as_ref().map(|draft| draft.package.clone()),
                &input.target.properties,
            )?;
            if !inputs.contains_key(&recommendable_model_bundle_key(&model)) {
                return Err(InventoryError::Integrity(format!(
                    "planner bundle key does not match catalog model {} variant {}",
                    declaration.id, variant.variant_id
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
    models: Arc<ManagedModelStore>,
}

impl ResolvingRecommendableCatalog {
    #[must_use]
    pub fn new(models: Arc<ManagedModelStore>) -> Self {
        Self { models }
    }

    async fn resolve_model(
        &self,
        catalog: &CatalogSource,
        declaration: &CatalogModel,
        source: &CatalogModelSource,
        variant: &CatalogVariant,
        snapshots: &BTreeMap<String, HuggingFaceRepositorySnapshot>,
    ) -> Result<
        (
            RecommendableModel,
            ReleasePlannerInput,
            BTreeMap<String, Vec<u8>>,
        ),
        InventoryError,
    > {
        let snapshot = snapshots.get(&source.repository).ok_or_else(|| {
            InventoryError::Integrity(format!(
                "target repository {} was not resolved",
                source.repository
            ))
        })?;
        let primary = match &variant.path {
            Some(path) => declared_primary_gguf(snapshot, path)?,
            None => select_primary_gguf(snapshot, &variant.format)?,
        };
        let projector = resolve_projector_path(declaration, source, snapshot)?;
        let additional_components = projector
            .map(|path| ModelPreviewComponentSource {
                path,
                role: ModelPreviewComponentRole::Projector,
            })
            .into_iter()
            .collect();
        let target_shards = self
            .models
            .resolve_snapshot_shards(snapshot, &primary.path)
            .await?;
        let target_prepared = self
            .models
            .prepare_preview_from_repository_snapshot(
                &ModelPreviewSource {
                    repository: snapshot.repository.clone(),
                    revision: snapshot.commit.clone(),
                    shards: target_shards.clone(),
                    additional_components,
                },
                snapshot,
            )
            .await?;
        let (mut target, mut headers) =
            self.release_planner_package(target_prepared, &primary.path)?;
        if let Some(sample) = self
            .models
            .hub_weight_sample_shards(snapshot, &target_shards)
            .await?
        {
            let bytes = sample.encode();
            let digest = planner_bundle::sha256(&bytes);
            target.weight_sample = Some(ReleasePlannerWeightSample {
                digest: digest.clone(),
                size_bytes: u64::try_from(bytes.len()).map_err(|_| {
                    InventoryError::Integrity("weight sample is too large".to_owned())
                })?,
            });
            merge_content_map(&mut headers, BTreeMap::from([(digest, bytes)]))?;
        }
        let draft = match declaration
            .speculative_decoding
            .as_ref()
            .and_then(|speculative| speculative.draft.file())
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
                let draft_shards = self
                    .models
                    .resolve_snapshot_shards(snapshot, draft_path)
                    .await?;
                let prepared = self
                    .models
                    .prepare_preview_from_repository_snapshot(
                        &ModelPreviewSource {
                            repository: snapshot.repository.clone(),
                            revision: snapshot.commit.clone(),
                            shards: draft_shards,
                            additional_components: Vec::new(),
                        },
                        snapshot,
                    )
                    .await?;
                let (draft, draft_headers) = self.release_planner_package(prepared, draft_path)?;
                merge_content_map(&mut headers, draft_headers)?;
                Some(draft)
            }
            None => None,
        };
        let model = recommendable_model(
            catalog,
            declaration,
            source,
            variant,
            target.package.clone(),
            draft.as_ref().map(|draft| draft.package.clone()),
            &target.properties,
        )?;
        let planner = ReleasePlannerInput {
            model_id: model.model_id.clone(),
            variant_id: model.variant_id.clone(),
            target,
            draft,
        };
        Ok((model, planner, headers))
    }

    fn release_planner_package(
        &self,
        prepared: PreparedPreview,
        primary: &Path,
    ) -> Result<(ReleasePlannerPackage, BTreeMap<String, Vec<u8>>), InventoryError> {
        let inspected = validated_package_from_resolved(&prepared.model)?;
        let headers = prepared
            .headers
            .iter()
            .map(|header| {
                self.models
                    .cache
                    .read_header(&header.digest)
                    .map(|bytes| (header.digest.clone(), bytes))
                    .ok_or_else(|| {
                        InventoryError::Integrity(format!(
                            "bundle construction lost source header {}",
                            header.digest
                        ))
                    })
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        if !prepared.headers.iter().any(|header| header.path == primary) {
            return Err(InventoryError::Integrity(
                "catalog primary has no planner header".to_owned(),
            ));
        }
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
                Ok(ReleasePlannerComponent {
                    component: component.clone(),
                    header_digest: header.digest.clone(),
                    header_size_bytes: u64::try_from(source.len()).map_err(|_| {
                        InventoryError::Integrity("planner header is too large".to_owned())
                    })?,
                })
            })
            .collect::<Result<Vec<_>, InventoryError>>()?;
        let planner = ReleasePlannerPackage {
            package: inspected.package,
            validation: inspected.validation,
            properties: prepared.model.model.properties.clone(),
            primary_gguf: primary.to_path_buf(),
            components,
            weight_sample: None,
        };
        Ok((planner, headers))
    }
}

/// The variant's primary GGUF: the one non-projector, non-imatrix file (or first shard) whose
/// repository path contains the variant's upstream format selector.
fn select_primary_gguf<'a>(
    snapshot: &'a HuggingFaceRepositorySnapshot,
    format: &str,
) -> Result<&'a HuggingFaceRepositoryFile, InventoryError> {
    let selector = format.to_ascii_lowercase();
    let matches = snapshot
        .gguf_files
        .iter()
        .filter(|file| {
            let path = file.path.to_string_lossy().to_ascii_lowercase();
            let basename = path.rsplit('/').next().unwrap_or(path.as_str());
            path.contains(&selector)
                && !is_projector_path(&file.path)
                && !basename.contains("imatrix")
                && (!is_later_shard(basename) || is_first_shard(basename))
        })
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [primary] => Ok(primary),
        _ => Err(InventoryError::Integrity(format!(
            "{} format {} resolved to {} primary files",
            snapshot.repository,
            format,
            matches.len()
        ))),
    }
}

/// A declared primary GGUF: an exact repository file that is unsharded or a first shard.
fn declared_primary_gguf<'a>(
    snapshot: &'a HuggingFaceRepositorySnapshot,
    path: &Path,
) -> Result<&'a HuggingFaceRepositoryFile, InventoryError> {
    let basename = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default();
    snapshot
        .gguf_files
        .iter()
        .find(|file| file.path == path)
        .filter(|_| !is_projector_path(path) && !is_later_shard(basename))
        .ok_or_else(|| {
            InventoryError::Integrity(format!(
                "{} declared primary {} is not a repository GGUF or first shard",
                snapshot.repository,
                path.display()
            ))
        })
}

fn merge_content_map(
    target: &mut BTreeMap<String, Vec<u8>>,
    source: BTreeMap<String, Vec<u8>>,
) -> Result<(), InventoryError> {
    for (digest, content) in source {
        if let Some(previous) = target.insert(digest.clone(), content.clone())
            && previous != content
        {
            return Err(InventoryError::Integrity(format!(
                "planner header digest collision {digest}"
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

/// One catalog variant addressed by its catalog `ModelId`, with the locked repository revisions
/// its target and separately packaged draft resolve from.
struct LockedCatalogVariant {
    declaration: CatalogModel,
    source: CatalogModelSource,
    variant: CatalogVariant,
    repositories: BTreeMap<String, String>,
}

fn locked_catalog_variant(
    source: &CatalogSource,
    lock: &ModelCatalogLock,
    id: &ModelId,
) -> Result<LockedCatalogVariant, InventoryError> {
    let ParsedModelId::Catalog {
        base_id,
        variant_id,
    } = id.parsed()
    else {
        return Err(InventoryError::InvalidRequest(format!(
            "{id} is not a catalog model ID"
        )));
    };
    let declaration = source
        .models
        .iter()
        .find(|model| model.id == base_id.as_str())
        .ok_or_else(|| InventoryError::NotFound(format!("catalog model {base_id}")))?;
    let (source, variant) = declaration
        .variants()
        .find(|(_, variant)| variant.variant_id == variant_id.as_str())
        .ok_or_else(|| {
            InventoryError::NotFound(format!("catalog model {base_id} variant {variant_id}"))
        })?;
    let repositories = std::iter::once(source.repository.as_str())
        .chain(declaration.draft_repository())
        .map(|repository| {
            lock.get(repository)
                .map(|commit| (repository.to_owned(), commit.clone()))
                .ok_or_else(|| {
                    InventoryError::Integrity(format!(
                        "model catalog lock is missing {repository}"
                    ))
                })
        })
        .collect::<Result<_, _>>()?;
    Ok(LockedCatalogVariant {
        declaration: declaration.clone(),
        source: source.clone(),
        variant: variant.clone(),
        repositories,
    })
}

impl ResolvingRecommendableCatalog {
    /// Resolve one catalog variant at its locked revisions into its exact servable bundle:
    /// the target package (with its projector), and the separately packaged draft if declared.
    pub async fn resolve_catalog_model(
        &self,
        id: &ModelId,
    ) -> Result<RecommendableModel, InventoryError> {
        let source = catalog_source()?;
        let lock = model_catalog_lock()?;
        let selected = locked_catalog_variant(&source, &lock, id)?;
        let mut snapshots = BTreeMap::new();
        for (repository, revision) in selected.repositories {
            let snapshot = refresh_hugging_face_repository(
                &self.models,
                HuggingFaceRepositoryRequest {
                    repository: repository.clone(),
                    revision,
                },
            )
            .await?;
            snapshots.insert(repository, snapshot);
        }
        let (model, _, _) = self
            .resolve_model(
                &source,
                &selected.declaration,
                &selected.source,
                &selected.variant,
                &snapshots,
            )
            .await?;
        Ok(model)
    }

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
        // The lock covers exactly the catalog's repositories (`validate_model_catalog_lock`).
        let repositories = lock.into_iter().collect::<Vec<_>>();
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
        let catalog = &source;
        let model_total = source.models.len();
        let mut model_completed = 0;
        let resolved =
            stream::iter(source.models.iter().cloned().enumerate())
                .map(|(declaration_index, declaration)| async move {
                    let mut variants = Vec::new();
                    for (variant_index, (source, variant)) in declaration.variants().enumerate() {
                        let missing_repository = std::iter::once(source.repository.as_str())
                            .chain(declaration.draft_repository())
                            .find(|repository| !resolved_snapshots.contains_key(*repository));
                        let result =
                            match missing_repository {
                                None => {
                                    self.resolve_model(
                                        catalog,
                                        &declaration,
                                        source,
                                        variant,
                                        resolved_snapshots,
                                    )
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
        let mut headers = BTreeMap::new();
        let mut diagnostics = Vec::new();
        for (_, _, declaration, variant, result) in resolved {
            match result {
                Ok((model, planner, model_headers)) => {
                    planner_inputs.insert(recommendable_model_bundle_key(&model), planner);
                    models.push(model);
                    merge_content_map(&mut headers, model_headers)?;
                }
                Err(error) => diagnostics.push(CatalogDiagnostic {
                    model_id: CatalogBaseId::new(declaration.id.clone())
                        .map_err(|error| InventoryError::Integrity(error.to_string()))?,
                    variant_id: CatalogVariantId::new(variant.variant_id.clone())
                        .map_err(|error| InventoryError::Integrity(error.to_string()))?,
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
            headers,
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

    fn repository_snapshot(paths: &[&str]) -> HuggingFaceRepositorySnapshot {
        HuggingFaceRepositorySnapshot {
            repository: "publisher/model".to_owned(),
            commit: "0123456789abcdef0123456789abcdef01234567".to_owned(),
            last_modified: None,
            downloads: None,
            likes: None,
            gated: false,
            private: false,
            license: None,
            license_url: None,
            base_models: Vec::new(),
            tags: Vec::new(),
            gguf_files: paths
                .iter()
                .map(
                    |path| magnitude_service_contracts::HuggingFaceRepositoryFile {
                        path: PathBuf::from(path),
                        size_bytes: 1,
                        content: magnitude_service_contracts::ContentIdentity::Unknown,
                    },
                )
                .collect(),
        }
    }

    #[test]
    fn every_registered_target_architecture_belongs_to_exactly_one_family() {
        // Llama is registered only as a drafter architecture, which no family serves as a target.
        const DRAFTER_ONLY: &[&str] = &["llama"];
        let source = catalog_source().expect("catalog source should be valid");
        let registered = magnitude_family_registry::FAMILIES
            .iter()
            .flat_map(|family| family.architectures().iter().copied())
            .collect::<BTreeSet<_>>();
        for architecture in registered.iter().filter(|architecture| !DRAFTER_ONLY.contains(architecture)) {
            let families = source
                .families
                .iter()
                .filter(|family| family.architectures.iter().any(|declared| declared == architecture))
                .count();
            assert_eq!(families, 1, "{architecture} belongs to {families} catalog families");
        }
        for family in &source.families {
            for architecture in &family.architectures {
                assert!(
                    registered.contains(architecture.as_str())
                        && !DRAFTER_ONLY.contains(&architecture.as_str()),
                    "{} declares {architecture}, which is not a registered target architecture",
                    family.name
                );
            }
        }
    }

    #[test]
    fn authored_catalog_declares_valid_parameterization_for_every_model() {
        let source = catalog_source().expect("catalog source should be valid");
        assert!(
            source
                .models
                .iter()
                .all(|model| valid_parameterization(&model.parameterization))
        );
    }

    #[test]
    fn projector_resolution_is_automatic_only_when_unambiguous() {
        let source = catalog_source().expect("catalog source should be valid");
        let mut declaration = source.models[0].clone();
        declaration.speculative_decoding = None;
        let mut model_source = declaration.sources[0].clone();
        model_source.projector = None;
        assert_eq!(
            resolve_projector_path(
                &declaration,
                &model_source,
                &repository_snapshot(&["model.gguf", "mmproj-model-F16.gguf"]),
            )
            .unwrap(),
            Some(PathBuf::from("mmproj-model-F16.gguf"))
        );
        let error = resolve_projector_path(
            &declaration,
            &model_source,
            &repository_snapshot(&[
                "model.gguf",
                "mmproj-model-BF16.gguf",
                "mmproj-model-F16.gguf",
            ]),
        )
        .unwrap_err();
        assert!(error.to_string().contains("2 projector candidates"));
    }

    #[test]
    fn projector_resolution_rejects_invalid_paths_and_embedded_mtp() {
        let source = catalog_source().expect("catalog source should be valid");
        let mut declaration = source.models[0].clone();
        let mut model_source = declaration.sources[0].clone();
        let snapshot = repository_snapshot(&["model.gguf", "mmproj-BF16.gguf"]);
        model_source.projector = Some(CatalogProjectorSource {
            path: PathBuf::from("mmproj-missing.gguf"),
        });
        let error = resolve_projector_path(&declaration, &model_source, &snapshot).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("is not a repository mmproj GGUF")
        );

        model_source.projector = Some(CatalogProjectorSource {
            path: PathBuf::from("mmproj-BF16.gguf"),
        });
        declaration.speculative_decoding = Some(CatalogSpeculativeDecoding {
            method: CatalogSpeculativeMethod::Mtp,
            draft: CatalogSpeculativeDraftSource::Embedded,
        });
        let error = resolve_projector_path(&declaration, &model_source, &snapshot).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("combines a projector with embedded MTP")
        );

        // A separately packaged assistant reads the target's history.
        declaration.speculative_decoding = Some(CatalogSpeculativeDecoding {
            method: CatalogSpeculativeMethod::Mtp,
            draft: CatalogSpeculativeDraftSource::File {
                repository: "publisher/model-MTP".to_owned(),
                path: PathBuf::from("mtp-gemma-4-12B-it.gguf"),
            },
        });
        assert_eq!(
            resolve_projector_path(&declaration, &model_source, &snapshot).unwrap(),
            Some(PathBuf::from("mmproj-BF16.gguf"))
        );
    }

    #[test]
    fn deprecation_names_a_supported_exact_replacement() {
        let source = catalog_source().expect("catalog source should be valid");
        let mut model = source
            .models
            .iter()
            .find(|model| model.support.is_deprecated())
            .expect("the authored catalog deprecates a model")
            .clone();
        assert!(valid_catalog_support(&model, &source));
        let deprecated = |since: &str, replacement: &str, reason: &str| {
            DeclaredSupport::Deprecated {
                since: since.to_owned(),
                replacement: replacement.parse().expect("model ID"),
                reason: reason.to_owned(),
            }
        };
        for (support, valid) in [
            (deprecated("2026-09-27", "qwen3.5-4b:gguf:q4", "x"), true),
            (deprecated("2026-02-30", "qwen3.5-4b:gguf:q4", "unsupported"), false),
            (deprecated("2026-09-27", "qwen3.5-4b:gguf:q4", ""), false),
            (deprecated("2026-09-27", "qwen3.5-4b:gguf:q3", "x"), false),
            (deprecated("2026-09-27", "missing:gguf:q4", "x"), false),
            (deprecated("2026-09-27", &format!("{}:gguf:q1-qat", model.id), "x"), false),
            (
                deprecated("2026-09-27", "hf:owner/repo/model.gguf", "x"),
                false,
            ),
            (
                DeclaredSupport::Disabled {
                    reason: String::new(),
                },
                false,
            ),
        ] {
            model.support = support;
            assert_eq!(valid_catalog_support(&model, &source), valid);
        }
    }

    #[test]
    fn authored_lock_covers_every_model_including_deprecated_ones() {
        let lock = model_catalog_lock().expect("authored lock should be valid");
        let source = catalog_source().expect("catalog source should be valid");
        assert!(
            source
                .models
                .iter()
                .filter(|model| model.support.is_deprecated())
                .flat_map(CatalogModel::repositories)
                .all(|repository| lock.contains_key(repository))
        );
    }

    #[test]
    fn a_replacement_is_never_itself_deprecated() {
        let mut source = catalog_source().expect("catalog source should be valid");
        let replacement = source
            .models
            .iter_mut()
            .find(|model| model.id == "qwen3.5-4b")
            .expect("replacement model");
        replacement.support = DeclaredSupport::Deprecated {
            since: "2026-09-27".to_owned(),
            replacement: "qwen3.5-4b:gguf:q4".parse().expect("model ID"),
            reason: "superseded".to_owned(),
        };
        let deprecated = source
            .models
            .iter()
            .find(|model| model.id == "bonsai-8b-q1")
            .expect("deprecated model");
        assert!(!valid_catalog_support(deprecated, &source));
    }

    #[test]
    fn labs_have_distinct_ids_and_organizations_and_a_country_code() {
        let source = catalog_source().expect("catalog source should be valid");
        assert!(valid_labs(&source.labs));
        assert!(source
            .models
            .iter()
            .all(|model| source.labs.iter().any(|lab| lab.id == model.lab)));
        let mut shared = source.labs.clone();
        shared[1].hugging_face_organizations.push("QWEN".to_owned());
        assert!(!valid_labs(&shared));
        let mut duplicate = source.labs.clone();
        duplicate[1].id = duplicate[0].id.clone();
        assert!(!valid_labs(&duplicate));
        let mut country = source.labs.clone();
        country[0].country = "China".to_owned();
        assert!(!valid_labs(&country));
    }

    #[test]
    fn expected_benchmark_score_follows_each_benchmarks_difficulty_and_slope() {
        let even = [BenchmarkDifficulty {
            difficulty: 150.0,
            slope: 0.1,
        }];
        assert!((expected_benchmark_score(150.0, &even) - 0.5).abs() < 1e-12);
        let one_logit_up = expected_benchmark_score(160.0, &even);
        assert!((one_logit_up - 1.0 / (1.0 + (-1.0f64).exp())).abs() < 1e-12);
        let mixed = [
            BenchmarkDifficulty {
                difficulty: 140.0,
                slope: 0.2,
            },
            BenchmarkDifficulty {
                difficulty: 170.0,
                slope: 0.2,
            },
        ];
        let low = expected_benchmark_score(140.0, &mixed);
        let high = expected_benchmark_score(170.0, &mixed);
        assert!(low < high);
        assert!((low - (0.5 + 1.0 / (1.0 + 6.0f64.exp())) / 2.0).abs() < 1e-12);
    }

    #[test]
    fn intelligence_is_checked_against_the_frontiers_benchmarks() {
        let source = catalog_source().expect("catalog source should be valid");
        let since = &source.intelligence_frontier.benchmarks_since;
        let csv = format!(
            "benchmark_name,is_anchor,benchmark_release_date,edi,estimated_slope_scaled\n\
             Old,False,2016-06-20,40.0,0.02\n\
             Current,False,{since},150.0,0.1\n"
        );
        let checks = catalog_intelligence_check(&csv).expect("the difficulties parse");
        assert_eq!(
            checks.len(),
            source.models.iter().filter(|model| model.intelligence.is_some()).count()
        );
        let frontier = source.intelligence_frontier.eci;
        let current = [BenchmarkDifficulty {
            difficulty: 150.0,
            slope: 0.1,
        }];
        for check in &checks {
            let expected = (100.0 * expected_benchmark_score(check.eci, &current)
                / expected_benchmark_score(frontier, &current))
            .round() as u32;
            assert_eq!(check.computed, expected, "{}", check.model);
        }
        assert!(catalog_intelligence_check("name,edi\n").is_err());
        assert!(catalog_intelligence_check(
            "benchmark_name,is_anchor,benchmark_release_date,edi,estimated_slope_scaled\n\
             Old,False,2016-06-20,40.0,0.02\n"
        )
        .is_err());
    }

    #[test]
    fn intelligence_never_exceeds_the_frontier_and_names_its_evidence() {
        let source = catalog_source().expect("catalog source should be valid");
        let frontier = &source.intelligence_frontier;
        assert!(valid_intelligence_frontier(frontier));
        for model in &source.models {
            assert!(valid_model_score(model, frontier));
        }
        let scored = |eci: f64, percent: u32, provenance: IntelligenceProvenance| IntelligenceScore {
            eci,
            percent,
            provenance,
        };
        let published = || IntelligenceProvenance::EpochPublished {
            url: "https://epoch.ai/benchmarks/eci".to_owned(),
        };
        assert!(valid_intelligence(&scored(150.0, 48, published()), frontier));
        assert!(!valid_intelligence(&scored(frontier.eci + 0.1, 48, published()), frontier));
        assert!(!valid_intelligence(&scored(150.0, 101, published()), frontier));
        assert!(!valid_intelligence(&scored(f64::NAN, 48, published()), frontier));
        assert!(!valid_intelligence(
            &scored(
                150.0,
                48,
                IntelligenceProvenance::EpochData {
                    benchmarks: Vec::new(),
                    url: "https://epoch.ai/benchmarks/eci".to_owned(),
                }
            ),
            frontier
        ));
        assert!(!valid_intelligence(
            &scored(
                150.0,
                48,
                IntelligenceProvenance::Bridged {
                    method: "fitted".to_owned(),
                    evidence_urls: vec!["http://example.com".to_owned()],
                }
            ),
            frontier
        ));
        for invalid in [0.0, -1.0, f64::NAN] {
            assert!(!valid_intelligence_frontier(&IntelligenceFrontier {
                eci: invalid,
                ..frontier.clone()
            }));
        }
        assert!(!valid_intelligence_frontier(&IntelligenceFrontier {
            benchmarks_since: frontier.as_of_date.clone(),
            ..frontier.clone()
        }));
    }

    #[test]
    fn only_a_supported_model_needs_a_score() {
        let source = catalog_source().expect("catalog source should be valid");
        let frontier = &source.intelligence_frontier;
        let mut model = source
            .models
            .iter()
            .find(|model| model.support.is_available())
            .expect("the catalog has a supported model")
            .clone();
        assert!(valid_model_score(&model, frontier));
        model.intelligence = None;
        assert!(!valid_model_score(&model, frontier));
        model.support = DeclaredSupport::Deprecated {
            since: "2026-01-01".to_owned(),
            replacement: "qwen3.5-4b:gguf:q4".parse().expect("model ID"),
            reason: "superseded".to_owned(),
        };
        assert!(valid_model_score(&model, frontier));
        model.support = DeclaredSupport::Disabled {
            reason: "not implemented yet".to_owned(),
        };
        assert!(valid_model_score(&model, frontier));
    }

    #[test]
    fn parameterization_requires_meaningful_architecture_counts() {
        assert!(valid_parameterization(&ModelParameterization::Dense {
            total_parameters: 8_000_000_000,
        }));
        assert!(!valid_parameterization(&ModelParameterization::Dense {
            total_parameters: 0,
        }));
        assert!(valid_parameterization(
            &ModelParameterization::MixtureOfExperts {
                total_parameters: 35_000_000_000,
                active_parameters: 3_000_000_000,
            },
        ));
        assert!(!valid_parameterization(
            &ModelParameterization::MixtureOfExperts {
                total_parameters: 3_000_000_000,
                active_parameters: 3_000_000_000,
            },
        ));
    }

    #[test]
    fn shard_selector_distinguishes_first_and_later_shards() {
        assert!(is_first_shard("model-00001-of-00003.gguf"));
        assert!(!is_later_shard("model-00001-of-00003.gguf"));
        assert!(is_later_shard("model-00002-of-00003.gguf"));
        assert!(!is_later_shard("model.gguf"));
    }

    #[test]
    fn package_source_must_match_the_authored_repository_and_locked_commit() {
        let repository = "publisher/model";
        let commit = "0123456789abcdef0123456789abcdef01234567";
        let package_source =
            |repository: String, revision: String| ModelPackageSource::HuggingFace {
                repository,
                revision,
            };
        assert!(package_source_matches(
            &package_source(repository.to_owned(), commit.to_owned()),
            repository,
            commit,
        ));
        assert!(!package_source_matches(
            &package_source("other/repository".to_owned(), commit.to_owned()),
            repository,
            commit,
        ));
        assert!(!package_source_matches(
            &package_source(repository.to_owned(), "0".repeat(40)),
            repository,
            commit,
        ));
    }

    fn select(id: &str) -> Result<LockedCatalogVariant, InventoryError> {
        let source = catalog_source().expect("catalog source should be valid");
        let lock = model_catalog_lock().expect("catalog lock should be valid");
        locked_catalog_variant(&source, &lock, &id.parse().expect("valid model ID"))
    }

    #[test]
    fn catalog_variant_selection_locks_target_and_separate_draft_repositories() {
        let selected = select("qwen3.5-4b:gguf:q4").unwrap();
        let lock = model_catalog_lock().unwrap();
        assert_eq!(selected.declaration.id, "qwen3.5-4b");
        assert_eq!(selected.variant.variant_id, "gguf:q4");
        assert_eq!(selected.variant.format, "UD-Q4_K_XL");
        let draft = "magnitudedev/Qwen3.5-4B-DFlash-GGUF";
        assert_eq!(selected.declaration.draft_repository(), Some(draft));
        assert_eq!(
            selected.repositories,
            BTreeMap::from([
                (
                    "unsloth/Qwen3.5-4B-MTP-GGUF".to_owned(),
                    lock["unsloth/Qwen3.5-4B-MTP-GGUF"].clone()
                ),
                (draft.to_owned(), lock[draft].clone()),
            ])
        );
    }

    #[test]
    fn catalog_variant_selection_locks_its_own_sources_repository() {
        let lock = model_catalog_lock().unwrap();
        let shared_draft = "unsloth/Qwen3.8-Flash-Next-GGUF";
        for (id, repository, license) in [
            ("qwen3.8-flash-next:gguf:q6", shared_draft, "qwen-community-1.0"),
            (
                "qwen3.8-flash-next:gguf:q2",
                "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
                "apache-2.0",
            ),
        ] {
            let selected = select(id).unwrap();
            assert_eq!(selected.source.repository, repository, "{id}");
            assert_eq!(selected.source.license, license, "{id}");
            assert_eq!(
                selected.repositories,
                [repository, shared_draft]
                    .into_iter()
                    .map(|repository| (repository.to_owned(), lock[repository].clone()))
                    .collect::<BTreeMap<_, _>>(),
                "{id}"
            );
        }
    }

    #[test]
    fn catalog_variant_selection_needs_no_draft_repository_for_embedded_or_absent_drafts() {
        let source = catalog_source().unwrap();
        let lock = model_catalog_lock().unwrap();
        for declaration in source
            .models
            .iter()
            .filter(|declaration| declaration.draft_repository().is_none())
        {
            for (model_source, variant) in declaration.variants() {
                let id = format!("{}:{}", declaration.id, variant.variant_id);
                assert_eq!(
                    select(&id).unwrap().repositories,
                    BTreeMap::from([(
                        model_source.repository.clone(),
                        lock[&model_source.repository].clone()
                    )]),
                    "{id}"
                );
            }
        }
    }

    #[test]
    fn catalog_variant_selection_rejects_unknown_and_non_catalog_ids() {
        assert!(matches!(
            select("no-such-model:gguf:q4"),
            Err(InventoryError::NotFound(_))
        ));
        assert!(matches!(
            select("qwen3.5-4b:gguf:q3"),
            Err(InventoryError::NotFound(_))
        ));
        assert!(matches!(
            select("hf:unsloth/Qwen3.5-4B-MTP-GGUF/Qwen3.5-4B-UD-Q4_K_XL.gguf"),
            Err(InventoryError::InvalidRequest(_))
        ));
    }

    #[test]
    fn primary_gguf_is_the_single_format_match_excluding_projectors_imatrix_and_later_shards() {
        let snapshot = repository_snapshot(&[
            "Model-UD-Q4_K_XL.gguf",
            "Model-UD-Q5_K_XL.gguf",
            "mmproj-UD-Q4_K_XL.gguf",
            "imatrix-UD-Q4_K_XL.gguf",
            "Q8/Model-UD-Q8_K_XL-00001-of-00002.gguf",
            "Q8/Model-UD-Q8_K_XL-00002-of-00002.gguf",
        ]);
        assert_eq!(
            select_primary_gguf(&snapshot, "UD-Q4_K_XL").unwrap().path,
            Path::new("Model-UD-Q4_K_XL.gguf")
        );
        assert_eq!(
            select_primary_gguf(&snapshot, "ud-q8_k_xl").unwrap().path,
            Path::new("Q8/Model-UD-Q8_K_XL-00001-of-00002.gguf")
        );
        assert!(select_primary_gguf(&snapshot, "Q6_K").is_err());
        assert!(select_primary_gguf(&snapshot, "UD-Q").is_err());
    }

    #[test]
    fn declared_primary_gguf_is_an_exact_repository_file_or_first_shard() {
        let snapshot = repository_snapshot(&[
            "Q8_0/Model-Q8_0-00001-of-00002.gguf",
            "Q8_0/Model-Q8_0-00002-of-00002.gguf",
            "MTP/mtp-Model-Q8_0.gguf",
            "mmproj-BF16.gguf",
        ]);
        assert!(select_primary_gguf(&snapshot, "Q8_0").is_err());
        assert_eq!(
            declared_primary_gguf(&snapshot, Path::new("Q8_0/Model-Q8_0-00001-of-00002.gguf"))
                .unwrap()
                .path,
            Path::new("Q8_0/Model-Q8_0-00001-of-00002.gguf")
        );
        assert!(
            declared_primary_gguf(&snapshot, Path::new("Q8_0/Model-Q8_0-00002-of-00002.gguf"))
                .is_err()
        );
        assert!(declared_primary_gguf(&snapshot, Path::new("mmproj-BF16.gguf")).is_err());
        assert!(declared_primary_gguf(&snapshot, Path::new("Q8_0/Missing.gguf")).is_err());
    }

    #[test]
    fn deepseek_shared_engram_and_glm_metadata_only_shard_1_resolve_and_classify() {
        // 1. ISTA DeepSeek-V4.1-Flash: 3 builds sharing shard 6 as DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf.
        let mut ista_paths = vec![
            "DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf".to_owned(),
            "mmproj-DeepSeek-V4.1-Flash-1017-BF16.gguf".to_owned(),
        ];
        for format in ["IQ1_M", "IQ2_XXS", "IQ2_XS"] {
            for idx in 1..=5 {
                ista_paths.push(format!(
                    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-{format}-{idx:05}-of-00006.gguf"
                ));
            }
        }
        let ista_refs = ista_paths.iter().map(String::as_str).collect::<Vec<_>>();
        let ista_snapshot = repository_snapshot(&ista_refs);
        let candidate_files = ista_snapshot
            .gguf_files
            .iter()
            .map(|file| crate::candidate_files::CandidateGguf {
                path: file.path.clone(),
                size_bytes: file.size_bytes,
            })
            .collect::<Vec<_>>();
        let mut split_headers = BTreeMap::new();
        split_headers.insert(
            PathBuf::from("DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf"),
            Some(crate::candidate_files::SplitMetadata {
                index: 5,
                count: 6,
                tensors_count: Some(1043),
            }),
        );
        for format in ["IQ1_M", "IQ2_XXS", "IQ2_XS"] {
            split_headers.insert(
                PathBuf::from(format!(
                    "DeepSeek-V4.1-Flash-1017-GSQ-RCO-{format}-00002-of-00006.gguf"
                )),
                Some(crate::candidate_files::SplitMetadata {
                    index: 1,
                    count: 6,
                    tensors_count: Some(1043),
                }),
            );
        }
        let resolved = crate::candidate_files::resolve_split_groups(
            crate::candidate_files::partition_ggufs(&candidate_files),
            &split_headers,
        );
        for format in ["IQ1_M", "IQ2_XXS", "IQ2_XS"] {
            let primary = select_primary_gguf(&ista_snapshot, format).unwrap();
            let group = resolved
                .iter()
                .find(|group| group.primary().path == primary.path)
                .expect("shared Engram shard completes variant");
            assert_eq!(group.files.len(), 6);
            assert_eq!(
                group.files[5].path,
                PathBuf::from("DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf")
            );
        }

        // 2. Unsloth GLM-5.3-Flash: metadata-only 0-tensor shard 1 with nextn_predict_layers = 1.
        let glm_snapshot = repository_snapshot(&[
            "UD-IQ2_XXS/GLM-5.3-Flash-UD-IQ2_XXS-00001-of-00004.gguf",
            "UD-IQ2_XXS/GLM-5.3-Flash-UD-IQ2_XXS-00002-of-00004.gguf",
            "UD-IQ2_XXS/GLM-5.3-Flash-UD-IQ2_XXS-00003-of-00004.gguf",
            "UD-IQ2_XXS/GLM-5.3-Flash-UD-IQ2_XXS-00004-of-00004.gguf",
            "mmproj-BF16.gguf",
        ]);
        assert_eq!(
            select_primary_gguf(&glm_snapshot, "UD-IQ2_XXS").unwrap().path,
            Path::new("UD-IQ2_XXS/GLM-5.3-Flash-UD-IQ2_XXS-00001-of-00004.gguf")
        );
        let mut glm_header = Vec::new();
        glm_header.extend_from_slice(b"GGUF");
        glm_header.extend_from_slice(&3_u32.to_le_bytes());
        glm_header.extend_from_slice(&0_u64.to_le_bytes());
        glm_header.extend_from_slice(&5_u64.to_le_bytes());
        let write_str = |buf: &mut Vec<u8>, s: &str| {
            buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
            buf.extend_from_slice(s.as_bytes());
        };
        let write_kv_str = |buf: &mut Vec<u8>, k: &str, v: &str| {
            write_str(buf, k);
            buf.extend_from_slice(&8_u32.to_le_bytes());
            write_str(buf, v);
        };
        let write_kv_u32 = |buf: &mut Vec<u8>, k: &str, v: u32| {
            write_str(buf, k);
            buf.extend_from_slice(&4_u32.to_le_bytes());
            buf.extend_from_slice(&v.to_le_bytes());
        };
        write_kv_str(&mut glm_header, "general.architecture", "glm5-next");
        write_kv_u32(&mut glm_header, "glm5-next.block_count", 46);
        write_kv_u32(&mut glm_header, "glm5-next.nextn_predict_layers", 1);
        write_kv_u32(&mut glm_header, "split.no", 0);
        write_kv_u32(&mut glm_header, "split.count", 4);
        let architectures = crate::candidate_files::RecognizedArchitectures::new(["glm5-next"], []);
        let classified =
            crate::candidate_files::classify_header(glm_header, &architectures).unwrap();
        assert_eq!(
            classified.role,
            crate::candidate_files::CandidateFileRole::Target {
                architecture: Some("glm5-next".to_owned()),
            }
        );
        assert!(classified.embedded_mtp);
    }
}
