//! Lineage: which catalog model a repository or file derives from.
//!
//! A catalog model's upstream is its target repository's Hub `baseModels` at the locked commit,
//! read at runtime and persisted through the commit-pinned repository snapshot cache, so it is
//! fetched once per release and never follows a mutable revision. A candidate's lineage is its
//! repository, the Hub `baseModels` up to two hops, and the repositories its GGUF names in
//! `general.base_model.N.repo_url`. Lineage is a display hint and a drafter-selection signal; the
//! engine verdict alone decides compatibility.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use futures_util::future::BoxFuture;
use magnitude_service_contracts::models::{
    ModelFileRole, ModelId, ModelPackage, ModelPackageSource, RecommendableModel,
    RecommendableModelCatalog, ServableModelBundle,
};
use magnitude_service_contracts::{HuggingFaceRepositoryRequest, InventoryError};
use serde::Deserialize;

use crate::inventory::ManagedModelStore;

const LINEAGE_TTL: Duration = Duration::from_secs(60 * 60);
const MAX_LINEAGE_CACHE_ENTRIES: usize = 2048;
const MAX_LINEAGE_REQUESTS: usize = 4;
const MAX_BASE_MODELS_PER_HOP: usize = 8;
const MAX_LINEAGE_METADATA_BYTES: usize = 4 * 1024 * 1024;

/// The catalog's view for lineage: exact targets, repositories it uses, and each catalog model's
/// upstream repositories.
pub struct CatalogLineage {
    models: Arc<ManagedModelStore>,
    catalog: RecommendableModelCatalog,
    hub: HubLineage,
    upstreams: tokio::sync::Mutex<Option<Arc<Upstreams>>>,
}

/// Each catalog model's upstream repositories, lowercased.
type Upstreams = BTreeMap<ModelId, BTreeSet<String>>;

/// One catalog match: the catalog variant a repository or file derives from.
#[derive(Clone, Debug, PartialEq)]
pub struct CatalogLineageMatch {
    pub model: RecommendableModel,
    pub model_id: ModelId,
}

impl CatalogLineage {
    #[must_use]
    pub fn new(models: Arc<ManagedModelStore>, catalog: RecommendableModelCatalog) -> Arc<Self> {
        Arc::new(Self {
            hub: HubLineage::new(Arc::clone(&models)),
            models,
            catalog,
            upstreams: tokio::sync::Mutex::new(None),
        })
    }

    #[must_use]
    pub fn catalog(&self) -> &RecommendableModelCatalog {
        &self.catalog
    }

    #[must_use]
    pub fn hub(&self) -> &HubLineage {
        &self.hub
    }

    /// Every repository a catalog package uses, lowercased: targets and drafters.
    #[must_use]
    pub fn repositories(&self) -> BTreeSet<String> {
        self.catalog
            .models
            .iter()
            .flat_map(|model| bundle_packages(&model.configuration.bundle))
            .filter_map(|package| match &package.source {
                ModelPackageSource::HuggingFace { repository, .. } => {
                    Some(repository.to_lowercase())
                }
                ModelPackageSource::Local { .. } => None,
            })
            .collect()
    }

    /// The catalog variant whose target is exactly `repository` at `commit` with primary GGUF
    /// `primary`.
    #[must_use]
    pub fn exact_target(
        &self,
        repository: &str,
        commit: &str,
        primary: &Path,
    ) -> Option<CatalogLineageMatch> {
        self.catalog
            .models
            .iter()
            .find(|model| {
                let target = catalog_target(model);
                matches!(
                    &target.source,
                    ModelPackageSource::HuggingFace { repository: target_repository, revision }
                        if target_repository.eq_ignore_ascii_case(repository) && revision == commit
                ) && primary_weights(target).as_deref() == Some(primary)
            })
            .map(catalog_match)
    }

    /// The first variant (in catalog order) of every catalog model.
    pub fn representatives(&self) -> impl Iterator<Item = CatalogLineageMatch> + '_ {
        let mut seen = BTreeSet::new();
        self.catalog
            .models
            .iter()
            .filter(move |model| seen.insert(model.model_id.clone()))
            .map(catalog_match)
    }

    /// The catalog variant `model_id` names.
    #[must_use]
    pub fn variant(&self, model_id: &ModelId) -> Option<CatalogLineageMatch> {
        self.catalog
            .models
            .iter()
            .map(catalog_match)
            .find(|candidate| &candidate.model_id == model_id)
    }

    /// The catalog model whose target repository or upstream intersects `lineage` (lowercased
    /// repositories), as its first variant: the first of [`Self::match_lineages`].
    pub async fn match_lineage(&self, lineage: &BTreeSet<String>) -> Option<CatalogLineageMatch> {
        self.match_lineages(lineage).await.into_iter().next()
    }

    /// Every catalog model whose target repository or upstream intersects `lineage` (lowercased
    /// repositories), each as its first variant, in catalog order.
    pub async fn match_lineages(&self, lineage: &BTreeSet<String>) -> Vec<CatalogLineageMatch> {
        if lineage.is_empty() {
            return Vec::new();
        }
        let upstreams = self.upstreams().await;
        self.representatives()
            .filter(|candidate| {
                let target = catalog_target(&candidate.model);
                let target_repository = match &target.source {
                    ModelPackageSource::HuggingFace { repository, .. } => {
                        Some(repository.to_lowercase())
                    }
                    ModelPackageSource::Local { .. } => None,
                };
                target_repository
                    .as_ref()
                    .is_some_and(|repository| lineage.contains(repository))
                    || upstreams
                        .get(&candidate.model_id)
                        .is_some_and(|upstream| !upstream.is_disjoint(lineage))
            })
            .collect()
    }

    /// The upstream repositories of `model_id`'s catalog model (lowercased), with its target
    /// repository.
    pub async fn lineage_of(&self, model_id: &ModelId) -> BTreeSet<String> {
        let mut lineage = self
            .upstreams()
            .await
            .get(model_id)
            .cloned()
            .unwrap_or_default();
        if let Some(candidate) = self.variant(model_id)
            && let ModelPackageSource::HuggingFace { repository, .. } =
                &catalog_target(&candidate.model).source
        {
            lineage.insert(repository.to_lowercase());
        }
        lineage
    }

    async fn upstreams(&self) -> Arc<Upstreams> {
        let mut cached = self.upstreams.lock().await;
        if let Some(upstreams) = cached.as_ref() {
            return Arc::clone(upstreams);
        }
        let representatives = self.representatives().collect::<Vec<_>>();
        let resolved = futures_util::stream::iter(representatives)
            .map(|candidate| async move {
                let target = catalog_target(&candidate.model);
                let ModelPackageSource::HuggingFace {
                    repository,
                    revision,
                } = &target.source
                else {
                    return Some((candidate.model_id, BTreeSet::new()));
                };
                match pinned_repository_base_models(&self.models, repository, revision).await {
                    Ok(base_models) => Some((candidate.model_id, base_models)),
                    Err(error) => {
                        tracing::warn!(
                            repository = %repository,
                            %error,
                            "catalog upstream lineage is unavailable"
                        );
                        None
                    }
                }
            })
            .buffer_unordered(MAX_LINEAGE_REQUESTS)
            .collect::<Vec<_>>()
            .await;
        let complete = resolved.iter().all(Option::is_some);
        let upstreams = Arc::new(resolved.into_iter().flatten().collect::<BTreeMap<_, _>>());
        if complete {
            *cached = Some(Arc::clone(&upstreams));
        }
        upstreams
    }
}

fn catalog_match(model: &RecommendableModel) -> CatalogLineageMatch {
    CatalogLineageMatch {
        model_id: ModelId::catalog(&model.model_id, &model.variant_id),
        model: model.clone(),
    }
}

/// The catalog variant's target package.
#[must_use]
pub fn catalog_target(model: &RecommendableModel) -> &ModelPackage {
    match &model.configuration.bundle {
        ServableModelBundle::Standalone { package } => package,
        ServableModelBundle::SpeculativeDecoding { target, .. } => target,
    }
}

/// A package's primary weights file: its only weights file, or a split file's first shard.
#[must_use]
pub fn primary_weights(package: &ModelPackage) -> Option<PathBuf> {
    package
        .files
        .iter()
        .filter(|file| file.role == ModelFileRole::Weights)
        .map(|file| file.path.clone())
        .min()
}

fn bundle_packages(bundle: &ServableModelBundle) -> Vec<&ModelPackage> {
    match bundle {
        ServableModelBundle::Standalone { package } => vec![package],
        ServableModelBundle::SpeculativeDecoding {
            target,
            draft_source,
            ..
        } => match draft_source {
            magnitude_service_contracts::models::SpeculativeDraftSource::Embedded => vec![target],
            magnitude_service_contracts::models::SpeculativeDraftSource::Separate { draft } => {
                vec![target, draft]
            }
        },
    }
}

/// The Hub `baseModels` of `repository` at an immutable commit, read through the persisted
/// repository snapshot without a request when it is cached.
async fn pinned_repository_base_models(
    models: &Arc<ManagedModelStore>,
    repository: &str,
    commit: &str,
) -> Result<BTreeSet<String>, InventoryError> {
    let snapshot = crate::preview::pinned_hugging_face_repository(
        models,
        HuggingFaceRepositoryRequest {
            repository: repository.to_owned(),
            revision: commit.to_owned(),
        },
    )
    .await?;
    Ok(snapshot
        .base_models
        .iter()
        .map(|base| base.to_lowercase())
        .collect())
}

/// A catalog model's architecture and parameter count, read from its release header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CatalogShape {
    pub model_id: ModelId,
    pub architecture: String,
    pub parameters: u64,
}

/// The catalog's shapes, computed by whoever holds the release headers.
pub trait CatalogShapes: Send + Sync {
    fn catalog_shapes(&self) -> BoxFuture<'_, Vec<CatalogShape>>;
}

/// Parameter counts within this fraction of a catalog model's are the same size: a GGUF with or
/// without its MTP layers differs by about 3%, while neighbouring sizes of one family differ by
/// far more.
const SHAPE_TOLERANCE: f64 = 0.05;

/// The catalog model of the same architecture whose parameter count is closest to `parameters`,
/// within `SHAPE_TOLERANCE`, and whose size does not contradict the size `name` spells (a
/// repository named E2B is never a variant of the E4B, whatever its file sizes add up to). The
/// weakest lineage signal: a display hint for repositories that declare no lineage at all.
#[must_use]
pub fn shape_match<'a>(
    shapes: &'a [CatalogShape],
    architecture: &str,
    parameters: u64,
    name: &str,
) -> Option<&'a CatalogShape> {
    let named = name_sizes(name);
    shapes
        .iter()
        .filter(|shape| shape.architecture == architecture && shape.parameters > 0)
        .filter(|shape| {
            let sizes = name_sizes(shape.model_id.as_str());
            named.is_empty() || sizes.is_empty() || !named.is_disjoint(&sizes)
        })
        .map(|shape| {
            let difference = shape.parameters.abs_diff(parameters) as f64 / shape.parameters as f64;
            (shape, difference)
        })
        .filter(|(_, difference)| *difference <= SHAPE_TOLERANCE)
        .min_by(|left, right| left.1.total_cmp(&right.1))
        .map(|(shape, _)| shape)
}

/// The sizes a model name spells, such as `4b`, `e2b`, `35b`, `a3b` or `270m`, lowercased.
fn name_sizes(name: &str) -> BTreeSet<String> {
    name.to_lowercase()
        .split(|character: char| !(character.is_ascii_alphanumeric() || character == '.'))
        .filter(|token| {
            let count = token
                .strip_prefix(['a', 'e'])
                .unwrap_or(token)
                .strip_suffix(['b', 'm']);
            count.is_some_and(|count| {
                count.starts_with(|character: char| character.is_ascii_digit())
                    && count
                        .chars()
                        .all(|character| character.is_ascii_digit() || character == '.')
            })
        })
        .map(str::to_owned)
        .collect()
}

/// Hub lineage of arbitrary repositories, cached in memory and fetched with bounded concurrency.
pub struct HubLineage {
    models: Arc<ManagedModelStore>,
    cache: tokio::sync::Mutex<BTreeMap<String, (Instant, Vec<String>)>>,
    permits: tokio::sync::Semaphore,
}

#[derive(Deserialize)]
struct HubBaseModelsResponse {
    #[serde(default, rename = "baseModels")]
    base_models: Option<serde_json::Value>,
}

impl HubLineage {
    fn new(models: Arc<ManagedModelStore>) -> Self {
        Self {
            models,
            cache: tokio::sync::Mutex::new(BTreeMap::new()),
            permits: tokio::sync::Semaphore::new(MAX_LINEAGE_REQUESTS),
        }
    }

    /// The direct Hub `baseModels` of `repository`, as the Hub spells them.
    pub async fn base_models(&self, repository: &str) -> Result<Vec<String>, InventoryError> {
        let key = repository.to_lowercase();
        {
            let mut cache = self.cache.lock().await;
            cache.retain(|_, (captured, _)| captured.elapsed() <= LINEAGE_TTL);
            if let Some((_, base_models)) = cache.get(&key) {
                return Ok(base_models.clone());
            }
        }
        let _permit = self
            .permits
            .acquire()
            .await
            .map_err(|error| InventoryError::Internal(error.to_string()))?;
        let mut url = reqwest::Url::parse(self.models.client.endpoint())
            .map_err(|error| InventoryError::Internal(error.to_string()))?;
        {
            let mut segments = url
                .path_segments_mut()
                .map_err(|()| InventoryError::Internal("invalid hub endpoint".to_owned()))?;
            segments.pop_if_empty().push("api").push("models");
            for segment in repository.split('/') {
                segments.push(segment);
            }
        }
        url.query_pairs_mut().append_pair("expand", "baseModels");
        let response =
            crate::preview::send_hub_get(&self.models.http, self.models.hub_token(), url).await?;
        if response.status() == reqwest::StatusCode::NOT_FOUND {
            return Err(InventoryError::NotFound(repository.to_owned()));
        }
        if !response.status().is_success() {
            return Err(InventoryError::Upstream(format!(
                "Hugging Face lineage returned HTTP {}",
                response.status()
            )));
        }
        let body: HubBaseModelsResponse = serde_json::from_slice(
            &crate::preview::bounded_response_bytes(response, MAX_LINEAGE_METADATA_BYTES).await?,
        )
        .map_err(|error| InventoryError::Upstream(error.to_string()))?;
        let base_models = body
            .base_models
            .as_ref()
            .map(hub_base_model_ids)
            .unwrap_or_default();
        let mut cache = self.cache.lock().await;
        cache.insert(key, (Instant::now(), base_models.clone()));
        if cache.len() > MAX_LINEAGE_CACHE_ENTRIES
            && let Some(oldest) = cache
                .iter()
                .min_by_key(|(_, (captured, _))| *captured)
                .map(|(key, _)| key.clone())
        {
            cache.remove(&oldest);
        }
        Ok(base_models)
    }

    /// `repository` with its Hub `baseModels` up to two hops (GGUF, then finetune, then base),
    /// lowercased. `first_hop` supplies base models already known (from search expansion or a
    /// repository snapshot). Unavailable lineage narrows the set rather than failing.
    pub async fn lineage(
        &self,
        repository: &str,
        first_hop: Option<Vec<String>>,
    ) -> BTreeSet<String> {
        let mut lineage = BTreeSet::from([repository.to_lowercase()]);
        let first_hop = match first_hop {
            Some(first_hop) => first_hop,
            None => self.base_models(repository).await.unwrap_or_default(),
        };
        let first_hop = first_hop
            .into_iter()
            .filter(|base| !lineage.contains(&base.to_lowercase()))
            .take(MAX_BASE_MODELS_PER_HOP)
            .collect::<Vec<_>>();
        lineage.extend(first_hop.iter().map(|base| base.to_lowercase()));
        let mut pending = Vec::new();
        for base in &first_hop {
            pending.push(self.base_models(base));
        }
        let second_hop = futures_util::stream::iter(pending)
            .buffer_unordered(MAX_LINEAGE_REQUESTS)
            .collect::<Vec<_>>()
            .await;
        lineage.extend(second_hop.into_iter().flat_map(|bases| {
            bases
                .unwrap_or_default()
                .into_iter()
                .take(MAX_BASE_MODELS_PER_HOP)
                .map(|base| base.to_lowercase())
        }));
        lineage
    }
}

/// Repository IDs in a Hub `baseModels` value: `{relation, models: [{id}]}`, or a list of those.
#[must_use]
pub(crate) fn hub_base_model_ids(value: &serde_json::Value) -> Vec<String> {
    let mut ids = Vec::new();
    let mut collect = |entry: &serde_json::Value| {
        if let Some(models) = entry.get("models").and_then(serde_json::Value::as_array) {
            ids.extend(
                models
                    .iter()
                    .filter_map(|model| model.get("id").and_then(serde_json::Value::as_str))
                    .map(ToOwned::to_owned),
            );
        }
    };
    match value {
        serde_json::Value::Array(entries) => entries.iter().for_each(&mut collect),
        entry => collect(entry),
    }
    ids.sort();
    ids.dedup();
    ids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_shape_matches_the_closest_catalog_model_of_its_architecture_within_tolerance() {
        let shape = |model_id: &str, architecture: &str, parameters| CatalogShape {
            model_id: model_id.parse().unwrap(),
            architecture: architecture.to_owned(),
            parameters,
        };
        let shapes = [
            shape("qwen3.5-4b:gguf:q4", "qwen35", 4_326_350_848),
            shape("qwen3.5-9b:gguf:q4", "qwen35", 8_953_803_264),
            shape("other-4b:gguf:q4", "llama", 4_205_751_296),
        ];
        assert_eq!(
            shape_match(
                &shapes,
                "qwen35",
                4_205_751_296,
                "someone/Qwen3.5-4B-Abliterated-GGUF"
            )
            .map(|shape| shape.model_id.as_str()),
            Some("qwen3.5-4b:gguf:q4"),
            "the same model without its MTP layers"
        );
        assert_eq!(
            shape_match(&shapes, "qwen35", 4_205_751_296, "someone/Unnamed-GGUF")
                .map(|shape| shape.model_id.as_str()),
            Some("qwen3.5-4b:gguf:q4"),
            "a name without a size leaves the shape to decide"
        );
        assert_eq!(shape_match(&shapes, "qwen35", 2_000_000_000, "a/b"), None);
        assert_eq!(shape_match(&shapes, "gemma4", 4_205_751_296, "a/b"), None);
    }

    #[test]
    fn a_shape_never_contradicts_the_size_the_name_spells() {
        let shapes = [
            CatalogShape {
                model_id: "gemma-4-e2b-it-qat:gguf:q4".parse().unwrap(),
                architecture: "gemma4".to_owned(),
                parameters: 4_628_569_635,
            },
            CatalogShape {
                model_id: "gemma-4-e4b-it-qat:gguf:q4".parse().unwrap(),
                architecture: "gemma4".to_owned(),
                parameters: 7_518_069_290,
            },
        ];
        // weareapexcreators/Gemma-4-E2B-it-LocalAI declares no lineage, and its GGUF metadata
        // totals the E4B's parameter count.
        assert_eq!(
            shape_match(
                &shapes,
                "gemma4",
                7_518_069_290,
                "weareapexcreators/Gemma-4-E2B-it-LocalAI"
            ),
            None
        );
        assert_eq!(
            shape_match(
                &shapes,
                "gemma4",
                7_518_069_290,
                "someone/Gemma-4-E4B-it-LocalAI"
            )
            .map(|shape| shape.model_id.as_str()),
            Some("gemma-4-e4b-it-qat:gguf:q4")
        );
        assert_eq!(
            name_sizes("Qwen/Qwen3.6-35B-A3B"),
            BTreeSet::from(["35b".to_owned(), "a3b".to_owned()])
        );
        assert!(name_sizes("Qwen3.5-Coder").is_empty());
    }

    #[test]
    fn hub_base_models_parse_both_shapes() {
        assert_eq!(
            hub_base_model_ids(&serde_json::json!({
                "relation": "quantized",
                "models": [{"id": "Qwen/Qwen3.6-35B-A3B"}]
            })),
            ["Qwen/Qwen3.6-35B-A3B"]
        );
        assert_eq!(
            hub_base_model_ids(&serde_json::json!([
                {"relation": "finetune", "models": [{"id": "B/b"}]},
                {"relation": "merge", "models": [{"id": "A/a"}, {"id": "B/b"}]}
            ])),
            ["A/a", "B/b"]
        );
    }
}
