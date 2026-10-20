//! Lineage: which catalog model a repository or file derives from.
//!
//! A catalog model's upstream is its target repository's Hub `baseModels` at the locked commit,
//! read at runtime and persisted through the commit-pinned repository snapshot cache, so it is
//! fetched once per release and never follows a mutable revision, and then those base models' own
//! `baseModels` (a second hop). A candidate's lineage is its repository, the Hub `baseModels` up
//! to two hops, and the repositories its GGUF names in `general.base_model.N.repo_url`. Both sides
//! follow two hops because a catalog target is often a quantization of a derivative (a QAT
//! checkpoint of an instruct model) whose parent community quantizations name instead. Every Hub
//! answer is persisted, so what is on this computer (`known_*`) answers without a request, offline
//! included. Lineage is a display hint and a drafter-selection signal; the engine verdict alone
//! decides compatibility.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use magnitude_service_contracts::models::{
    ModelFileRole, ModelId, ModelPackage, ModelPackageSource, RecommendableModel,
    RecommendableModelCatalog, ServableModelBundle,
};
use magnitude_service_contracts::{HuggingFaceRepositoryRequest, InventoryError};
use serde::Deserialize;

use crate::cache::ModelIndexKind;
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

    /// The catalog variant whose target weights in `repository` are exactly `weights`, each path
    /// with its SHA-256, at whatever commit: such a file is that catalog target's bytes, which
    /// install as that catalog model (packages are identified by their files' contents).
    #[must_use]
    pub fn target_with_weights(
        &self,
        repository: &str,
        weights: &BTreeSet<(PathBuf, String)>,
    ) -> Option<CatalogLineageMatch> {
        if weights.is_empty() {
            return None;
        }
        self.catalog
            .models
            .iter()
            .find(|model| {
                let target = catalog_target(model);
                matches!(
                    &target.source,
                    ModelPackageSource::HuggingFace { repository: target_repository, .. }
                        if target_repository.eq_ignore_ascii_case(repository)
                ) && target
                    .files
                    .iter()
                    .filter(|file| file.role == ModelFileRole::Weights)
                    .map(|file| (file.path.clone(), file.sha256.clone()))
                    .collect::<BTreeSet<_>>()
                    == *weights
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

    /// Every catalog model whose target repository or upstream intersects `lineage` (lowercased
    /// repositories), each as its first variant, in catalog order.
    pub async fn match_lineages(&self, lineage: &BTreeSet<String>) -> Vec<CatalogLineageMatch> {
        if lineage.is_empty() {
            return Vec::new();
        }
        let upstreams = self.upstreams().await;
        self.lineage_matches(lineage, &upstreams)
    }

    /// `match_lineages` without a request: catalog upstreams match as far as the Hub's answers
    /// are on this computer.
    #[must_use]
    pub fn match_known_lineages(&self, lineage: &BTreeSet<String>) -> Vec<CatalogLineageMatch> {
        if lineage.is_empty() {
            return Vec::new();
        }
        self.lineage_matches(lineage, &self.known_upstreams())
    }

    fn lineage_matches(
        &self,
        lineage: &BTreeSet<String>,
        upstreams: &Upstreams,
    ) -> Vec<CatalogLineageMatch> {
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

    /// Each catalog model's upstream (see the module documentation), read from the Hub where it
    /// is not on this computer, and kept once every one is known.
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
                let upstream = async {
                    let first_hop = next_hop(
                        repository,
                        pinned_repository_base_models(&self.models, repository, revision).await?,
                    );
                    let mut second_hop = Vec::new();
                    for base in &first_hop {
                        match self.hub.base_models(base).await {
                            Ok(bases) => second_hop.push(bases),
                            Err(InventoryError::NotFound(_)) => {}
                            Err(error) => return Err(error),
                        }
                    }
                    Ok::<_, InventoryError>(upstream(first_hop, second_hop))
                };
                match upstream.await {
                    Ok(upstream) => Some((candidate.model_id, upstream)),
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

    /// `upstreams` from what is on this computer: every one once all are known, else each catalog
    /// model's as far as its Hub answers are persisted.
    fn known_upstreams(&self) -> Arc<Upstreams> {
        if let Ok(cached) = self.upstreams.try_lock()
            && let Some(upstreams) = cached.as_ref()
        {
            return Arc::clone(upstreams);
        }
        Arc::new(
            self.representatives()
                .filter_map(|candidate| {
                    let ModelPackageSource::HuggingFace {
                        repository,
                        revision,
                    } = &catalog_target(&candidate.model).source
                    else {
                        return None;
                    };
                    let snapshot = crate::preview::cached_hugging_face_repository(
                        &self.models,
                        &HuggingFaceRepositoryRequest {
                            repository: repository.clone(),
                            revision: revision.clone(),
                        },
                    )?;
                    let first_hop = next_hop(repository, snapshot.base_models);
                    let second_hop = first_hop
                        .iter()
                        .filter_map(|base| self.hub.known_base_models(base))
                        .collect();
                    Some((candidate.model_id, upstream(first_hop, second_hop)))
                })
                .collect(),
        )
    }
}

/// The base models lineage follows from `repository`: at most `MAX_BASE_MODELS_PER_HOP`, never
/// the repository itself.
fn next_hop(repository: &str, base_models: Vec<String>) -> Vec<String> {
    base_models
        .into_iter()
        .filter(|base| !base.eq_ignore_ascii_case(repository))
        .take(MAX_BASE_MODELS_PER_HOP)
        .collect()
}

/// The repositories of two hops, lowercased.
fn upstream(first_hop: Vec<String>, second_hop: Vec<Vec<String>>) -> BTreeSet<String> {
    first_hop
        .into_iter()
        .chain(
            second_hop
                .into_iter()
                .flat_map(|bases| bases.into_iter().take(MAX_BASE_MODELS_PER_HOP)),
        )
        .map(|base| base.to_lowercase())
        .collect()
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
) -> Result<Vec<String>, InventoryError> {
    let snapshot = crate::preview::pinned_hugging_face_repository(
        models,
        HuggingFaceRepositoryRequest {
            repository: repository.to_owned(),
            revision: commit.to_owned(),
        },
    )
    .await?;
    Ok(snapshot.base_models)
}

/// Hub lineage of arbitrary repositories, fetched with bounded concurrency, kept in memory for
/// `LINEAGE_TTL` and persisted, so a repository's lineage answers without a request once known.
pub struct HubLineage {
    models: Arc<ManagedModelStore>,
    cache: std::sync::Mutex<BTreeMap<String, (Instant, Vec<String>)>>,
    permits: tokio::sync::Semaphore,
}

#[derive(Deserialize)]
struct HubBaseModelsResponse {
    #[serde(default, rename = "baseModels")]
    base_models: Option<serde_json::Value>,
}

/// A repository's direct Hub `baseModels` as last read.
#[derive(Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct PersistedBaseModels {
    base_models: Vec<String>,
}

impl HubLineage {
    fn new(models: Arc<ManagedModelStore>) -> Self {
        Self {
            models,
            cache: std::sync::Mutex::new(BTreeMap::new()),
            permits: tokio::sync::Semaphore::new(MAX_LINEAGE_REQUESTS),
        }
    }

    fn persisted_key(&self, repository: &str) -> String {
        format!(
            "{}:{}",
            self.models.client.endpoint(),
            repository.to_lowercase()
        )
    }

    /// The direct Hub `baseModels` of `repository` as last read, without a request.
    #[must_use]
    pub fn known_base_models(&self, repository: &str) -> Option<Vec<String>> {
        let remembered = self.cache.lock().ok().and_then(|cache| {
            cache
                .get(&repository.to_lowercase())
                .map(|(_, base_models)| base_models.clone())
        });
        remembered.or_else(|| {
            self.models
                .cache
                .read_index::<PersistedBaseModels>(
                    ModelIndexKind::HuggingFaceBaseModels,
                    &self.persisted_key(repository),
                )
                .map(|persisted| persisted.base_models)
        })
    }

    /// The direct Hub `baseModels` of `repository`, as the Hub spells them; the last answer read
    /// when the Hub can't be reached.
    pub async fn base_models(&self, repository: &str) -> Result<Vec<String>, InventoryError> {
        let key = repository.to_lowercase();
        if let Ok(mut cache) = self.cache.lock() {
            cache.retain(|_, (captured, _)| captured.elapsed() <= LINEAGE_TTL);
            if let Some((_, base_models)) = cache.get(&key) {
                return Ok(base_models.clone());
            }
        }
        match self.fetch_base_models(repository).await {
            Ok(base_models) => {
                self.models.cache.write_index(
                    ModelIndexKind::HuggingFaceBaseModels,
                    &self.persisted_key(repository),
                    &PersistedBaseModels {
                        base_models: base_models.clone(),
                    },
                );
                self.remember(key, &base_models);
                Ok(base_models)
            }
            Err(error @ InventoryError::NotFound(_)) => Err(error),
            Err(error) => self.known_base_models(repository).ok_or(error),
        }
    }

    fn remember(&self, key: String, base_models: &[String]) {
        let Ok(mut cache) = self.cache.lock() else {
            return;
        };
        cache.insert(key, (Instant::now(), base_models.to_vec()));
        if cache.len() > MAX_LINEAGE_CACHE_ENTRIES
            && let Some(oldest) = cache
                .iter()
                .min_by_key(|(_, (captured, _))| *captured)
                .map(|(key, _)| key.clone())
        {
            cache.remove(&oldest);
        }
    }

    async fn fetch_base_models(&self, repository: &str) -> Result<Vec<String>, InventoryError> {
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
        let token = self.models.hub_token();
        let response =
            crate::preview::send_hub_get(&self.models.http, token.as_deref(), url).await?;
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
        Ok(body
            .base_models
            .as_ref()
            .map(hub_base_model_ids)
            .unwrap_or_default())
    }

    /// `repository` with its Hub `baseModels` up to two hops (GGUF, then finetune, then base),
    /// lowercased. `first_hop` supplies base models already known (from search expansion or a
    /// repository snapshot). Unavailable lineage narrows the set rather than failing.
    pub async fn lineage(
        &self,
        repository: &str,
        first_hop: Option<Vec<String>>,
    ) -> BTreeSet<String> {
        let first_hop = match first_hop {
            Some(first_hop) => first_hop,
            None => self.base_models(repository).await.unwrap_or_default(),
        };
        let first_hop = next_hop(repository, first_hop);
        let mut pending = Vec::new();
        for base in &first_hop {
            pending.push(self.base_models(base));
        }
        let second_hop = futures_util::stream::iter(pending)
            .buffer_unordered(MAX_LINEAGE_REQUESTS)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(Result::unwrap_or_default)
            .collect();
        repository_lineage(repository, first_hop, second_hop)
    }

    /// `lineage` from the Hub answers on this computer, without a request.
    #[must_use]
    pub fn known_lineage(
        &self,
        repository: &str,
        first_hop: Option<Vec<String>>,
    ) -> BTreeSet<String> {
        let first_hop = next_hop(
            repository,
            first_hop
                .or_else(|| self.known_base_models(repository))
                .unwrap_or_default(),
        );
        let second_hop = first_hop
            .iter()
            .filter_map(|base| self.known_base_models(base))
            .collect();
        repository_lineage(repository, first_hop, second_hop)
    }
}

/// A repository's lineage: itself and its two hops, lowercased.
fn repository_lineage(
    repository: &str,
    first_hop: Vec<String>,
    second_hop: Vec<Vec<String>>,
) -> BTreeSet<String> {
    let mut lineage = upstream(first_hop, second_hop);
    lineage.insert(repository.to_lowercase());
    lineage
}

/// Repository IDs in a Hub `baseModels` value: `{relation, models: [{id}]}`, or a list of those.
#[must_use]
fn hub_base_model_ids(value: &serde_json::Value) -> Vec<String> {
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
