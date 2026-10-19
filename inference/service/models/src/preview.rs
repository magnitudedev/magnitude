use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use futures_util::future::BoxFuture;
use magnitude_service_contracts::models::{
    KnownUnsupportedReason, ModelFingerprint, ModelId, ModelPackage, UnsupportedReason,
};
use magnitude_service_contracts::{
    ComponentRelationship, ComponentRole, ContentIdentity, HuggingFaceModelCatalog,
    HuggingFaceModelSearchRequest, HuggingFaceModelSearchResult, HuggingFaceModelSearchResults,
    HuggingFaceRepositoryFile, HuggingFaceRepositoryRequest, HuggingFaceRepositorySnapshot,
    Integrity, InventoryEntryId, InventoryError, ModelComponent, ModelLocation, ModelPreviewSource,
    ModelSource, ResolvedComponent, ResolvedModel,
};
use reqwest::header::{CONTENT_RANGE, ETAG, IF_NONE_MATCH, RANGE};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::cache::{ModelCacheWorkspace, ModelIndexKind};
use crate::candidate_files::RecognizedArchitectures;
use crate::hugging_face::{require_requested_revision, revision_metadata_url};
use crate::identity::content_id;
use crate::inventory::{ManagedModelStore, build_model, now};
use crate::lineage::CatalogLineage;
use crate::search_shape::{CatalogFingerprints, RepositoryModel};

const MAX_HUB_METADATA_BYTES: usize = 16 * 1024 * 1024;
const MAX_HUB_SIBLINGS: usize = 100_000;
const MAX_HUB_SEARCH_RESULTS: u32 = 50;
const MAX_HUB_SEARCH_QUERY_BYTES: usize = 200;
const HUB_SEARCH_TTL: Duration = Duration::from_secs(60);
const MAX_HUB_SEARCH_CACHE_ENTRIES: usize = 256;
const MAX_MODEL_COMPONENTS: u32 = 256;
const MAX_COMPONENT_LOGICAL_BYTES: u64 = 1024 * 1024 * 1024 * 1024;
const HUB_REQUEST_ATTEMPTS: usize = 3;
const HUB_RETRY_BASE_DELAY: Duration = Duration::from_millis(250);
const SEARCH_SHAPE_CONCURRENCY: usize = 8;

/// Live Hub search and immutable resolution for the service's API. Catalog publication and
/// managed inventory do not depend on these queries.
pub struct HuggingFaceDiscovery {
    models: Arc<ManagedModelStore>,
    architectures: RecognizedArchitectures,
    lineage: Arc<CatalogLineage>,
    catalog: Arc<dyn CatalogFingerprints>,
    searches: tokio::sync::Mutex<BTreeMap<String, (Instant, HuggingFaceModelSearchResults)>>,
}

impl HuggingFaceDiscovery {
    #[must_use]
    pub fn new(
        models: Arc<ManagedModelStore>,
        architectures: RecognizedArchitectures,
        lineage: Arc<CatalogLineage>,
        catalog: Arc<dyn CatalogFingerprints>,
    ) -> Self {
        Self {
            models,
            architectures,
            lineage,
            catalog,
            searches: tokio::sync::Mutex::new(BTreeMap::new()),
        }
    }

    async fn search_uncached(
        &self,
        query: &str,
        request: &HuggingFaceModelSearchRequest,
    ) -> Result<HuggingFaceModelSearchResults, InventoryError> {
        let mut url = reqwest::Url::parse(self.models.client.endpoint())
            .map_err(|error| InventoryError::Internal(error.to_string()))?;
        url.path_segments_mut()
            .map_err(|()| InventoryError::Internal("invalid hub endpoint".to_owned()))?
            .pop_if_empty()
            .push("api")
            .push("models");
        url.query_pairs_mut()
            .append_pair("search", query)
            .append_pair("filter", "gguf")
            .append_pair("sort", "downloads")
            .append_pair("direction", "-1")
            .append_pair("limit", &request.limit.to_string());
        for field in [
            "sha",
            "lastModified",
            "downloads",
            "likes",
            "tags",
            "private",
            "gated",
            "gguf",
            "siblings",
        ] {
            url.query_pairs_mut().append_pair("expand", field);
        }
        let response = send_hub_get(&self.models.http, self.models.hub_token(), url.clone()).await?;
        if response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return Err(hub_rate_limited(hub_pause(&url).unwrap_or(DEFAULT_RATE_LIMIT_PAUSE)));
        }
        if !response.status().is_success() {
            return Err(InventoryError::Upstream(format!(
                "Hugging Face search returned HTTP {}",
                response.status()
            )));
        }
        let metadata: Vec<HubSearchModel> = serde_json::from_slice(
            &bounded_response_bytes(response, MAX_HUB_METADATA_BYTES).await?,
        )
        .map_err(|error| InventoryError::Upstream(error.to_string()))?;
        let catalog_repositories = self.lineage.repositories();
        let mut kept = Vec::new();
        for model in metadata {
            // A listing the search itself returned that names no GGUF: nothing in it can be added
            // (the Hub's gguf filter also matches a tag). No listing says nothing.
            if model
                .siblings
                .as_ref()
                .is_some_and(|siblings| !siblings.iter().any(|sibling| valid_gguf_path(Path::new(&sibling.rfilename))))
            {
                continue;
            }
            let Some(mut result) = model.into_contract() else {
                continue;
            };
            if catalog_repositories.contains(&result.repository.to_lowercase())
                || self
                    .architectures
                    .is_drafter_repository(result.architecture.as_deref(), &result.tags)
            {
                continue;
            }
            if let Some(architecture) = result.architecture.clone().filter(|architecture| {
                !self.architectures.admits_repository(Some(architecture))
            }) {
                result.unsupported = vec![UnsupportedReason::Known(
                    KnownUnsupportedReason::Architecture { architecture },
                )];
            }
            kept.push(result);
        }
        let catalog = if kept.iter().any(|result| result.unsupported.is_empty()) {
            self.catalog.catalog_fingerprints().await
        } else {
            Vec::new()
        };
        let catalog = &catalog;
        let mut results = futures_util::stream::iter(kept)
            .map(|mut result| async move {
                let model = !result.unsupported.is_empty()
                    || self.read_model_file(&mut result, catalog).await;
                model.then_some(result)
            })
            .buffered(SEARCH_SHAPE_CONCURRENCY)
            .filter_map(std::future::ready)
            .collect::<Vec<_>>()
            .await;
        results.sort_by_key(|result| {
            (
                !result.unsupported.is_empty(),
                !names_exactly(&result.repository, query),
                std::cmp::Reverse(result.downloads.unwrap_or(0)),
            )
        });
        Ok(HuggingFaceModelSearchResults { results })
    }

    /// What `result`'s model file says: its header's architecture, which is unsupported when the
    /// engine doesn't run it, and the first catalog model whose shape fingerprint it has. Names and
    /// declared lineage never decide. A repository whose headers were read and hold no model file
    /// (no architecture, a projector's or a drafter's, or a layer-split piece) is unreadable,
    /// whatever the Hub's repository-level architecture says; a read that failed or timed out
    /// (a gated repository's refused files among them) leaves the result as the Hub describes it.
    /// `false` for a repository whose files read are all drafters' (a prediction head alone, as an
    /// MTP block): drafters are offered only in drafter lists, never as results.
    async fn read_model_file(
        &self,
        result: &mut HuggingFaceModelSearchResult,
        catalog: &[(ModelId, ModelFingerprint)],
    ) -> bool {
        let hub_projector = result.architecture.as_deref()
            == Some(magnitude_artifacts::gguf::PROJECTOR_ARCHITECTURE);
        match crate::search_shape::repository_model(
            &self.models,
            &self.architectures,
            catalog,
            &result.repository,
            &result.commit,
        )
        .await
        {
            Some(RepositoryModel::Model {
                architecture,
                shape,
            }) => {
                result.architecture = Some(architecture.clone());
                if !self.architectures.targets.contains(&architecture) {
                    result.unsupported = vec![UnsupportedReason::Known(
                        KnownUnsupportedReason::Architecture { architecture },
                    )];
                    return true;
                }
                result.variant_of = shape.and_then(|shape| shape.variant_of(catalog));
            }
            Some(RepositoryModel::Unreadable) => {
                result.unsupported = vec![UnsupportedReason::Known(
                    KnownUnsupportedReason::Unreadable,
                )];
            }
            None if hub_projector => {
                result.unsupported = vec![UnsupportedReason::Known(
                    KnownUnsupportedReason::Architecture {
                        architecture: magnitude_artifacts::gguf::PROJECTOR_ARCHITECTURE.to_owned(),
                    },
                )];
            }
            Some(RepositoryModel::Drafter) => return false,
            None => {}
        }
        true
    }
}

/// Whether a repository is exactly the query: its `owner/repository`, or its name in the same words, ignoring case, separators
/// and a trailing `GGUF`, with or without an uploader's `Org_` prefix (`bartowski/Qwen_Qwen3.5-4B-GGUF`
/// is exactly "Qwen3.5 4B").
fn names_exactly(repository: &str, query: &str) -> bool {
    if repository.eq_ignore_ascii_case(query) {
        return true;
    }
    let words = |text: &str| {
        text.to_lowercase()
            .split(|character: char| !(character.is_alphanumeric() || character == '.'))
            .filter(|word| !word.is_empty())
            .map(str::to_owned)
            .collect::<Vec<_>>()
    };
    let query = words(query);
    let name = repository.rsplit('/').next().unwrap_or(repository);
    std::iter::once(name)
        .chain(name.split_once('_').map(|(_, rest)| rest))
        .any(|name| {
            let mut name = words(name);
            if name.last().is_some_and(|word| word == "gguf") {
                name.pop();
            }
            name == query
        })
}

impl HuggingFaceModelCatalog for HuggingFaceDiscovery {
    fn search(
        &self,
        request: HuggingFaceModelSearchRequest,
    ) -> BoxFuture<'_, Result<HuggingFaceModelSearchResults, InventoryError>> {
        Box::pin(async move {
            // A pasted Hugging Face link (`https://huggingface.co/owner/repo/…`, `hf.co/owner/repo`)
            // searches for its repository.
            let repository = crate::candidate_files::hugging_face_repository_from_url(&request.query);
            let query = repository.as_deref().unwrap_or(request.query.trim());
            if query.is_empty()
                || query.len() > MAX_HUB_SEARCH_QUERY_BYTES
                || request.limit == 0
                || request.limit > MAX_HUB_SEARCH_RESULTS
            {
                return Err(InventoryError::InvalidRequest(format!(
                    "Hugging Face search requires a non-empty query of at most {MAX_HUB_SEARCH_QUERY_BYTES} bytes and a limit between 1 and {MAX_HUB_SEARCH_RESULTS}"
                )));
            }
            let cache_key = format!("{}:{}", query.to_lowercase(), request.limit);
            {
                let mut searches = self.searches.lock().await;
                searches.retain(|_, (captured, _)| captured.elapsed() <= HUB_SEARCH_TTL);
                if let Some((_, result)) = searches.get(&cache_key) {
                    return Ok(result.clone());
                }
            }
            let result = self.search_uncached(query, &request).await?;
            let mut searches = self.searches.lock().await;
            searches.insert(cache_key, (Instant::now(), result.clone()));
            if searches.len() > MAX_HUB_SEARCH_CACHE_ENTRIES
                && let Some(oldest) = searches
                    .iter()
                    .min_by_key(|(_, (captured, _))| *captured)
                    .map(|(key, _)| key.clone())
            {
                searches.remove(&oldest);
            }
            Ok(result)
        })
    }

    fn resolve(
        &self,
        request: HuggingFaceRepositoryRequest,
    ) -> BoxFuture<'_, Result<HuggingFaceRepositorySnapshot, InventoryError>> {
        Box::pin(refresh_hugging_face_repository(&self.models, request))
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct CachedHuggingFaceRepositorySnapshot {
    captured_at: u64,
    #[serde(default)]
    etag: Option<String>,
    snapshot: HuggingFaceRepositorySnapshot,
}

pub async fn refresh_hugging_face_repository(
    models: &Arc<ManagedModelStore>,
    request: HuggingFaceRepositoryRequest,
) -> Result<HuggingFaceRepositorySnapshot, InventoryError> {
    validate_repository_request(&request)?;
    let cache_key = repository_cache_key(models.client.endpoint(), &request);
    let cached = models
        .cache
        .read_index::<CachedHuggingFaceRepositorySnapshot>(
            ModelIndexKind::HuggingFaceRepositorySnapshot,
            &cache_key,
        );
    refresh_hub_repository_snapshot(
        models,
        &models.http,
        models.client.endpoint(),
        request,
        cache_key,
        cached,
    )
    .await
}

fn validate_repository_request(
    request: &HuggingFaceRepositoryRequest,
) -> Result<(), InventoryError> {
    if !valid_repository(&request.repository) {
        return Err(InventoryError::InvalidRequest(
            "Hugging Face repository must use owner/repository form".to_owned(),
        ));
    }
    if !valid_hub_revision(&request.revision) {
        return Err(InventoryError::InvalidRequest(
            "Hugging Face revision contains unsupported characters".to_owned(),
        ));
    }
    Ok(())
}

fn repository_cache_key(endpoint: &str, request: &HuggingFaceRepositoryRequest) -> String {
    format!("{endpoint}:{}@{}", request.repository, request.revision)
}

pub struct PreparedPreview {
    pub model: ResolvedModel,
    pub repository: String,
    pub commit: String,
    pub components: Vec<ModelComponent>,
    pub headers: Vec<PreparedPreviewHeader>,
    pub artifact_fingerprint: String,
    _workspace: ModelCacheWorkspace,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreparedPreviewHeader {
    pub path: PathBuf,
    pub digest: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CachedArtifact {
    repository: String,
    commit: String,
    primary_gguf: PathBuf,
    components: Vec<CachedComponent>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CachedComponent {
    component: ModelComponent,
    header_digest: String,
    #[serde(skip)]
    acquired_header: Option<Vec<u8>>,
}

#[derive(Debug, Deserialize)]
struct HubModel {
    #[serde(default)]
    id: Option<String>,
    sha: Option<String>,
    #[serde(default, rename = "lastModified")]
    last_modified: Option<String>,
    #[serde(default)]
    downloads: Option<u64>,
    #[serde(default)]
    likes: Option<u64>,
    #[serde(default)]
    gated: Option<serde_json::Value>,
    #[serde(default)]
    private: Option<bool>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default, rename = "cardData")]
    card_data: Option<serde_json::Value>,
    #[serde(default)]
    siblings: Vec<HubSibling>,
}

#[derive(Debug, Deserialize)]
struct HubSearchModel {
    id: String,
    sha: Option<String>,
    #[serde(default, rename = "lastModified")]
    last_modified: Option<String>,
    #[serde(default)]
    downloads: Option<u64>,
    #[serde(default)]
    likes: Option<u64>,
    #[serde(default)]
    gated: Option<serde_json::Value>,
    #[serde(default)]
    private: Option<bool>,
    #[serde(default)]
    tags: Vec<String>,
    #[serde(default)]
    gguf: Option<serde_json::Value>,
    #[serde(default)]
    siblings: Option<Vec<HubSearchSibling>>,
}

#[derive(Debug, Deserialize)]
struct HubSearchSibling {
    rfilename: String,
}

impl HubSearchModel {
    fn into_contract(self) -> Option<HuggingFaceModelSearchResult> {
        let architecture = self
            .gguf
            .as_ref()
            .and_then(|gguf| gguf.get("architecture"))
            .and_then(serde_json::Value::as_str)
            .map(ToOwned::to_owned);
        Some(HuggingFaceModelSearchResult {
            repository: self.id,
            commit: immutable_commit(self.sha?)?,
            last_modified: self.last_modified,
            downloads: self.downloads,
            likes: self.likes,
            gated: hub_gated(self.gated.as_ref()),
            private: self.private.unwrap_or(false),
            tags: self.tags,
            architecture,
            variant_of: None,
            unsupported: Vec::new(),
        })
    }
}

impl HubModel {
    fn into_snapshot(
        self,
        requested_repository: String,
        requested_revision: &str,
    ) -> Result<HuggingFaceRepositorySnapshot, InventoryError> {
        if self.siblings.len() > MAX_HUB_SIBLINGS {
            return Err(InventoryError::Integrity(
                "Hugging Face metadata contains too many files".to_owned(),
            ));
        }
        let commit = self.sha.and_then(immutable_commit).ok_or_else(|| {
            InventoryError::Integrity(
                "Hugging Face repository did not resolve to an immutable commit".to_owned(),
            )
        })?;
        validate_snapshot_revision(requested_revision, &commit)
            .map_err(InventoryError::Integrity)?;
        let mut gguf_files = self
            .siblings
            .iter()
            .filter(|sibling| valid_gguf_path(Path::new(&sibling.rfilename)))
            .map(|sibling| {
                let (size_bytes, content) = sibling_identity(sibling)?;
                Ok(HuggingFaceRepositoryFile {
                    path: PathBuf::from(&sibling.rfilename),
                    size_bytes,
                    content,
                })
            })
            .collect::<Result<Vec<_>, InventoryError>>()?;
        gguf_files.sort_by(|left, right| left.path.cmp(&right.path));
        if gguf_files.is_empty() {
            return Err(InventoryError::Unsupported(format!(
                "{} does not contain GGUF artifacts",
                requested_repository
            )));
        }
        let card = self.card_data.as_ref();
        let license = card
            .and_then(|value| value.get("license"))
            .and_then(serde_json::Value::as_str)
            .map(ToOwned::to_owned)
            .or_else(|| tag_value(&self.tags, "license:"));
        let license_url = card
            .and_then(|value| value.get("license_link"))
            .and_then(serde_json::Value::as_str)
            .map(ToOwned::to_owned);
        let mut base_models = card
            .and_then(|value| value.get("base_model"))
            .map(json_strings)
            .unwrap_or_default();
        if base_models.is_empty() {
            base_models = self
                .tags
                .iter()
                .filter_map(|tag| {
                    tag.strip_prefix("base_model:quantized:")
                        .or_else(|| tag.strip_prefix("base_model:"))
                        .map(ToOwned::to_owned)
                })
                .collect();
        }
        base_models.sort();
        base_models.dedup();
        Ok(HuggingFaceRepositorySnapshot {
            repository: self.id.unwrap_or(requested_repository),
            commit,
            last_modified: self.last_modified,
            downloads: self.downloads,
            likes: self.likes,
            gated: hub_gated(self.gated.as_ref()),
            private: self.private.unwrap_or(false),
            license,
            license_url,
            base_models,
            tags: self.tags,
            gguf_files,
        })
    }
}

#[derive(Debug, Deserialize)]
struct HubSibling {
    rfilename: String,
    size: Option<u64>,
    #[serde(rename = "blobId")]
    blob_id: Option<String>,
    lfs: Option<HubLfs>,
}

#[derive(Debug, Deserialize)]
struct HubLfs {
    sha256: String,
    size: u64,
}

enum HubModelFetch {
    NotModified,
    Modified {
        model: HubModel,
        etag: Option<String>,
    },
}

async fn refresh_hub_repository_snapshot(
    models: &ManagedModelStore,
    http: &reqwest::Client,
    endpoint: &str,
    request: HuggingFaceRepositoryRequest,
    cache_key: String,
    cached: Option<CachedHuggingFaceRepositorySnapshot>,
) -> Result<HuggingFaceRepositorySnapshot, InventoryError> {
    let fetched = fetch_hub_model(
        http,
        models.hub_token(),
        endpoint,
        &request.repository,
        request.revision.as_str(),
        cached.as_ref().and_then(|cached| cached.etag.as_deref()),
    )
    .await?;
    let (snapshot, etag) = match fetched {
        HubModelFetch::NotModified => {
            let cached = cached.ok_or_else(|| {
                InventoryError::Upstream(
                    "Hugging Face returned not modified without cached metadata".to_owned(),
                )
            })?;
            (cached.snapshot, cached.etag)
        }
        HubModelFetch::Modified { model, etag } => (
            model.into_snapshot(request.repository, request.revision.as_str())?,
            etag,
        ),
    };
    models.cache.write_index(
        ModelIndexKind::HuggingFaceRepositorySnapshot,
        &cache_key,
        &CachedHuggingFaceRepositorySnapshot {
            captured_at: now(),
            etag,
            snapshot: snapshot.clone(),
        },
    );
    Ok(snapshot)
}

async fn fetch_hub_model(
    http: &reqwest::Client,
    token: Option<&str>,
    endpoint: &str,
    repository: &str,
    revision: &str,
    etag: Option<&str>,
) -> Result<HubModelFetch, InventoryError> {
    let url =
        revision_metadata_url(endpoint, repository, revision).map_err(InventoryError::Internal)?;
    let response = send_hub_request(|| {
        let mut request = http.get(url.clone()).query(&[("blobs", "true")]);
        if let Some(token) = &token {
            request = request.bearer_auth(token);
        }
        if let Some(etag) = etag {
            request = request.header(IF_NONE_MATCH, etag);
        }
        request
    })
    .await?;
    if response.status() == reqwest::StatusCode::NOT_MODIFIED {
        return Ok(HubModelFetch::NotModified);
    }
    if matches!(
        response.status(),
        reqwest::StatusCode::UNAUTHORIZED | reqwest::StatusCode::NOT_FOUND
    ) {
        return Err(hub_repository_not_found());
    }
    if !response.status().is_success() {
        return Err(InventoryError::Upstream(format!(
            "Hugging Face metadata returned HTTP {}",
            response.status()
        )));
    }
    let etag = response
        .headers()
        .get(ETAG)
        .and_then(|value| value.to_str().ok())
        .map(ToOwned::to_owned);
    let model =
        serde_json::from_slice(&bounded_response_bytes(response, MAX_HUB_METADATA_BYTES).await?)
            .map_err(|error| InventoryError::Upstream(error.to_string()))?;
    Ok(HubModelFetch::Modified { model, etag })
}

fn retryable_hub_status(status: reqwest::StatusCode) -> bool {
    status.is_server_error()
}

/// The longest pause a Hub `Retry-After` imposes, and the pause when it names none.
const MAX_RATE_LIMIT_PAUSE: Duration = Duration::from_secs(60);
const DEFAULT_RATE_LIMIT_PAUSE: Duration = Duration::from_secs(30);

/// Until when each host (`host:port`) asked this process to stop sending requests (HTTP 429). A
/// host's rate limit covers every request to it, so no request goes out until it passes.
static HUB_PAUSES: std::sync::LazyLock<std::sync::Mutex<BTreeMap<String, Instant>>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(BTreeMap::new()));

fn host_key(url: &reqwest::Url) -> String {
    format!(
        "{}:{}",
        url.host_str().unwrap_or_default(),
        url.port_or_known_default().unwrap_or_default()
    )
}

/// How long `url`'s host has asked this process to wait, if it has.
pub(crate) fn hub_pause(url: &reqwest::Url) -> Option<Duration> {
    let pauses = HUB_PAUSES.lock().ok()?;
    pauses
        .get(&host_key(url))
        .and_then(|until| until.checked_duration_since(Instant::now()))
        .filter(|left| !left.is_zero())
}

fn note_rate_limit(url: &reqwest::Url, response: &reqwest::Response) {
    let pause = response
        .headers()
        .get(reqwest::header::RETRY_AFTER)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.trim().parse::<u64>().ok())
        .map_or(DEFAULT_RATE_LIMIT_PAUSE, Duration::from_secs)
        .min(MAX_RATE_LIMIT_PAUSE);
    if let Ok(mut pauses) = HUB_PAUSES.lock() {
        pauses.insert(host_key(url), Instant::now() + pause);
    }
}

pub(crate) fn hub_rate_limited(left: Duration) -> InventoryError {
    InventoryError::ModelOperation {
        code: "hugging_face_rate_limited".to_owned(),
        message: format!(
            "Hugging Face is limiting requests from this computer. Try again in {} seconds, or set HF_TOKEN for a higher limit.",
            left.as_secs().max(1)
        ),
        retryable: true,
    }
}

/// One Hub request, retried on server failures. A rate limit (HTTP 429) is never retried: it
/// pauses every request to that host for its `Retry-After` (at most a minute), and a request
/// during a pause fails at once without being sent.
pub(crate) async fn send_hub_request(
    mut request: impl FnMut() -> reqwest::RequestBuilder,
) -> Result<reqwest::Response, InventoryError> {
    let mut last_failure = None;
    for attempt in 0..HUB_REQUEST_ATTEMPTS {
        let (client, built) = request().build_split();
        let built = built.map_err(|error| InventoryError::Internal(error.to_string()))?;
        let url = built.url().clone();
        if let Some(left) = hub_pause(&url) {
            return Err(hub_rate_limited(left));
        }
        match client.execute(built).await {
            Ok(response) if response.status() == reqwest::StatusCode::TOO_MANY_REQUESTS => {
                note_rate_limit(&url, &response);
                return Ok(response);
            }
            Ok(response)
                if retryable_hub_status(response.status())
                    && attempt + 1 < HUB_REQUEST_ATTEMPTS =>
            {
                last_failure = Some(format!("HTTP {}", response.status()));
            }
            Ok(response) => return Ok(response),
            Err(error) if attempt + 1 < HUB_REQUEST_ATTEMPTS => {
                last_failure = Some(error.to_string());
            }
            Err(error) => return Err(InventoryError::Upstream(error.to_string())),
        }
        let multiplier = u32::try_from(attempt + 1).unwrap_or(u32::MAX);
        tokio::time::sleep(HUB_RETRY_BASE_DELAY.saturating_mul(multiplier)).await;
    }
    Err(InventoryError::Upstream(last_failure.unwrap_or_else(
        || "Hugging Face request failed".to_owned(),
    )))
}

fn immutable_commit(value: String) -> Option<String> {
    ((40..=64).contains(&value.len())
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)))
    .then_some(value)
}

fn validate_snapshot_revision(requested: &str, commit: &str) -> Result<(), String> {
    if immutable_commit(requested.to_owned()).is_some() {
        require_requested_revision(requested, Some(commit))?;
    }
    Ok(())
}

fn hub_gated(value: Option<&serde_json::Value>) -> bool {
    value.is_some_and(|value| {
        !matches!(
            value,
            serde_json::Value::Bool(false) | serde_json::Value::Null
        )
    })
}

fn tag_value(tags: &[String], prefix: &str) -> Option<String> {
    tags.iter()
        .find_map(|tag| tag.strip_prefix(prefix).map(ToOwned::to_owned))
}

fn json_strings(value: &serde_json::Value) -> Vec<String> {
    match value {
        serde_json::Value::String(value) => vec![value.clone()],
        serde_json::Value::Array(values) => values
            .iter()
            .filter_map(serde_json::Value::as_str)
            .map(ToOwned::to_owned)
            .collect(),
        _ => Vec::new(),
    }
}

struct SelectedComponent<'a> {
    sibling: &'a HubSibling,
    role: ComponentRole,
    shard_index: Option<u32>,
    relationship: Option<ComponentRelationship>,
}

impl ManagedModelStore {
    fn cached_preview_artifact(&self, source: &ModelPreviewSource) -> Option<CachedArtifact> {
        let evidence = serde_json::to_string(source).ok()?;
        self.cache
            .read_index::<CachedArtifact>(ModelIndexKind::Artifact, &evidence)
            .filter(|artifact| artifact_matches_source(artifact, source))
    }

    pub async fn prepare_preview(
        &self,
        source: &ModelPreviewSource,
    ) -> Result<PreparedPreview, InventoryError> {
        validate_source(source)?;
        let evidence = serde_json::to_string(source)
            .map_err(|error| InventoryError::Internal(error.to_string()))?;
        let cached = self.cached_preview_artifact(source).filter(|artifact| {
            artifact
                .components
                .iter()
                .all(|component| self.cache.has_header(&component.header_digest))
        });
        let artifact = match cached {
            Some(artifact) => artifact,
            None => {
                let artifact = self.acquire_preview_artifact(source).await?;
                self.cache
                    .write_index(ModelIndexKind::Artifact, &evidence, &artifact);
                artifact
            }
        };
        self.materialize_preview(artifact)
    }

    pub async fn prepare_preview_from_repository_snapshot(
        &self,
        source: &ModelPreviewSource,
        snapshot: &HuggingFaceRepositorySnapshot,
    ) -> Result<PreparedPreview, InventoryError> {
        validate_source(source)?;
        if snapshot.repository != source.repository || snapshot.commit != source.revision {
            return Err(InventoryError::Integrity(
                "repository snapshot does not match the preview source".to_owned(),
            ));
        }
        let evidence = serde_json::to_string(source)
            .map_err(|error| InventoryError::Internal(error.to_string()))?;
        let cached = self.cached_preview_artifact(source).filter(|artifact| {
            artifact
                .components
                .iter()
                .all(|component| self.cache.has_header(&component.header_digest))
        });
        let artifact = match cached {
            Some(artifact) => artifact,
            None => {
                let artifact = self
                    .acquire_preview_artifact_from_repository_snapshot(source, snapshot)
                    .await?;
                self.cache
                    .write_index(ModelIndexKind::Artifact, &evidence, &artifact);
                artifact
            }
        };
        let store = self.clone();
        tokio::task::spawn_blocking(move || store.materialize_preview(artifact))
            .await
            .map_err(|error| InventoryError::Internal(error.to_string()))?
    }

    async fn acquire_preview_artifact(
        &self,
        source: &ModelPreviewSource,
    ) -> Result<CachedArtifact, InventoryError> {
        let metadata = fetch_hub_model(
            &self.http,
            self.hub_token(),
            self.client.endpoint(),
            &source.repository,
            &source.revision,
            None,
        )
        .await?;
        let HubModelFetch::Modified {
            model: metadata, ..
        } = metadata
        else {
            return Err(InventoryError::Upstream(
                "Hugging Face returned not modified without cached metadata".to_owned(),
            ));
        };
        if metadata.siblings.len() > MAX_HUB_SIBLINGS {
            return Err(InventoryError::Integrity(
                "Hugging Face metadata contains too many files".to_owned(),
            ));
        }
        let commit = metadata
            .sha
            .filter(|commit| commit == &source.revision)
            .ok_or_else(|| {
                InventoryError::Integrity(
                    "preview revision did not resolve to the requested immutable commit".to_owned(),
                )
            })?;
        let selected = select_artifact_components(&metadata.siblings, source)?;
        let mut components = Vec::with_capacity(selected.len());
        for selected in selected {
            let sibling = selected.sibling;
            let (size, content) = sibling_identity(sibling)?;
            let component = ModelComponent {
                path: PathBuf::from(&sibling.rfilename),
                role: selected.role,
                size_bytes: size,
                content,
                shard_index: selected.shard_index,
                relationship: selected.relationship,
            };
            let header = fetch_header(
                &self.http,
                self.hub_token(),
                self.client.endpoint(),
                &source.repository,
                &commit,
                &component.path,
                size,
            )
            .await?;
            let header_digest = self.write_header(&header);
            components.push(CachedComponent {
                component,
                header_digest,
                acquired_header: Some(header),
            });
        }
        Ok(CachedArtifact {
            repository: source.repository.clone(),
            commit,
            primary_gguf: source.primary_gguf.clone(),
            components,
        })
    }

    async fn acquire_preview_artifact_from_repository_snapshot(
        &self,
        source: &ModelPreviewSource,
        snapshot: &HuggingFaceRepositorySnapshot,
    ) -> Result<CachedArtifact, InventoryError> {
        let selected = select_repository_snapshot_components(&snapshot.gguf_files, source)?;
        let fetched = futures_util::stream::iter(selected.into_iter().enumerate())
            .map(|(index, component)| async move {
                let header = self
                    .hugging_face_gguf_header(
                        &source.repository,
                        &snapshot.commit,
                        &HuggingFaceRepositoryFile {
                            path: component.path.clone(),
                            size_bytes: component.size_bytes,
                            content: component.content.clone(),
                        },
                        &[],
                    )
                    .await?;
                Ok::<_, InventoryError>((
                    index,
                    CachedComponent {
                        component,
                        header_digest: header.digest,
                        acquired_header: Some(header.bytes),
                    },
                ))
            })
            .buffer_unordered(6)
            .collect::<Vec<_>>()
            .await;
        let mut components = fetched
            .into_iter()
            .collect::<Result<Vec<_>, InventoryError>>()?;
        components.sort_by_key(|(index, _)| *index);
        Ok(CachedArtifact {
            repository: source.repository.clone(),
            commit: snapshot.commit.clone(),
            primary_gguf: source.primary_gguf.clone(),
            components: components
                .into_iter()
                .map(|(_, component)| component)
                .collect(),
        })
    }

    fn materialize_preview(
        &self,
        artifact: CachedArtifact,
    ) -> Result<PreparedPreview, InventoryError> {
        let workspace = self
            .cache
            .workspace()
            .map_err(|error| InventoryError::Io(error.to_string()))?;
        for cached in &artifact.components {
            let bytes = self
                .cache
                .read_header(&cached.header_digest)
                .or_else(|| cached.acquired_header.clone())
                .ok_or_else(|| InventoryError::Internal("preview header cache miss".to_owned()))?;
            let path = workspace.path().join(&cached.component.path);
            let parent = path.parent().ok_or_else(|| {
                InventoryError::InvalidRequest("preview component has no parent".to_owned())
            })?;
            fs::create_dir_all(parent).map_err(|error| InventoryError::Io(error.to_string()))?;
            let mut file = OpenOptions::new()
                .create_new(true)
                .write(true)
                .open(&path)
                .map_err(|error| InventoryError::Io(error.to_string()))?;
            file.write_all(&bytes)
                .and_then(|()| {
                    magnitude_service_utils::sparse_file::set_sparse_len(
                        &file,
                        cached.component.size_bytes,
                    )
                })
                .map_err(|error| InventoryError::Io(error.to_string()))?;
        }
        let components = artifact
            .components
            .iter()
            .map(|component| component.component.clone())
            .collect::<Vec<_>>();
        let content = content_id(&components);
        let source_fingerprint =
            format!("{}:{}:{}", artifact.repository, artifact.commit, content.0);
        let id = InventoryEntryId(format!(
            "mdl_{:x}",
            Sha256::digest(source_fingerprint.as_bytes())
        ));
        let timestamp = now();
        let location = ModelLocation::Directory {
            source_id: "remote_preview".to_owned(),
            root: workspace.path().to_path_buf(),
            components: components.clone(),
            total_bytes: components
                .iter()
                .map(|component| component.size_bytes)
                .sum(),
            integrity: Integrity::Verified {
                method: "immutable_huggingface_revision_and_header_digest".to_owned(),
            },
        };
        let source = ModelSource::HuggingFace {
            repository: artifact.repository.clone(),
            requested_revision: artifact.commit.clone(),
            commit: artifact.commit.clone(),
            metadata: None,
        };
        let primary = workspace.path().join(&artifact.primary_gguf);
        let model = build_model(
            id,
            content,
            timestamp,
            timestamp,
            source,
            location,
            &primary,
            false,
            &self.cache,
        )?;
        let resolved = ResolvedModel {
            model,
            components: components
                .iter()
                .map(|component| ResolvedComponent {
                    path: workspace.path().join(&component.path),
                    role: component.role.clone(),
                    shard_index: component.shard_index,
                    relationship: component.relationship.clone(),
                })
                .collect(),
        };
        Ok(PreparedPreview {
            model: resolved,
            repository: artifact.repository,
            commit: artifact.commit,
            components,
            headers: artifact
                .components
                .iter()
                .map(|component| PreparedPreviewHeader {
                    path: component.component.path.clone(),
                    digest: component.header_digest.clone(),
                })
                .collect(),
            artifact_fingerprint: source_fingerprint,
            _workspace: workspace,
        })
    }
}

fn validate_source(source: &ModelPreviewSource) -> Result<(), InventoryError> {
    let valid_repository = valid_repository(&source.repository);
    let valid_revision = (40..=64).contains(&source.revision.len())
        && source
            .revision
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
    let valid_path = valid_gguf_path(&source.primary_gguf);
    let mut selected_paths = std::collections::BTreeSet::from([&source.primary_gguf]);
    let valid_additional = source.additional_components.len() < MAX_MODEL_COMPONENTS as usize
        && source.additional_components.iter().all(|component| {
            valid_gguf_path(&component.path)
                && !is_split_path(&component.path)
                && selected_paths.insert(&component.path)
        });
    if valid_repository && valid_revision && valid_path && valid_additional {
        Ok(())
    } else {
        Err(InventoryError::InvalidRequest(
            "preview requires owner/repository, an immutable hexadecimal commit, and unique relative GGUF components with supported execution roles"
                .to_owned(),
        ))
    }
}

fn valid_repository(repository: &str) -> bool {
    let mut parts = repository.split('/');
    parts.next().is_some_and(valid_repository_part)
        && parts.next().is_some_and(valid_repository_part)
        && parts.next().is_none()
}

fn valid_repository_part(part: &str) -> bool {
    !part.is_empty()
        && part != "."
        && part != ".."
        && part
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
}

fn valid_hub_revision(revision: &str) -> bool {
    !revision.is_empty()
        && revision.len() <= 200
        && revision != "."
        && revision != ".."
        && revision
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'/'))
        && revision
            .split('/')
            .all(|part| !part.is_empty() && part != "." && part != "..")
}

fn valid_gguf_path(path: &Path) -> bool {
    !path.as_os_str().is_empty()
        && path.is_relative()
        && path
            .components()
            .all(|component| matches!(component, std::path::Component::Normal(_)))
        && path
            .extension()
            .and_then(|value| value.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
}

fn artifact_matches_source(artifact: &CachedArtifact, source: &ModelPreviewSource) -> bool {
    if artifact.repository != source.repository
        || artifact.commit != source.revision
        || artifact.primary_gguf != source.primary_gguf
    {
        return false;
    }
    let primary_paths = if let Some((prefix, total)) = split_parts(&source.primary_gguf) {
        let parent = source
            .primary_gguf
            .parent()
            .unwrap_or_else(|| Path::new(""));
        (1..=total)
            .map(|index| parent.join(format!("{prefix}-{index:05}-of-{total:05}.gguf")))
            .collect::<Vec<_>>()
    } else {
        vec![source.primary_gguf.clone()]
    };
    if artifact.components.len() != primary_paths.len() + source.additional_components.len() {
        return false;
    }
    let mut paths = std::collections::BTreeSet::new();
    let structurally_valid = artifact.components.iter().all(|cached| {
        cached.component.size_bytes > 0
            && valid_hex_digest(&cached.header_digest, 64, 64)
            && valid_published_identity(&cached.component.content)
            && paths.insert(&cached.component.path)
    });
    let primary_valid = primary_paths.iter().enumerate().all(|(index, path)| {
        artifact.components.iter().any(|cached| {
            cached.component.path == *path
                && if primary_paths.len() == 1 {
                    cached.component.role == ComponentRole::Weights
                        && cached.component.shard_index.is_none()
                } else {
                    cached.component.role == ComponentRole::Shard
                        && cached.component.shard_index == Some(index as u32 + 1)
                }
        })
    });
    let additional_valid = source.additional_components.iter().all(|expected| {
        artifact.components.iter().any(|cached| {
            cached.component.path == expected.path
                && cached.component.role == expected.role.component_role()
                && cached.component.shard_index.is_none()
                && cached.component.relationship
                    == Some(component_relationship(expected, &source.primary_gguf))
        })
    });
    structurally_valid && primary_valid && additional_valid
}

fn is_split_path(path: &Path) -> bool {
    split_parts(path).is_some()
}

fn split_parts(path: &Path) -> Option<(String, u32)> {
    let name = path.file_name()?.to_str()?;
    let stem = name.strip_suffix(".gguf")?;
    let (left, total) = stem.rsplit_once("-of-")?;
    let (prefix, index) = left.rsplit_once('-')?;
    if index != "00001" || total.len() != 5 {
        return None;
    }
    let total = total.parse().ok()?;
    (1..=MAX_MODEL_COMPONENTS)
        .contains(&total)
        .then(|| (prefix.to_owned(), total))
}

fn select_components<'a>(
    siblings: &'a [HubSibling],
    primary: &Path,
) -> Result<Vec<&'a HubSibling>, InventoryError> {
    if let Some((prefix, total)) = split_parts(primary) {
        let parent = primary.parent().unwrap_or_else(|| Path::new(""));
        return (1..=total)
            .map(|index| {
                let path = parent.join(format!("{prefix}-{index:05}-of-{total:05}.gguf"));
                siblings
                    .iter()
                    .find(|sibling| Path::new(&sibling.rfilename) == path)
                    .ok_or_else(|| {
                        InventoryError::Integrity(format!(
                            "missing preview shard {}",
                            path.display()
                        ))
                    })
            })
            .collect();
    }
    siblings
        .iter()
        .find(|sibling| Path::new(&sibling.rfilename) == primary)
        .map(|sibling| vec![sibling])
        .ok_or_else(|| InventoryError::Integrity("preview GGUF does not exist".to_owned()))
}

fn select_artifact_components<'a>(
    siblings: &'a [HubSibling],
    source: &ModelPreviewSource,
) -> Result<Vec<SelectedComponent<'a>>, InventoryError> {
    let split = is_split_path(&source.primary_gguf);
    let mut selected = select_components(siblings, &source.primary_gguf)?
        .into_iter()
        .enumerate()
        .map(|(index, sibling)| SelectedComponent {
            sibling,
            role: if split {
                ComponentRole::Shard
            } else {
                ComponentRole::Weights
            },
            shard_index: split.then_some(index as u32 + 1),
            relationship: None,
        })
        .collect::<Vec<_>>();
    for additional in &source.additional_components {
        let sibling = siblings
            .iter()
            .find(|sibling| Path::new(&sibling.rfilename) == additional.path)
            .ok_or_else(|| {
                InventoryError::Integrity(format!(
                    "preview component {} does not exist",
                    additional.path.display()
                ))
            })?;
        selected.push(SelectedComponent {
            sibling,
            role: additional.role.component_role(),
            shard_index: None,
            relationship: Some(component_relationship(additional, &source.primary_gguf)),
        });
    }
    if selected.len() > MAX_MODEL_COMPONENTS as usize {
        return Err(InventoryError::Integrity(
            "preview artifact contains too many selected components".to_owned(),
        ));
    }
    Ok(selected)
}

fn select_repository_snapshot_components(
    files: &[HuggingFaceRepositoryFile],
    source: &ModelPreviewSource,
) -> Result<Vec<ModelComponent>, InventoryError> {
    let primary_paths = if let Some((prefix, total)) = split_parts(&source.primary_gguf) {
        let parent = source
            .primary_gguf
            .parent()
            .unwrap_or_else(|| Path::new(""));
        (1..=total)
            .map(|index| parent.join(format!("{prefix}-{index:05}-of-{total:05}.gguf")))
            .collect::<Vec<_>>()
    } else {
        vec![source.primary_gguf.clone()]
    };
    let mut selected = primary_paths
        .iter()
        .enumerate()
        .map(|(index, path)| {
            let file = files
                .iter()
                .find(|file| file.path == *path)
                .ok_or_else(|| {
                    InventoryError::Integrity(format!(
                        "preview GGUF {} does not exist in the repository snapshot",
                        path.display()
                    ))
                })?;
            Ok(ModelComponent {
                path: file.path.clone(),
                role: if primary_paths.len() == 1 {
                    ComponentRole::Weights
                } else {
                    ComponentRole::Shard
                },
                size_bytes: file.size_bytes,
                content: file.content.clone(),
                shard_index: (primary_paths.len() > 1).then_some(index as u32 + 1),
                relationship: None,
            })
        })
        .collect::<Result<Vec<_>, InventoryError>>()?;
    for additional in &source.additional_components {
        let file = files
            .iter()
            .find(|file| file.path == additional.path)
            .ok_or_else(|| {
                InventoryError::Integrity(format!(
                    "preview component {} does not exist in the repository snapshot",
                    additional.path.display()
                ))
            })?;
        selected.push(ModelComponent {
            path: file.path.clone(),
            role: additional.role.component_role(),
            size_bytes: file.size_bytes,
            content: file.content.clone(),
            shard_index: None,
            relationship: Some(component_relationship(additional, &source.primary_gguf)),
        });
    }
    if selected.len() > MAX_MODEL_COMPONENTS as usize {
        return Err(InventoryError::Integrity(
            "preview artifact contains too many selected components".to_owned(),
        ));
    }
    Ok(selected)
}

fn component_relationship(
    component: &magnitude_service_contracts::ModelPreviewComponentSource,
    primary: &Path,
) -> ComponentRelationship {
    match &component.role {
        magnitude_service_contracts::ModelPreviewComponentRole::Projector => {
            ComponentRelationship::ProjectorFor {
                projector: component.path.clone(),
                model: primary.to_path_buf(),
            }
        }
        magnitude_service_contracts::ModelPreviewComponentRole::Draft { method } => {
            ComponentRelationship::DraftFor {
                draft: component.path.clone(),
                model: primary.to_path_buf(),
                method: method.clone(),
            }
        }
        magnitude_service_contracts::ModelPreviewComponentRole::Mtp => {
            ComponentRelationship::MtpFor {
                mtp: component.path.clone(),
                model: primary.to_path_buf(),
            }
        }
    }
}

fn sibling_identity(sibling: &HubSibling) -> Result<(u64, ContentIdentity), InventoryError> {
    if let Some(lfs) = sibling.lfs.as_ref() {
        validate_component_size(&sibling.rfilename, lfs.size)?;
        if sibling.size.is_some_and(|size| size != lfs.size)
            || !valid_hex_digest(&lfs.sha256, 64, 64)
        {
            return Err(InventoryError::Integrity(format!(
                "{} has inconsistent LFS identity metadata",
                sibling.rfilename
            )));
        }
        return Ok((
            lfs.size,
            ContentIdentity::Sha256 {
                value: lfs.sha256.to_ascii_lowercase(),
            },
        ));
    }
    let size = sibling.size.ok_or_else(|| {
        InventoryError::Integrity(format!("{} has no published size", sibling.rfilename))
    })?;
    validate_component_size(&sibling.rfilename, size)?;
    let oid = sibling.blob_id.clone().ok_or_else(|| {
        InventoryError::Integrity(format!("{} has no content identity", sibling.rfilename))
    })?;
    if !valid_hex_digest(&oid, 40, 64) {
        return Err(InventoryError::Integrity(format!(
            "{} has an invalid Git object identity",
            sibling.rfilename
        )));
    }
    Ok((size, ContentIdentity::GitOid { value: oid }))
}

fn valid_hex_digest(value: &str, minimum: usize, maximum: usize) -> bool {
    (minimum..=maximum).contains(&value.len())
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_published_identity(identity: &ContentIdentity) -> bool {
    match identity {
        ContentIdentity::Sha256 { value } => valid_hex_digest(value, 64, 64),
        ContentIdentity::GitOid { value } => valid_hex_digest(value, 40, 64),
        _ => false,
    }
}

fn validate_component_size(path: &str, size: u64) -> Result<(), InventoryError> {
    if size == 0 || size > MAX_COMPONENT_LOGICAL_BYTES {
        return Err(InventoryError::Integrity(format!(
            "{path} has an invalid logical size"
        )));
    }
    Ok(())
}

pub(crate) async fn bounded_response_bytes(
    mut response: reqwest::Response,
    maximum_bytes: usize,
) -> Result<Vec<u8>, InventoryError> {
    if response
        .content_length()
        .is_some_and(|length| length > maximum_bytes as u64)
    {
        return Err(InventoryError::Upstream(
            "upstream metadata response exceeds its size bound".to_owned(),
        ));
    }
    let mut bytes = Vec::with_capacity(
        response
            .content_length()
            .and_then(|length| usize::try_from(length).ok())
            .unwrap_or(0),
    );
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|error| InventoryError::Upstream(error.to_string()))?
    {
        if bytes.len().saturating_add(chunk.len()) > maximum_bytes {
            return Err(InventoryError::Upstream(
                "upstream metadata response exceeds its size bound".to_owned(),
            ));
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}

/// The Hub URL of one file of a repository at a commit.
pub(crate) fn resolve_url(
    endpoint: &str,
    repository: &str,
    commit: &str,
    path: &Path,
) -> Result<reqwest::Url, InventoryError> {
    let mut url = reqwest::Url::parse(endpoint)
        .map_err(|error| InventoryError::Internal(error.to_string()))?;
    {
        let mut segments = url
            .path_segments_mut()
            .map_err(|()| InventoryError::Internal("invalid hub endpoint".to_owned()))?;
        segments.pop_if_empty();
        for segment in repository.split('/') {
            segments.push(segment);
        }
        segments.push("resolve").push(commit);
        for component in path.components() {
            if let std::path::Component::Normal(segment) = component {
                segments.push(&segment.to_string_lossy());
            }
        }
    }
    Ok(url)
}

/// One remote file read by exact range requests. After the first request it reads from the
/// location the Hub redirected to, without the Hub credential, so later ranges skip the redirect.
#[derive(Clone)]
pub(crate) struct RemoteRanges<'a> {
    http: &'a reqwest::Client,
    token: Option<&'a str>,
    url: reqwest::Url,
    hub_host: Option<String>,
    logical_size: u64,
}

impl<'a> RemoteRanges<'a> {
    pub(crate) fn new(
        http: &'a reqwest::Client,
        token: Option<&'a str>,
        endpoint: &str,
        repository: &str,
        commit: &str,
        path: &Path,
        logical_size: u64,
    ) -> Result<Self, InventoryError> {
        let url = resolve_url(endpoint, repository, commit, path)?;
        Ok(Self {
            http,
            token,
            hub_host: url.host_str().map(ToOwned::to_owned),
            url,
            logical_size,
        })
    }

    pub(crate) fn logical_length(&self) -> usize {
        usize::try_from(self.logical_size).unwrap_or(usize::MAX)
    }

    /// Exactly the bytes `start..end` of the file.
    pub(crate) async fn read(
        &mut self,
        start: usize,
        end: usize,
    ) -> Result<Vec<u8>, InventoryError> {
        let range_length = end.saturating_sub(start);
        let expected_start = u64::try_from(start)
            .map_err(|_| InventoryError::Internal("range length overflow".to_owned()))?;
        let expected_length = u64::try_from(range_length)
            .map_err(|_| InventoryError::Internal("range length overflow".to_owned()))?;
        let expected_end = u64::try_from(end)
            .map_err(|_| InventoryError::Internal("range length overflow".to_owned()))?;
        if range_length == 0 || expected_end > self.logical_size {
            return Err(InventoryError::Internal("invalid header range".to_owned()));
        }
        let range = format!("bytes={start}-{}", end - 1);
        let token = self
            .token
            .filter(|_| self.url.host_str() == self.hub_host.as_deref());
        let url = self.url.clone();
        let mut response = send_hub_request(|| {
            let mut request = self.http.get(url.clone()).header(RANGE, &range);
            if let Some(token) = &token {
                request = request.bearer_auth(token);
            }
            request
        })
        .await?;
        if matches!(
            response.status(),
            reqwest::StatusCode::UNAUTHORIZED | reqwest::StatusCode::FORBIDDEN
        ) {
            return Err(hub_access_required(token.is_some()));
        }
        if !response.status().is_success() {
            return Err(InventoryError::Upstream(format!(
                "GGUF header request returned HTTP {}",
                response.status()
            )));
        }
        let complete_response = expected_start == 0
            && expected_end == self.logical_size
            && response.status().as_u16() == 200;
        let valid_partial = response.status().as_u16() == 206
            && response
                .headers()
                .get(CONTENT_RANGE)
                .and_then(|value| value.to_str().ok())
                .is_some_and(|value| {
                    valid_content_range(value, expected_start, expected_end, self.logical_size)
                });
        if !complete_response && !valid_partial {
            return Err(InventoryError::Upstream(
                "GGUF source did not honor the exact bounded range request".to_owned(),
            ));
        }
        if response.content_length() != Some(expected_length) {
            return Err(InventoryError::Upstream(
                "GGUF header response length did not match its content range".to_owned(),
            ));
        }
        self.url = response.url().clone();
        let mut bytes = Vec::with_capacity(range_length);
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|error| InventoryError::Upstream(error.to_string()))?
        {
            if bytes.len().saturating_add(chunk.len()) > range_length {
                return Err(InventoryError::Upstream(
                    "GGUF header response exceeded its requested range".to_owned(),
                ));
            }
            bytes.extend_from_slice(&chunk);
        }
        if bytes.len() != range_length {
            return Err(InventoryError::Upstream(
                "GGUF header response ended before its requested range".to_owned(),
            ));
        }
        Ok(bytes)
    }
}

/// Every byte of a remote GGUF through its aligned tensor-data offset (see `remote_header`),
/// when the engine's parser accepts it.
async fn fetch_header(
    http: &reqwest::Client,
    token: Option<&str>,
    endpoint: &str,
    repository: &str,
    commit: &str,
    path: &Path,
    logical_size: u64,
) -> Result<Vec<u8>, InventoryError> {
    let mut remote = RemoteRanges::new(
        http,
        token,
        endpoint,
        repository,
        commit,
        path,
        logical_size,
    )?;
    let header = crate::remote_header::read_header(&mut remote, &[]).await?;
    match header_verdict(header).await? {
        (header, Ok(())) => Ok(header),
        (_, Err(detail)) => Err(InventoryError::Integrity(detail)),
    }
}

/// A complete header and whether the engine's parser accepts it, or why not.
async fn header_verdict(header: Vec<u8>) -> Result<(Vec<u8>, Result<(), String>), InventoryError> {
    tokio::task::spawn_blocking(move || {
        let verdict = match header_prefix(&header) {
            Ok(HeaderPrefix::Complete(length)) if length >= header.len() => Ok(()),
            Ok(_) => Err("the GGUF header's layout is inconsistent".to_owned()),
            Err(InventoryError::Integrity(detail)) => Err(detail),
            Err(error) => return Err(error),
        };
        Ok((header, verdict))
    })
    .await
    .map_err(|error| InventoryError::Internal(error.to_string()))?
}

/// What a GGUF prefix says about its header.
pub(crate) enum HeaderPrefix {
    Complete(usize),
    Truncated,
}

/// The aligned tensor-data offset of a GGUF prefix, `Truncated` when the prefix ends before the
/// header does, or an integrity error when the header itself is invalid: reading further can't
/// change that verdict.
pub(crate) fn header_prefix(prefix: &[u8]) -> Result<HeaderPrefix, InventoryError> {
    let temporary =
        tempfile::NamedTempFile::new().map_err(|error| InventoryError::Io(error.to_string()))?;
    fs::write(temporary.path(), prefix).map_err(|error| InventoryError::Io(error.to_string()))?;
    let directory = match crate::gguf::known_tensors(magnitude_artifacts::gguf::inspect_header(
        temporary.path(),
    )) {
        Ok(directory) => directory,
        Err(magnitude_artifacts::Error::Truncated { .. }) => return Ok(HeaderPrefix::Truncated),
        Err(magnitude_artifacts::Error::Io(error))
            if error.kind() == std::io::ErrorKind::UnexpectedEof =>
        {
            return Ok(HeaderPrefix::Truncated);
        }
        Err(error) => return Err(InventoryError::Integrity(error.to_string())),
    };
    usize::try_from(directory.data_offset)
        .map(HeaderPrefix::Complete)
        .map_err(|_| InventoryError::Integrity("GGUF header is too large".to_owned()))
}

/// The aligned tensor-data offset of a GGUF prefix, or an error when the prefix is too short to
/// hold the whole header.
#[cfg(test)]
fn header_length(prefix: &[u8]) -> Result<usize, InventoryError> {
    match header_prefix(prefix)? {
        HeaderPrefix::Complete(length) => Ok(length),
        HeaderPrefix::Truncated => Err(InventoryError::Integrity(
            "the GGUF prefix ends before its header".to_owned(),
        )),
    }
}

fn valid_content_range(
    value: &str,
    expected_start: u64,
    expected_end: u64,
    logical_size: u64,
) -> bool {
    let Some(value) = value.strip_prefix("bytes ") else {
        return false;
    };
    let Some((range, total)) = value.split_once('/') else {
        return false;
    };
    let Some((start, end)) = range.split_once('-') else {
        return false;
    };
    start.parse::<u64>().ok() == Some(expected_start)
        && end.parse::<u64>().ok() == expected_end.checked_sub(1)
        && total.parse::<u64>().ok() == Some(logical_size)
}

/// A gated or private repository's files refused to this request: without `HF_TOKEN`, or with a
/// token whose account hasn't been granted access.
pub(crate) fn hub_access_required(with_token: bool) -> InventoryError {
    InventoryError::ModelOperation {
        code: "hugging_face_access_required".to_owned(),
        message: if with_token {
            "This repository requires accepting its terms on Hugging Face, and the account of your HF_TOKEN doesn't have access yet. Accept its terms on the repository's page with that account, then try again."
        } else {
            "This repository requires accepting its terms on Hugging Face and an access token. Set HF_TOKEN to a token from an account that has accepted them, then try again."
        }
        .to_owned(),
        retryable: false,
    }
}

/// A repository whose metadata the Hub refuses: it doesn't exist, or it's private or gated and this
/// request has no access. The Hub answers an anonymous request for a missing repository with 401,
/// so neither status tells them apart. A gated repository shows its metadata, and its refused files
/// get [`hub_access_required`].
pub(crate) fn hub_repository_not_found() -> InventoryError {
    InventoryError::ModelOperation {
        code: "hugging_face_repository_not_found".to_owned(),
        message: "Repository not found, or it's private or gated. Check the name, or set HF_TOKEN if you have access.".to_owned(),
        retryable: false,
    }
}

/// A Hub metadata GET with the explicit `HF_TOKEN` when present, retried on rate limits and server
/// failures. A rejected credential (401) falls back to an anonymous request, so an expired token
/// never hides public repositories.
pub(crate) async fn send_hub_get(
    http: &reqwest::Client,
    token: Option<&str>,
    url: reqwest::Url,
) -> Result<reqwest::Response, InventoryError> {
    let response = send_hub_request(|| {
        let request = http.get(url.clone());
        match &token {
            Some(token) => request.bearer_auth(token),
            None => request,
        }
    })
    .await?;
    if token.is_some() && response.status() == reqwest::StatusCode::UNAUTHORIZED {
        return send_hub_request(|| http.get(url.clone())).await;
    }
    Ok(response)
}

impl ManagedModelStore {
    /// Resolves which repository file each unresolved `hf.co` Ollama pull is, by matching the
    /// pull's SHA-256 against the repository's files, and caches the answer (no match included).
    /// Returns the enabled Ollama locations to rescan when any pull resolved to a file; a pull
    /// whose repository can't be read stays unresolved and is tried again on a later scan.
    pub(crate) async fn resolve_ollama_hub_files(&self) -> BTreeSet<PathBuf> {
        let Ok(configuration) = self.local_models_configuration() else {
            return BTreeSet::new();
        };
        let roots = configuration.roots;
        let cache = self.cache.clone();
        let scanned = roots.clone();
        let unresolved = tokio::task::spawn_blocking(move || {
            crate::local_roots::unresolved_ollama_hub_pulls(&scanned, &cache)
        })
        .await
        .unwrap_or_default();
        let store = Arc::new(self.clone());
        let mut resolved_any = false;
        for (repository, sha256) in unresolved {
            let snapshot = match refresh_hugging_face_repository(
                &store,
                HuggingFaceRepositoryRequest {
                    repository: repository.clone(),
                    revision: "main".to_owned(),
                },
            )
            .await
            {
                Ok(snapshot) => snapshot,
                Err(InventoryError::NotFound(_)) => {
                    crate::local_roots::write_ollama_hub_file(
                        &self.cache,
                        &repository,
                        &sha256,
                        &crate::local_roots::OllamaHubFile { file: None },
                    );
                    continue;
                }
                Err(error) => {
                    tracing::debug!(%error, repository = %repository, "Ollama pull's repository is unavailable");
                    continue;
                }
            };
            let file = snapshot
                .gguf_files
                .iter()
                .find(|file| matches!(&file.content, ContentIdentity::Sha256 { value } if value == &sha256))
                .map(|file| file.path.to_string_lossy().replace('\\', "/"));
            resolved_any |= file.is_some();
            crate::local_roots::write_ollama_hub_file(
                &self.cache,
                &repository,
                &sha256,
                &crate::local_roots::OllamaHubFile { file },
            );
        }
        if !resolved_any {
            return BTreeSet::new();
        }
        roots
            .into_iter()
            .filter(|root| {
                root.enabled
                    && root.kind == magnitude_service_contracts::models::ModelRootKind::Ollama
            })
            .map(|root| crate::inventory::canonical_or_self(&root.path))
            .collect()
    }
}

/// A repository snapshot. An immutable commit already cached is served without a request, since
/// its files and lineage can never change; anything else is refreshed.
pub async fn pinned_hugging_face_repository(
    models: &Arc<ManagedModelStore>,
    request: HuggingFaceRepositoryRequest,
) -> Result<HuggingFaceRepositorySnapshot, InventoryError> {
    validate_repository_request(&request)?;
    if immutable_commit(request.revision.clone()).is_some() {
        let cache_key = repository_cache_key(models.client.endpoint(), &request);
        if let Some(cached) = models
            .cache
            .read_index::<CachedHuggingFaceRepositorySnapshot>(
                ModelIndexKind::HuggingFaceRepositorySnapshot,
                &cache_key,
            )
            .filter(|cached| cached.snapshot.commit == request.revision)
        {
            return Ok(cached.snapshot);
        }
    }
    refresh_hugging_face_repository(models, request).await
}

fn classification_evidence(
    digest: &str,
    architectures: &crate::candidate_files::RecognizedArchitectures,
) -> String {
    format!(
        "candidate-classification-v5:{digest}:{}",
        architectures.key()
    )
}

fn remote_header_evidence(endpoint: &str, repository: &str, commit: &str, path: &Path) -> String {
    format!("{endpoint}:{repository}@{commit}/{}", path.display())
}

/// One remote GGUF header: every byte through the aligned tensor-data offset, and its sha256.
#[derive(Clone, Debug)]
pub struct RemoteGgufHeader {
    pub digest: String,
    pub bytes: Vec<u8>,
}

/// What reading one remote file's header found: its bytes (stored in the header cache by
/// digest), or why the engine can't read it, with the bytes when the header is complete but
/// invalid (a tensor type the engine doesn't know). Both are facts of the immutable file, so
/// neither is fetched again.
#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CachedRemoteGgufHeader {
    content: ContentIdentity,
    header: CachedRemoteHeader,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "_tag", rename_all = "PascalCase", deny_unknown_fields)]
enum CachedRemoteHeader {
    Read {
        digest: String,
    },
    Unreadable {
        detail: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        digest: Option<String>,
    },
}

impl CachedRemoteHeader {
    fn digest(&self) -> Option<&str> {
        match self {
            Self::Read { digest } => Some(digest),
            Self::Unreadable { digest, .. } => digest.as_deref(),
        }
    }
}

impl ManagedModelStore {
    /// The header of one file of an immutable repository snapshot, fetched by range requests once
    /// and then served from the header cache. `references` are headers of other files of the same
    /// repository, whose large metadata entries (a shared tokenizer) this file's header most
    /// likely holds too (see `remote_header`).
    pub async fn hugging_face_gguf_header(
        &self,
        repository: &str,
        commit: &str,
        file: &HuggingFaceRepositoryFile,
        references: &[&[u8]],
    ) -> Result<RemoteGgufHeader, InventoryError> {
        let evidence =
            remote_header_evidence(self.client.endpoint(), repository, commit, &file.path);
        match self.cached_hugging_face_header(repository, commit, file) {
            Some(CachedRemoteHeader::Unreadable { detail, .. }) => {
                return Err(InventoryError::Integrity(detail));
            }
            Some(CachedRemoteHeader::Read { digest }) => {
                let cache = self.cache.clone();
                let cached = tokio::task::spawn_blocking(move || {
                    cache
                        .read_header(&digest)
                        .map(|bytes| RemoteGgufHeader { digest, bytes })
                })
                .await
                .map_err(|error| InventoryError::Internal(error.to_string()))?;
                if let Some(cached) = cached {
                    return Ok(cached);
                }
            }
            None => {}
        }
        let mut remote = RemoteRanges::new(
            &self.http,
            self.hub_token(),
            self.client.endpoint(),
            repository,
            commit,
            &file.path,
            file.size_bytes,
        )?;
        let (bytes, verdict) =
            match crate::remote_header::read_header(&mut remote, references).await {
                Ok(bytes) => header_verdict(bytes).await?,
                Err(InventoryError::Integrity(detail)) => (Vec::new(), Err(detail)),
                Err(error) => return Err(error),
            };
        let store = self.clone();
        let content = file.content.clone();
        tokio::task::spawn_blocking(move || {
            let digest = (!bytes.is_empty()).then(|| store.write_header(&bytes));
            let header = match (&verdict, digest.clone()) {
                (Ok(()), Some(digest)) => CachedRemoteHeader::Read { digest },
                (Ok(()), None) => CachedRemoteHeader::Unreadable {
                    detail: "the GGUF header is empty".to_owned(),
                    digest: None,
                },
                (Err(detail), digest) => CachedRemoteHeader::Unreadable {
                    detail: detail.clone(),
                    digest,
                },
            };
            store.cache.write_index(
                ModelIndexKind::HuggingFaceGgufHeader,
                &evidence,
                &CachedRemoteGgufHeader { content, header },
            );
            match (verdict, digest) {
                (Ok(()), Some(digest)) => Ok(RemoteGgufHeader { digest, bytes }),
                (Ok(()), None) => Err(InventoryError::Integrity(
                    "the GGUF header is empty".to_owned(),
                )),
                (Err(detail), _) => Err(InventoryError::Integrity(detail)),
            }
        })
        .await
        .map_err(|error| InventoryError::Internal(error.to_string()))?
    }

    /// The cached header of one file of a repository snapshot, readable or not, when it holds
    /// large metadata entries other files of the repository may share: a reference for reading
    /// their headers (see `remote_header`). Nothing is fetched.
    #[must_use]
    pub fn hugging_face_reference_header(
        &self,
        repository: &str,
        commit: &str,
        file: &HuggingFaceRepositoryFile,
    ) -> Option<Vec<u8>> {
        let cached = self.cached_hugging_face_header(repository, commit, file)?;
        let bytes = self.cache.read_header(cached.digest()?)?;
        matches!(
            crate::header_layout::walk(&bytes),
            Ok(crate::header_layout::Layout::Complete { entries, .. })
                if entries.iter().any(crate::header_layout::Entry::shared)
        )
        .then_some(bytes)
    }

    /// Stores a header in the header cache, keeping the headers installed packages and the
    /// release catalog need when the cache evicts.
    pub(crate) fn write_header(&self, header: &[u8]) -> String {
        self.cache
            .write_header(header, || self.pinned_header_digests())
    }

    /// The headers the header cache never evicts: the release catalog's, and those of every file
    /// of an installed Hugging Face package.
    fn pinned_header_digests(&self) -> BTreeSet<String> {
        let mut pinned = self
            .config
            .catalog_header_digests
            .values()
            .cloned()
            .collect::<BTreeSet<_>>();
        if let Ok(snapshot) = self.installed_packages.read() {
            for record in snapshot.records.values() {
                let package = &record.installed.package;
                let magnitude_service_contracts::models::ModelPackageSource::HuggingFace {
                    repository,
                    revision,
                } = &package.source
                else {
                    continue;
                };
                for file in &package.files {
                    if let Some(digest) = self
                        .cache
                        .read_index::<CachedRemoteGgufHeader>(
                            ModelIndexKind::HuggingFaceGgufHeader,
                            &remote_header_evidence(
                                self.client.endpoint(),
                                repository,
                                revision,
                                &file.path,
                            ),
                        )
                        .and_then(|cached| cached.header.digest().map(ToOwned::to_owned))
                    {
                        pinned.insert(digest);
                    }
                }
            }
        }
        pinned
    }

    /// A local GGUF's facts and classification. Its header is read only when the file changed or
    /// was never classified, and then from the header cache when it holds it.
    pub fn classify_local_gguf(
        &self,
        path: &Path,
        architectures: &crate::candidate_files::RecognizedArchitectures,
    ) -> Result<(crate::gguf::LocalGguf, crate::candidate_files::HeaderClassification), InventoryError>
    {
        let mut read = None;
        let gguf = crate::inventory::local_gguf_cached(&self.cache, path, |header| {
            self.write_header(header);
            read = Some(header.to_vec());
        })?;
        let size_bytes = std::fs::metadata(path)
            .map_err(|error| InventoryError::Integrity(format!("{}: {error}", path.display())))?
            .len();
        let evidence = classification_evidence(&gguf.inspection.header_digest, architectures);
        if let Some(classification) = self
            .cache
            .read_index::<crate::candidate_files::HeaderClassification>(
                ModelIndexKind::CandidateClassification,
                &evidence,
            )
        {
            return Ok((gguf, classification.for_file(size_bytes)));
        }
        let header = match read.or_else(|| self.cache.read_header(&gguf.inspection.header_digest)) {
            Some(header) => header,
            None => self.local_gguf_header(path)?.1,
        };
        let classification = crate::candidate_files::classify_header(&header, architectures)?;
        self.cache
            .write_index(ModelIndexKind::CandidateClassification, &evidence, &classification);
        Ok((gguf, classification.for_file(size_bytes)))
    }

    /// The header digest and classification of one file of an immutable repository snapshot.
    /// Once classified, a file costs one index read: neither its header nor the network is
    /// touched again.
    pub async fn classify_hugging_face_gguf(
        &self,
        repository: &str,
        commit: &str,
        file: &HuggingFaceRepositoryFile,
        architectures: &crate::candidate_files::RecognizedArchitectures,
        references: &[&[u8]],
    ) -> Result<(String, crate::candidate_files::HeaderClassification), InventoryError> {
        if let Some(classified) =
            self.cached_hugging_face_classification(repository, commit, file, architectures)
        {
            return Ok(classified);
        }
        let header = self
            .hugging_face_gguf_header(repository, commit, file, references)
            .await?;
        let classifier = architectures.clone();
        let classification = tokio::task::spawn_blocking(move || {
            crate::candidate_files::classify_header(&header.bytes, &classifier)
        })
        .await
        .map_err(|error| InventoryError::Internal(error.to_string()))??;
        self.cache.write_index(
            ModelIndexKind::CandidateClassification,
            &classification_evidence(&header.digest, architectures),
            &classification,
        );
        Ok((header.digest, classification.for_file(file.size_bytes)))
    }

    /// A file's header digest and classification when it was classified before.
    #[must_use]
    pub fn cached_hugging_face_classification(
        &self,
        repository: &str,
        commit: &str,
        file: &HuggingFaceRepositoryFile,
        architectures: &crate::candidate_files::RecognizedArchitectures,
    ) -> Option<(String, crate::candidate_files::HeaderClassification)> {
        let digest = self.cached_hugging_face_header_digest(repository, commit, file)?;
        let classification = self
            .cache
            .read_index::<crate::candidate_files::HeaderClassification>(
                ModelIndexKind::CandidateClassification,
                &classification_evidence(&digest, architectures),
            )?;
        Some((digest, classification.for_file(file.size_bytes)))
    }

    /// The header digest of one readable file of an immutable repository snapshot when its
    /// header was fetched before, without reading the header itself.
    #[must_use]
    pub fn cached_hugging_face_header_digest(
        &self,
        repository: &str,
        commit: &str,
        file: &HuggingFaceRepositoryFile,
    ) -> Option<String> {
        match self.cached_hugging_face_header(repository, commit, file)? {
            CachedRemoteHeader::Read { digest } => Some(digest),
            CachedRemoteHeader::Unreadable { .. } => None,
        }
    }

    fn cached_hugging_face_header(
        &self,
        repository: &str,
        commit: &str,
        file: &HuggingFaceRepositoryFile,
    ) -> Option<CachedRemoteHeader> {
        self.cache
            .read_index::<CachedRemoteGgufHeader>(
                ModelIndexKind::HuggingFaceGgufHeader,
                &remote_header_evidence(self.client.endpoint(), repository, commit, &file.path),
            )
            .filter(|cached| cached.content == file.content)
            .map(|cached| cached.header)
    }

    /// The exact package a repository-snapshot preview describes, identified as the same package
    /// a rescan of its downloaded files produces.
    pub async fn hugging_face_package(
        &self,
        source: &ModelPreviewSource,
        snapshot: &HuggingFaceRepositorySnapshot,
    ) -> Result<(ModelPackage, PreparedPreview), InventoryError> {
        let prepared = self
            .prepare_preview_from_repository_snapshot(source, snapshot)
            .await?;
        let package = crate::package_service::preview_package(&prepared.model)?;
        Ok((package, prepared))
    }

    /// Whether an installed package, in any location, another app's included, holds these
    /// drafter bytes.
    #[must_use]
    pub fn has_installed_content(&self, content: &crate::DrafterContent) -> bool {
        self.installed_packages.read().is_ok_and(|snapshot| {
            snapshot
                .records
                .values()
                .any(|record| content.held_by(record))
        })
    }

    /// The catalog model whose installed target, in any location, has these bytes: the `local:`
    /// digest and primary size of a file on disk. A copy of a catalog target's bytes, wherever it
    /// is, is that catalog model.
    #[must_use]
    pub fn catalog_model_with_content(
        &self,
        local_digest: &str,
        primary_size: u64,
    ) -> Option<ModelId> {
        let snapshot = self.installed_packages.read().ok()?;
        snapshot.records.values().find_map(|record| {
            let magnitude_service_contracts::models::InstalledCatalogAttribution::Attributed {
                model_id,
                variant_id,
            } = &record.installed.catalog_attribution
            else {
                return None;
            };
            let primary = record
                .installed
                .package
                .files
                .iter()
                .filter(|file| {
                    file.role == magnitude_service_contracts::models::ModelFileRole::Weights
                })
                .min_by(|left, right| left.path.cmp(&right.path))?;
            (primary.size_bytes == primary_size
                && record
                    .primary
                    .as_ref()
                    .and_then(|evidence| evidence.local_digest.as_deref())
                    == Some(local_digest))
            .then(|| ModelId::catalog(model_id, variant_id))
        })
    }

    /// Whether an installed package, in any location, is exactly `package_id`.
    #[must_use]
    pub fn has_installed_package(
        &self,
        package_id: &magnitude_service_contracts::models::ModelPackageId,
    ) -> bool {
        self.installed_packages.read().is_ok_and(|snapshot| {
            snapshot
                .records
                .values()
                .any(|record| record.installed.package.id == *package_id)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use magnitude_service_contracts::LocalDeclaration;

    #[test]
    fn hub_coordinates_reject_path_traversal_but_allow_branch_names() {
        assert!(valid_repository("owner/model.gguf"));
        assert!(!valid_repository("../model"));
        assert!(!valid_repository("owner/model/extra"));
        assert!(valid_hub_revision("refs/pr/123"));
        assert!(valid_hub_revision("feature/catalog-v2"));
        assert!(!valid_hub_revision("../main"));
        assert!(!valid_hub_revision("main?blobs=true"));
    }

    #[test]
    fn hub_snapshot_pins_symbolic_revisions_and_preserves_immutable_requests() {
        let commit = "a".repeat(40);
        assert!(validate_snapshot_revision("main", &commit).is_ok());
        assert!(validate_snapshot_revision(&commit, &commit).is_ok());
        assert!(validate_snapshot_revision(&"b".repeat(40), &commit).is_err());
    }

    #[test]
    fn hub_snapshot_keeps_only_identified_gguf_files_and_live_metadata() {
        let snapshot = HubModel {
            id: Some("owner/model-gguf".to_owned()),
            sha: Some("b".repeat(40)),
            last_modified: None,
            downloads: Some(11),
            likes: Some(3),
            gated: Some(serde_json::Value::String("manual".to_owned())),
            private: Some(false),
            tags: vec![
                "gguf".to_owned(),
                "license:apache-2.0".to_owned(),
                "base_model:quantized:owner/base".to_owned(),
            ],
            card_data: Some(serde_json::json!({
                "license": "apache-2.0",
                "license_link": "https://example.invalid/license",
                "base_model": ["owner/base"]
            })),
            siblings: vec![
                HubSibling {
                    rfilename: "model-Q4_K_M.gguf".to_owned(),
                    size: Some(123),
                    blob_id: Some("c".repeat(40)),
                    lfs: Some(HubLfs {
                        sha256: "d".repeat(64),
                        size: 123,
                    }),
                },
                HubSibling {
                    rfilename: "README.md".to_owned(),
                    size: Some(5),
                    blob_id: Some("e".repeat(40)),
                    lfs: None,
                },
            ],
        }
        .into_snapshot("owner/model-gguf".to_owned(), &"b".repeat(40))
        .unwrap();

        assert_eq!(snapshot.commit, "b".repeat(40));
        assert!(snapshot.gated);
        assert_eq!(snapshot.license.as_deref(), Some("apache-2.0"));
        assert_eq!(snapshot.base_models, vec!["owner/base"]);
        assert_eq!(snapshot.gguf_files.len(), 1);
        assert_eq!(snapshot.gguf_files[0].size_bytes, 123);
        assert!(matches!(
            snapshot.gguf_files[0].content,
            ContentIdentity::Sha256 { .. }
        ));
    }

    #[test]
    fn repository_snapshot_supplies_exact_preview_components_without_metadata_refetch() {
        let source = ModelPreviewSource {
            repository: "owner/model-gguf".to_owned(),
            revision: "a".repeat(40),
            primary_gguf: PathBuf::from("model-00001-of-00002.gguf"),
            additional_components: Vec::new(),
        };
        let files = vec![
            HuggingFaceRepositoryFile {
                path: PathBuf::from("model-00001-of-00002.gguf"),
                size_bytes: 10,
                content: ContentIdentity::Sha256 {
                    value: "b".repeat(64),
                },
            },
            HuggingFaceRepositoryFile {
                path: PathBuf::from("model-00002-of-00002.gguf"),
                size_bytes: 20,
                content: ContentIdentity::Sha256 {
                    value: "c".repeat(64),
                },
            },
        ];

        let selected = select_repository_snapshot_components(&files, &source).unwrap();
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].role, ComponentRole::Shard);
        assert_eq!(selected[0].shard_index, Some(1));
        assert_eq!(selected[1].shard_index, Some(2));
        assert_eq!(selected[1].size_bytes, 20);
    }

    fn sibling(path: &str, size: u64) -> HubSibling {
        HubSibling {
            rfilename: path.to_owned(),
            size: Some(size),
            blob_id: Some("a".repeat(40)),
            lfs: None,
        }
    }

    #[test]
    fn source_validation_requires_an_immutable_commit_and_safe_gguf_path() {
        let valid = ModelPreviewSource {
            repository: "owner/repository".to_owned(),
            revision: "a".repeat(40),
            primary_gguf: PathBuf::from("models/model.gguf"),
            additional_components: Vec::new(),
        };
        assert!(validate_source(&valid).is_ok());
        assert!(
            validate_source(&ModelPreviewSource {
                revision: "main".to_owned(),
                ..valid.clone()
            })
            .is_err()
        );
        assert!(
            validate_source(&ModelPreviewSource {
                primary_gguf: PathBuf::from("../model.gguf"),
                ..valid
            })
            .is_err()
        );
    }

    #[tokio::test]
    async fn fresh_preview_survives_an_unusable_persistent_cache() {
        let temporary = tempfile::tempdir().unwrap();
        let store = temporary.path().join("model-store");
        let cache_root = temporary.path().join("cache");
        let mut config =
            crate::inventory::InventoryConfig::with_roots(store.clone(), cache_root.clone())
                .unwrap();
        config.local_models.roots.clear();
        let manager = ManagedModelStore::open(config).await.unwrap();
        let _ = fs::remove_dir_all(&cache_root);
        fs::write(&cache_root, b"not a directory").unwrap();

        let mut header = Vec::new();
        header.extend_from_slice(b"GGUF");
        header.extend_from_slice(&3_u32.to_le_bytes());
        header.extend_from_slice(&0_u64.to_le_bytes());
        header.extend_from_slice(&0_u64.to_le_bytes());
        header.resize(32, 0);
        let header_digest = format!("{:x}", Sha256::digest(&header));
        let prepared = manager
            .materialize_preview(CachedArtifact {
                repository: "owner/repository".to_owned(),
                commit: "a".repeat(40),
                primary_gguf: PathBuf::from("model.gguf"),
                components: vec![CachedComponent {
                    component: ModelComponent {
                        path: PathBuf::from("model.gguf"),
                        role: ComponentRole::Weights,
                        size_bytes: header.len() as u64,
                        content: ContentIdentity::Sha256 {
                            value: header_digest.clone(),
                        },
                        shard_index: None,
                        relationship: None,
                    },
                    header_digest,
                    acquired_header: Some(header),
                }],
            })
            .unwrap();
        assert!(matches!(
            prepared.model.model.properties,
            magnitude_service_contracts::InventoryProperties::Inspected { .. }
        ));
    }

    #[test]
    fn sharded_preview_resolves_the_complete_ordered_set() {
        let siblings = vec![
            sibling("weights/model-00002-of-00003.gguf", 2),
            sibling("weights/model-00001-of-00003.gguf", 1),
            sibling("weights/model-00003-of-00003.gguf", 3),
            sibling("weights/unrelated.gguf", 4),
        ];
        let selected =
            select_components(&siblings, Path::new("weights/model-00001-of-00003.gguf")).unwrap();
        assert_eq!(
            selected
                .iter()
                .map(|item| item.rfilename.as_str())
                .collect::<Vec<_>>(),
            vec![
                "weights/model-00001-of-00003.gguf",
                "weights/model-00002-of-00003.gguf",
                "weights/model-00003-of-00003.gguf",
            ]
        );

        let incomplete = vec![
            sibling("model-00001-of-00002.gguf", 1),
            sibling("other.gguf", 2),
        ];
        assert!(select_components(&incomplete, Path::new("model-00001-of-00002.gguf")).is_err());
    }

    #[test]
    fn preview_selects_explicit_execution_companions_with_typed_relationships() {
        let siblings = vec![
            sibling("model.gguf", 1),
            sibling("projector.gguf", 2),
            sibling("draft.gguf", 3),
        ];
        let source = ModelPreviewSource {
            repository: "owner/repository".to_owned(),
            revision: "a".repeat(40),
            primary_gguf: PathBuf::from("model.gguf"),
            additional_components: vec![
                magnitude_service_contracts::ModelPreviewComponentSource {
                    path: PathBuf::from("projector.gguf"),
                    role: magnitude_service_contracts::ModelPreviewComponentRole::Projector,
                },
                magnitude_service_contracts::ModelPreviewComponentSource {
                    path: PathBuf::from("draft.gguf"),
                    role: magnitude_service_contracts::ModelPreviewComponentRole::Draft {
                        method: magnitude_service_contracts::models::SpeculativeMethod::DFlash,
                    },
                },
            ],
        };
        validate_source(&source).unwrap();
        let selected = select_artifact_components(&siblings, &source).unwrap();
        assert_eq!(selected.len(), 3);
        assert_eq!(selected[0].role, ComponentRole::Weights);
        assert!(matches!(
            selected[1].relationship,
            Some(ComponentRelationship::ProjectorFor { .. })
        ));
        assert!(matches!(
            selected[2].relationship,
            Some(ComponentRelationship::DraftFor { .. })
        ));

        let duplicate = ModelPreviewSource {
            additional_components: vec![magnitude_service_contracts::ModelPreviewComponentSource {
                path: PathBuf::from("model.gguf"),
                role: magnitude_service_contracts::ModelPreviewComponentRole::Mtp,
            }],
            ..source
        };
        assert!(validate_source(&duplicate).is_err());
    }

    #[test]
    fn content_identity_prefers_published_lfs_digest() {
        let sibling = HubSibling {
            rfilename: "model.gguf".to_owned(),
            size: Some(42),
            blob_id: Some("git-oid".to_owned()),
            lfs: Some(HubLfs {
                sha256: "a".repeat(64),
                size: 42,
            }),
        };
        assert_eq!(
            sibling_identity(&sibling).unwrap(),
            (
                42,
                ContentIdentity::Sha256 {
                    value: "a".repeat(64)
                }
            )
        );
    }

    #[test]
    fn content_range_must_cover_the_exact_prefix_and_logical_file() {
        assert!(valid_content_range("bytes 0-1023/4096", 0, 1024, 4096));
        assert!(valid_content_range(
            "bytes 1024-2047/4096",
            1024,
            2048,
            4096
        ));
        assert!(!valid_content_range("bytes 1-1024/4096", 0, 1024, 4096));
        assert!(!valid_content_range("bytes 0-1024/4096", 0, 1024, 4096));
        assert!(!valid_content_range("bytes 0-1023/8192", 0, 1024, 4096));
    }

    #[test]
    fn retries_only_server_failures_never_a_rate_limit() {
        assert!(!retryable_hub_status(reqwest::StatusCode::TOO_MANY_REQUESTS));
        assert!(retryable_hub_status(reqwest::StatusCode::BAD_GATEWAY));
        assert!(!retryable_hub_status(reqwest::StatusCode::NOT_FOUND));
        assert!(!retryable_hub_status(reqwest::StatusCode::UNAUTHORIZED));
    }

    #[tokio::test]
    async fn sparse_preview_headers_produce_the_same_properties_as_complete_models() {
        use std::io::Read;

        let inference_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let fixtures = [
            inference_root.join("target/parity-models/tinyllamas/stories15M-q4_0.gguf"),
            inference_root
                .join("target/parity-models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
        ];
        let fixtures = fixtures
            .into_iter()
            .filter(|path| path.is_file())
            .collect::<Vec<_>>();
        if fixtures.is_empty() {
            return;
        }

        let temporary = tempfile::tempdir().unwrap();
        let mut config = crate::inventory::InventoryConfig::with_roots(
            temporary.path().join("model-store"),
            temporary.path().join("cache"),
        )
        .unwrap();
        config.local_models.roots.clear();
        let manager = ManagedModelStore::open(config).await.unwrap();

        for (index, fixture) in fixtures.into_iter().enumerate() {
            let inspection = crate::gguf::inspect(&fixture).unwrap();
            let mut header = vec![0_u8; usize::try_from(inspection.header_bytes).unwrap()];
            std::fs::File::open(&fixture)
                .unwrap()
                .read_exact(&mut header)
                .unwrap();
            let header_digest = manager.write_header(&header);
            let relative =
                PathBuf::from(fixture.file_name().and_then(|name| name.to_str()).unwrap());
            let preview_component = ModelComponent {
                path: relative.clone(),
                role: ComponentRole::Weights,
                size_bytes: fixture.metadata().unwrap().len(),
                content: ContentIdentity::FileIdentity {
                    value: format!("fixture-{index}"),
                },
                shard_index: None,
                relationship: None,
            };
            let prepared = manager
                .materialize_preview(CachedArtifact {
                    repository: "test/fixtures".to_owned(),
                    commit: format!("{index:040x}"),
                    primary_gguf: relative,
                    components: vec![CachedComponent {
                        component: preview_component.clone(),
                        header_digest,
                        acquired_header: None,
                    }],
                })
                .unwrap();

            let full_component = ModelComponent {
                path: fixture.clone(),
                ..preview_component
            };
            let full = build_model(
                InventoryEntryId(format!("mdl_{index:064x}")),
                content_id(std::slice::from_ref(&full_component)),
                1,
                1,
                ModelSource::Local {
                    declared_by: LocalDeclaration::Discovery,
                },
                ModelLocation::File {
                    path: fixture.clone(),
                    component: full_component,
                    integrity: Integrity::Unverified {
                        reason: "test fixture".to_owned(),
                    },
                },
                &fixture,
                false,
                &manager.cache,
            )
            .unwrap();
            assert_eq!(
                prepared.model.model.properties,
                full.properties,
                "preview metadata diverged for {}",
                fixture.display()
            );
        }
    }

    /// A minimal Hub: search and commit-pinned repository metadata, from a route table.
    fn serve_hub(
        routes: impl Fn(&str, &str) -> Option<serde_json::Value> + Send + Sync + 'static,
    ) -> (String, Arc<std::sync::atomic::AtomicUsize>) {
        use std::io::{BufRead, BufReader, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("hub listener");
        let endpoint = format!("http://{}", listener.local_addr().expect("hub address"));
        let requests = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let counted = Arc::clone(&requests);
        let routes = Arc::new(routes);
        std::thread::spawn(move || {
            for connection in listener.incoming() {
                let Ok(mut connection) = connection else {
                    return;
                };
                counted.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let mut reader = BufReader::new(connection.try_clone().expect("stream"));
                let mut request_line = String::new();
                if reader.read_line(&mut request_line).is_err() {
                    continue;
                }
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                        break;
                    }
                }
                let target = request_line
                    .split_whitespace()
                    .nth(1)
                    .unwrap_or("/")
                    .to_owned();
                let (path, query) = target.split_once('?').unwrap_or((&target, ""));
                let (status, body) = match routes(path, query) {
                    Some(body) => ("200 OK", serde_json::to_vec(&body).unwrap()),
                    None => ("404 Not Found", Vec::new()),
                };
                let _ = write!(
                    connection,
                    "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = connection.write_all(&body);
            }
        });
        (endpoint, requests)
    }

    /// A file server that honors exact range requests on files named by their last path segment,
    /// and counts the bytes it sends.
    fn serve_ranges(
        files: Vec<(&'static str, Vec<u8>)>,
    ) -> (
        String,
        Arc<std::sync::atomic::AtomicUsize>,
        Arc<std::sync::atomic::AtomicUsize>,
    ) {
        use std::io::{BufRead, BufReader, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("range listener");
        let endpoint = format!("http://{}", listener.local_addr().expect("range address"));
        let requests = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let sent = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (counted, sent_counted) = (Arc::clone(&requests), Arc::clone(&sent));
        let files = Arc::new(files.into_iter().collect::<BTreeMap<_, _>>());
        std::thread::spawn(move || {
            for connection in listener.incoming() {
                let Ok(mut connection) = connection else {
                    return;
                };
                let (files, counted, sent) = (
                    Arc::clone(&files),
                    Arc::clone(&counted),
                    Arc::clone(&sent_counted),
                );
                std::thread::spawn(move || {
                    counted.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let mut reader = BufReader::new(connection.try_clone().expect("stream"));
                    let mut request_line = String::new();
                    let _ = reader.read_line(&mut request_line);
                    let name = request_line
                        .split_whitespace()
                        .nth(1)
                        .and_then(|target| target.rsplit('/').next())
                        .unwrap_or_default()
                        .to_owned();
                    let mut range = None;
                    loop {
                        let mut line = String::new();
                        if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                            break;
                        }
                        if let Some((header, value)) = line.split_once(':')
                            && header.eq_ignore_ascii_case("range")
                        {
                            let (start, end) = value
                                .trim()
                                .trim_start_matches("bytes=")
                                .split_once('-')
                                .unwrap();
                            range = Some((
                                start.parse::<usize>().unwrap(),
                                end.parse::<usize>().unwrap(),
                            ));
                        }
                    }
                    let file = &files[name.as_str()];
                    let (start, end) = range.unwrap_or((0, file.len() - 1));
                    let body = &file[start..=end];
                    sent.fetch_add(body.len(), std::sync::atomic::Ordering::SeqCst);
                    let _ = write!(
                        connection,
                        "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: bytes {start}-{end}/{}\r\nConnection: close\r\n\r\n",
                        body.len(),
                        file.len()
                    );
                    let _ = connection.write_all(body);
                });
            }
        });
        (endpoint, requests, sent)
    }

    /// A model file with a token list of `tokens` entries (`renamed` replaces the middle one), a
    /// file type and `weights` bytes of tensor data.
    fn model_file(name: &str, tokens: usize, renamed: Option<&str>, file_type: u32) -> Vec<u8> {
        let mut file = crate::test_gguf::gguf_bytes(
            &[
                ("general.architecture", crate::test_gguf::Meta::Str("llama")),
                ("general.name", crate::test_gguf::Meta::Str(name)),
            ],
            &["token_embd.weight"],
        );
        let tensor_data = file.split_off(file.len() - 128);
        let tensor_info_start = file
            .windows(b"token_embd.weight".len())
            .position(|window| window == b"token_embd.weight")
            .unwrap()
            - 8;
        let mut tail = file.split_off(tensor_info_start);
        while tail.last() == Some(&0) {
            tail.pop();
        }
        let mut tokenizer = Vec::new();
        let push_string = |bytes: &mut Vec<u8>, value: &str| {
            bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
            bytes.extend_from_slice(value.as_bytes());
        };
        push_string(&mut tokenizer, "tokenizer.ggml.tokens");
        tokenizer.extend_from_slice(&9_u32.to_le_bytes());
        tokenizer.extend_from_slice(&8_u32.to_le_bytes());
        tokenizer.extend_from_slice(&(tokens as u64).to_le_bytes());
        for token in 0..tokens {
            let value = format!("token-{token:07}");
            push_string(
                &mut tokenizer,
                renamed.filter(|_| token == tokens / 2).unwrap_or(&value),
            );
        }
        push_string(&mut tokenizer, "general.file_type");
        tokenizer.extend_from_slice(&4_u32.to_le_bytes());
        tokenizer.extend_from_slice(&file_type.to_le_bytes());
        file[16..24].copy_from_slice(&4_u64.to_le_bytes());
        file.extend_from_slice(&tokenizer);
        file.extend_from_slice(&tail);
        file.resize(file.len().next_multiple_of(32), 0);
        file.extend_from_slice(&tensor_data);
        file.resize(file.len() + 4 * 1024 * 1024, 0);
        file
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_sibling_header_shares_its_tokenizer_and_a_different_one_is_read_whole() {
        let sibling_file = model_file("Model Q8_0", 150_000, None, 7);
        let same = model_file("Model Q4_K_M", 150_000, None, 15);
        let renamed = model_file("Model Q4_K_M", 150_000, Some("token-RENAMED"), 15);
        let (endpoint, _, sent) = serve_ranges(vec![
            ("sibling.gguf", sibling_file.clone()),
            ("same.gguf", same.clone()),
            ("renamed.gguf", renamed.clone()),
        ]);
        let (_temporary, store) = stub_store(endpoint).await;
        let file = |path: &str, bytes: &[u8]| HuggingFaceRepositoryFile {
            path: PathBuf::from(path),
            size_bytes: bytes.len() as u64,
            content: ContentIdentity::Sha256 {
                value: format!("{:x}", Sha256::digest(bytes)),
            },
        };
        let commit = "a".repeat(40);
        let sibling = store
            .hugging_face_gguf_header(
                "owner/repo",
                &commit,
                &file("sibling.gguf", &sibling_file),
                &[],
            )
            .await
            .unwrap();
        let same_length = header_length(&same).unwrap();
        assert!(same_length > 2 * 1024 * 1024);
        let before = sent.load(std::sync::atomic::Ordering::SeqCst);
        let shared = store
            .hugging_face_gguf_header(
                "owner/repo",
                &commit,
                &file("same.gguf", &same),
                &[&sibling.bytes],
            )
            .await
            .unwrap();
        assert_eq!(
            shared.bytes,
            same[..same_length],
            "the shared header is exactly the file's"
        );
        assert!(
            sent.load(std::sync::atomic::Ordering::SeqCst) - before < 512 * 1024,
            "only the prefix, the verification windows and the remainder are read"
        );
        let different = store
            .hugging_face_gguf_header(
                "owner/repo",
                &commit,
                &file("renamed.gguf", &renamed),
                &[&sibling.bytes],
            )
            .await
            .unwrap();
        assert_eq!(
            different.bytes,
            renamed[..header_length(&renamed).unwrap()],
            "a tokenizer that differs inside a verification window is read from the file"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_header_is_read_in_about_its_own_length_in_two_requests() {
        let description = "x".repeat(3 * 1024 * 1024);
        let mut file = crate::test_gguf::gguf_bytes(
            &[
                ("general.architecture", crate::test_gguf::Meta::Str("llama")),
                (
                    "general.description",
                    crate::test_gguf::Meta::Str(&description),
                ),
            ],
            &["token_embd.weight"],
        );
        file.resize(file.len() + 8 * 1024 * 1024, 0);
        let size = u64::try_from(file.len()).unwrap();
        let (endpoint, requests, sent) = serve_ranges(vec![("model-Q4_K_M.gguf", file.clone())]);
        let header = fetch_header(
            &reqwest::Client::new(),
            None,
            &endpoint,
            "owner/repo",
            &"a".repeat(40),
            Path::new("model-Q4_K_M.gguf"),
            size,
        )
        .await
        .unwrap();
        assert!(header.len() > 3 * 1024 * 1024 && file.starts_with(&header));
        assert_eq!(
            requests.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "a probe, then the rest its layout declares"
        );
        assert!(sent.load(std::sync::atomic::Ordering::SeqCst) <= header.len() + 64 * 1024);
    }

    /// A split's metadata-only first shard: no tensors, its header unpadded at the end of the file,
    /// as unsloth's `Qwen3.8-Flash-Next-UD-Q5_K_XL-00001-of-00006.gguf` (10,946,618 bytes) is.
    fn metadata_only_shard(description_bytes: usize) -> Vec<u8> {
        (description_bytes..)
            .map(|length| {
                let description = "x".repeat(length);
                let mut file = crate::test_gguf::gguf_bytes(
                    &[
                        (
                            "general.architecture",
                            crate::test_gguf::Meta::Str("qwen35moe"),
                        ),
                        ("split.count", crate::test_gguf::Meta::U32(6)),
                        (
                            "general.description",
                            crate::test_gguf::Meta::Str(&description),
                        ),
                    ],
                    &[],
                );
                while file.last() == Some(&0) {
                    file.pop();
                }
                file
            })
            .find(|file| file.len() % 32 != 0)
            .unwrap()
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_metadata_only_shard_without_trailing_padding_is_its_whole_header() {
        for description_bytes in [6 * 1024, 10 * 1024 * 1024] {
            let file = metadata_only_shard(description_bytes);
            assert_ne!(
                file.len() % 32,
                0,
                "the file ends before its aligned data offset"
            );
            assert!(
                matches!(header_prefix(&file), Ok(HeaderPrefix::Complete(length)) if length > file.len())
            );
            let size = u64::try_from(file.len()).unwrap();
            let (endpoint, _, _) = serve_ranges(vec![("model-00001-of-00006.gguf", file.clone())]);
            let header = fetch_header(
                &reqwest::Client::new(),
                None,
                &endpoint,
                "owner/repo",
                &"a".repeat(40),
                Path::new("model-00001-of-00006.gguf"),
                size,
            )
            .await
            .unwrap();
            assert_eq!(header, file);
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_header_larger_than_the_probe_by_far_is_read_whole() {
        let description = "x".repeat(40 * 1024 * 1024);
        let mut file = crate::test_gguf::gguf_bytes(
            &[
                ("general.architecture", crate::test_gguf::Meta::Str("llama")),
                (
                    "general.description",
                    crate::test_gguf::Meta::Str(&description),
                ),
            ],
            &["token_embd.weight"],
        );
        let header_end = header_length(&file).unwrap();
        file.resize(file.len() + 1024 * 1024, 0);
        let size = u64::try_from(file.len()).unwrap();
        let (endpoint, _, _) = serve_ranges(vec![("model.gguf", file.clone())]);
        let header = fetch_header(
            &reqwest::Client::new(),
            None,
            &endpoint,
            "owner/repo",
            &"a".repeat(40),
            Path::new("model.gguf"),
            size,
        )
        .await
        .unwrap();
        assert_eq!(header, file[..header_end]);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_file_that_ends_inside_its_header_is_reported_as_truncated() {
        let description = "x".repeat(2 * 1024 * 1024);
        let mut file = crate::test_gguf::gguf_bytes(
            &[
                ("general.architecture", crate::test_gguf::Meta::Str("llama")),
                (
                    "general.description",
                    crate::test_gguf::Meta::Str(&description),
                ),
            ],
            &["token_embd.weight"],
        );
        file.truncate(1024 * 1024 + 7);
        let size = u64::try_from(file.len()).unwrap();
        let (endpoint, _, _) = serve_ranges(vec![("model.gguf", file)]);
        let error = fetch_header(
            &reqwest::Client::new(),
            None,
            &endpoint,
            "owner/repo",
            &"a".repeat(40),
            Path::new("model.gguf"),
            size,
        )
        .await
        .unwrap_err();
        assert!(
            matches!(&error, InventoryError::Integrity(detail) if detail == "the file ends before its GGUF header does"),
            "{error:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_gated_repository_refusing_its_files_asks_for_access_and_a_token() {
        use std::io::{BufRead, BufReader, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("gated listener");
        let endpoint = format!("http://{}", listener.local_addr().expect("gated address"));
        std::thread::spawn(move || {
            for connection in listener.incoming() {
                let Ok(mut connection) = connection else {
                    return;
                };
                let mut reader = BufReader::new(connection.try_clone().expect("stream"));
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                        break;
                    }
                }
                let _ = write!(
                    connection,
                    "HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                );
            }
        });
        let error = fetch_header(
            &reqwest::Client::new(),
            None,
            &endpoint,
            "google/gated-model-GGUF",
            &"a".repeat(40),
            Path::new("model.gguf"),
            1024 * 1024 * 1024,
        )
        .await
        .unwrap_err();
        let InventoryError::ModelOperation {
            code,
            message,
            retryable,
        } = error
        else {
            panic!("a refused gated file is a model operation failure: {error:?}");
        };
        assert_eq!(code, "hugging_face_access_required");
        assert!(message.contains("HF_TOKEN"), "{message}");
        assert!(!retryable);
    }

    /// The Hub answers an anonymous metadata request for a repository that doesn't exist with 401,
    /// and one with a token with 404; neither confirms the repository exists, so both say it may
    /// be missing, private or gated.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_refused_repository_says_it_may_be_missing_private_or_gated() {
        use std::io::{BufRead, BufReader, Write};
        for status in ["401 Unauthorized", "404 Not Found"] {
            let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("hub listener");
            let endpoint = format!("http://{}", listener.local_addr().expect("hub address"));
            std::thread::spawn(move || {
                for connection in listener.incoming() {
                    let Ok(mut connection) = connection else {
                        return;
                    };
                    let mut reader = BufReader::new(connection.try_clone().expect("stream"));
                    loop {
                        let mut line = String::new();
                        if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                            break;
                        }
                    }
                    let _ = write!(
                        connection,
                        "HTTP/1.1 {status}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                    );
                }
            });
            let error = match fetch_hub_model(
                &reqwest::Client::new(),
                None,
                &endpoint,
                "nobody-xyz/does-not-exist-GGUF",
                "main",
                None,
            )
            .await
            {
                Err(error) => error,
                Ok(_) => panic!("{status}: a refused repository fails"),
            };
            let InventoryError::ModelOperation {
                code,
                message,
                retryable,
            } = error
            else {
                panic!("{status}: a refused repository is a model operation failure: {error:?}");
            };
            assert_eq!(code, "hugging_face_repository_not_found", "{status}");
            assert_eq!(
                message,
                "Repository not found, or it's private or gated. Check the name, or set HF_TOKEN if you have access."
            );
            assert!(!retryable);
        }
    }

    /// The Linux QA lanes saw one inspect pull 446 MB to 1.1 GB for files with invalid headers,
    /// and every retry pull it again. A header proven invalid stops at the probe that
    /// proves it, and the verdict is cached for the immutable file, so a retry fetches nothing.
    #[tokio::test(flavor = "multi_thread")]
    async fn an_unreadable_header_costs_one_probe_once_and_nothing_on_retry() {
        let mut file = crate::test_gguf::gguf_bytes(
            &[("general.architecture", crate::test_gguf::Meta::Str("llama"))],
            &["token_embd.weight"],
        );
        let mut name = ("token_embd.weight".len() as u64).to_le_bytes().to_vec();
        name.extend(b"token_embd.weight");
        let type_at = file
            .windows(name.len())
            .position(|window| window == name)
            .unwrap()
            + name.len();
        // A rank no GGUF tensor has: the header's structure reads to its end, and the parser
        // refuses it.
        file[type_at..type_at + 4].copy_from_slice(&5_u32.to_le_bytes());
        file.resize(file.len() + 256 * 1024 * 1024, 0);
        let (endpoint, _, sent) = serve_ranges(vec![("model-Q4_0_4_4.gguf", file.clone())]);
        let (_temporary, store) = stub_store(endpoint).await;
        let repository_file = HuggingFaceRepositoryFile {
            path: PathBuf::from("model-Q4_0_4_4.gguf"),
            size_bytes: file.len() as u64,
            content: ContentIdentity::Sha256 {
                value: "c".repeat(64),
            },
        };
        let commit = "a".repeat(40);
        for attempt in 0..2 {
            let error = store
                .hugging_face_gguf_header("owner/repo", &commit, &repository_file, &[])
                .await
                .unwrap_err();
            assert!(
                matches!(&error, InventoryError::Integrity(detail) if detail.contains("invalid or duplicate tensor directory entry")),
                "attempt {attempt}: {error:?}"
            );
        }
        assert_eq!(
            sent.load(std::sync::atomic::Ordering::SeqCst),
            crate::remote_header::INITIAL_READ_BYTES,
            "one probe, and nothing on the retry"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn an_invalid_header_stops_the_read_once_its_entries_are_in_hand() {
        let description = "x".repeat(3 * 1024 * 1024);
        let mut file = crate::test_gguf::gguf_bytes(
            &[
                ("general.architecture", crate::test_gguf::Meta::Str("llama")),
                (
                    "general.description",
                    crate::test_gguf::Meta::Str(&description),
                ),
            ],
            &["token_embd.weight"],
        );
        let mut name = ("token_embd.weight".len() as u64).to_le_bytes().to_vec();
        name.extend(b"token_embd.weight");
        let type_at = file
            .windows(name.len())
            .position(|window| window == name)
            .unwrap()
            + name.len();
        // A rank no GGUF tensor has: the header's structure reads to its end, and the parser
        // refuses it.
        file[type_at..type_at + 4].copy_from_slice(&5_u32.to_le_bytes());
        file.resize(file.len() + 64 * 1024 * 1024, 0);
        let size = u64::try_from(file.len()).unwrap();
        let (endpoint, requests, _) = serve_ranges(vec![("model-Q4_0_4_4.gguf", file)]);
        let error = fetch_header(
            &reqwest::Client::new(),
            None,
            &endpoint,
            "owner/repo",
            &"a".repeat(40),
            Path::new("model-Q4_0_4_4.gguf"),
            size,
        )
        .await
        .unwrap_err();

        assert!(
            matches!(&error, InventoryError::Integrity(detail) if detail.contains("invalid or duplicate tensor directory entry")),
            "{error:?}"
        );
        assert_eq!(
            requests.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "a truncated probe reads on; an invalid header ends the read"
        );
    }

    /// An explicit Hugging Face token is presented as a bearer credential on every Hub request:
    /// search, repository resolution and header reads.
    #[tokio::test(flavor = "multi_thread")]
    async fn the_hf_token_is_sent_on_search_resolve_and_header_reads() {
        use std::io::{BufRead, BufReader, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("hub listener");
        let endpoint = format!("http://{}", listener.local_addr().expect("hub address"));
        let seen = Arc::new(std::sync::Mutex::new(Vec::<(String, Option<String>)>::new()));
        let recorded = Arc::clone(&seen);
        let commit = "a".repeat(40);
        let metadata = revision_metadata("owner/Repo-GGUF", &commit, &[]);
        std::thread::spawn(move || {
            for connection in listener.incoming() {
                let Ok(mut connection) = connection else {
                    return;
                };
                let mut reader = BufReader::new(connection.try_clone().expect("stream"));
                let mut request_line = String::new();
                if reader.read_line(&mut request_line).is_err() {
                    continue;
                }
                let mut authorization = None;
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                        break;
                    }
                    if let Some((name, value)) = line.split_once(':')
                        && name.eq_ignore_ascii_case("authorization")
                    {
                        authorization = Some(value.trim().to_owned());
                    }
                }
                let target = request_line
                    .split_whitespace()
                    .nth(1)
                    .unwrap_or("/")
                    .to_owned();
                let path = target
                    .split_once('?')
                    .map_or(target.as_str(), |(path, _)| path)
                    .to_owned();
                recorded.lock().unwrap().push((path.clone(), authorization));
                let body = if path == "/api/models" {
                    b"[]".to_vec()
                } else if path.starts_with("/api/models/owner/Repo-GGUF/revision/") {
                    serde_json::to_vec(&metadata).unwrap()
                } else {
                    vec![0_u8; 10]
                };
                let _ = write!(
                    connection,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = connection.write_all(&body);
            }
        });
        let temporary = tempfile::tempdir().unwrap();
        let mut config = crate::inventory::InventoryConfig::with_roots(
            temporary.path().join("model-store"),
            temporary.path().join("cache"),
        )
        .unwrap();
        config.local_models.roots.clear();
        config.hub_endpoint = Some(endpoint);
        config.hub_token = Some("test-token".to_owned());
        let store = Arc::new(ManagedModelStore::open(config).await.unwrap());

        let discovery = HuggingFaceDiscovery::new(
            Arc::clone(&store),
            RecognizedArchitectures::new(["llama"], ["dflash"]),
            CatalogLineage::new(
                Arc::clone(&store),
                catalog_with_target("catalog/Llama-GGUF", &commit),
            ),
            StubCatalog::empty(),
        );
        discovery
            .search(HuggingFaceModelSearchRequest {
                query: "repo".to_owned(),
                limit: 5,
            })
            .await
            .expect("search");
        let snapshot = refresh_hugging_face_repository(
            &store,
            HuggingFaceRepositoryRequest {
                repository: "owner/Repo-GGUF".to_owned(),
                revision: "main".to_owned(),
            },
        )
        .await
        .expect("resolve");
        let _ = store
            .hugging_face_gguf_header(
                "owner/Repo-GGUF",
                &snapshot.commit,
                &snapshot.gguf_files[0],
                &[],
            )
            .await;

        let seen = seen.lock().unwrap().clone();
        let kinds = |predicate: fn(&str) -> bool| {
            seen.iter()
                .filter(|(path, _)| predicate(path))
                .map(|(_, authorization)| authorization.clone())
                .collect::<Vec<_>>()
        };
        let bearer = Some("Bearer test-token".to_owned());
        for (kind, requests) in [
            ("search", kinds(|path| path == "/api/models")),
            ("resolve", kinds(|path| path.contains("/revision/"))),
            ("header read", kinds(|path| path.contains("/resolve/"))),
        ] {
            assert!(!requests.is_empty(), "no {kind} request: {seen:?}");
            assert!(
                requests
                    .iter()
                    .all(|authorization| *authorization == bearer),
                "{kind}: {seen:?}"
            );
        }
    }

    async fn stub_store(endpoint: String) -> (tempfile::TempDir, Arc<ManagedModelStore>) {
        let temporary = tempfile::tempdir().unwrap();
        let mut config = crate::inventory::InventoryConfig::with_roots(
            temporary.path().join("model-store"),
            temporary.path().join("cache"),
        )
        .unwrap();
        config.local_models.roots.clear();
        config.hub_endpoint = Some(endpoint);
        let store = Arc::new(ManagedModelStore::open(config).await.unwrap());
        (temporary, store)
    }

    fn catalog_with_target(
        repository: &str,
        commit: &str,
    ) -> magnitude_service_contracts::models::RecommendableModelCatalog {
        use magnitude_service_contracts::models::*;
        let package = ModelPackage {
            id: ModelPackageId("package_catalog".to_owned()),
            source: ModelPackageSource::HuggingFace {
                repository: repository.to_owned(),
                revision: commit.to_owned(),
            },
            files: vec![ModelFile {
                id: ModelFileId("file_catalog".to_owned()),
                path: PathBuf::from("llama-Q8_0.gguf"),
                role: ModelFileRole::Weights,
                size_bytes: 1,
                tensor_storage_bytes: None,
                sha256: "a".repeat(64),
            }],
            relationships: Vec::new(),
            properties: ModelPackageProperties {
                format: "gguf".to_owned(),
                quantization: "Q8_0".to_owned(),
                quantization_name: "8-bit".to_owned(),
                architecture: "llama".to_owned(),
                maximum_context_length: Some(8192),
                intrinsic_model_id: None,
                intrinsic_quality_id: None,
            },
        };
        RecommendableModelCatalog {
            models: vec![RecommendableModel {
                model_id: CatalogBaseId::new("llama-test").unwrap(),
                variant_id: CatalogVariantId::new("gguf:q8").unwrap(),
                configuration: ModelServingConfiguration {
                    bundle: ServableModelBundle::Standalone { package },
                    profile: ServingProfile {
                        context_length: 8192,
                    },
                },
                display_name: "Llama Test".to_owned(),
                variant_label: "Q8".to_owned(),
                description: "A test model".to_owned(),
                release_date: ModelReleaseDate::new("2026-01-01").unwrap(),
                license: "apache-2.0".to_owned(),
                parameterization: ModelParameterization::Dense {
                    total_parameters: 1,
                },
                intelligence: 1,
                support: CatalogSupport::Supported,
                fidelity_rank: 1,
                quantization_aware: false,
            }],
            diagnostics: Vec::new(),
        }
    }

    fn revision_metadata(
        repository: &str,
        commit: &str,
        base_models: &[&str],
    ) -> serde_json::Value {
        serde_json::json!({
            "id": repository,
            "sha": commit,
            "tags": ["gguf"],
            "cardData": { "base_model": base_models },
            "siblings": [{
                "rfilename": "llama-Q8_0.gguf",
                "size": 10,
                "lfs": { "sha256": "b".repeat(64), "size": 10 },
            }],
        })
    }

    fn search_hub() -> (String, Arc<std::sync::atomic::AtomicUsize>) {
        serve_hub(|path, query| {
            let result = |id: &str, seed: char, architecture: &str, base: &[&str]| {
                serde_json::json!({
                    "id": id,
                    "sha": seed.to_string().repeat(40),
                    "downloads": 7,
                    "tags": ["gguf"],
                    "gguf": { "architecture": architecture, "total": 8_030_000_000_u64 },
                    "siblings": [{ "rfilename": "model-Q4_K_M.gguf" }],
                    "baseModels": {
                        "relation": "quantized",
                        "models": base.iter().map(|id| serde_json::json!({ "id": id })).collect::<Vec<_>>(),
                    },
                })
            };
            let speculative = |mut result: serde_json::Value| {
                result["tags"] = serde_json::json!(["gguf", "speculative-decoding"]);
                result
            };
            match path {
                "/api/models" if query.contains("search=") => Some(serde_json::json!([
                    result("catalog/Llama-GGUF", 'a', "llama", &["base/Llama"]),
                    result("unsloth/DeepSeek-V4-GGUF", 'f', "deepseek4", &[]),
                    result("someone/Llama-DFlash-GGUF", 'd', "dflash", &["base/Llama"]),
                    speculative(result(
                        "someone/Llama-EAGLE3-GGUF",
                        'b',
                        "eagle3",
                        &["base/Llama"]
                    )),
                    result("someone/Llama-MTP-GGUF", '9', "llama", &[]),
                    result("vision/Llama-Vision-GGUF", 'e', "clip", &[]),
                    result("community/Llama-Finetune-GGUF", 'c', "llama", &["base/Llama"]),
                    result("anonymous/Llama-Uncensored-GGUF", '7', "llama", &[]),
                ])),
                path if path.starts_with("/api/models/catalog/Llama-GGUF/revision/") => Some(
                    revision_metadata("catalog/Llama-GGUF", &"a".repeat(40), &["base/Llama"]),
                ),
                _ => None,
            }
        })
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn search_lists_unknown_architectures_as_unsupported_and_excludes_the_catalog_and_drafters()
     {
        let (endpoint, _) = search_hub();
        let (_temporary, store) = stub_store(endpoint).await;
        let catalog = catalog_with_target("catalog/Llama-GGUF", &"a".repeat(40));
        let discovery = HuggingFaceDiscovery::new(
            Arc::clone(&store),
            RecognizedArchitectures::new(["llama"], ["dflash"]),
            CatalogLineage::new(Arc::clone(&store), catalog),
            StubCatalog::empty(),
        );
        let results = discovery
            .search(HuggingFaceModelSearchRequest {
                query: "llama".to_owned(),
                limit: 10,
            })
            .await
            .unwrap();
        assert_eq!(
            results
                .results
                .iter()
                .map(|result| result.repository.as_str())
                .collect::<Vec<_>>(),
            [
                "someone/Llama-MTP-GGUF",
                "community/Llama-Finetune-GGUF",
                "anonymous/Llama-Uncensored-GGUF",
                "unsloth/DeepSeek-V4-GGUF",
                "vision/Llama-Vision-GGUF",
            ],
            "a repository whose name says MTP is a model result: names never decide; drafter \
             repositories are never model results"
        );
        assert!(
            results.results[..3]
                .iter()
                .all(|result| result.unsupported.is_empty() && result.variant_of.is_none()),
            "declared lineage never makes a result a variant"
        );
        assert_eq!(
            results.results[4].unsupported,
            [UnsupportedReason::Known(KnownUnsupportedReason::Architecture {
                architecture: "clip".to_owned()
            })],
            "a repository the Hub describes by a projector, whose model file can't be read, is \
             shown as the Hub describes it: not runnable"
        );
        assert_eq!(
            results.results[3].unsupported,
            [UnsupportedReason::Known(
                KnownUnsupportedReason::Architecture {
                    architecture: "deepseek4".to_owned()
                }
            )],
            "an unsupported architecture is listed with its reason"
        );
    }

    /// Each catalog model's fingerprint, supplied directly.
    struct StubCatalog(Vec<(ModelId, ModelFingerprint)>);

    impl StubCatalog {
        fn empty() -> Arc<dyn CatalogFingerprints> {
            Arc::new(Self(Vec::new()))
        }
    }

    impl CatalogFingerprints for StubCatalog {
        fn catalog_fingerprints(&self) -> BoxFuture<'_, Vec<(ModelId, ModelFingerprint)>> {
            Box::pin(async move { self.0.clone() })
        }
    }

    /// One repository of a stub Hub: its search fields and its GGUFs, each with its listed size
    /// and the bytes served for it (`None`: the file refuses reads).
    struct StubRepository {
        id: &'static str,
        architecture: Option<&'static str>,
        gated: bool,
        files: Vec<(&'static str, u64, Option<Vec<u8>>)>,
        listing_429s: usize,
        file_429s: usize,
    }

    impl StubRepository {
        fn new(id: &'static str, architecture: Option<&'static str>) -> Self {
            Self {
                id,
                architecture,
                gated: false,
                files: Vec::new(),
                listing_429s: 0,
                file_429s: 0,
            }
        }

        /// The first `listing` requests for its file listing and the first `files` requests for
        /// its files are refused with HTTP 429 (`Retry-After: 1`), whose body is `body` (a GGUF
        /// header, to prove a refusal is never read as one).
        fn throttled(mut self, listing: usize, files: usize) -> Self {
            self.listing_429s = listing;
            self.file_429s = files;
            self
        }

        fn file(mut self, path: &'static str, size: u64, bytes: Option<Vec<u8>>) -> Self {
            self.files.push((path, size, bytes));
            self
        }

        fn commit(&self) -> String {
            let digest = Sha256::digest(self.id.as_bytes());
            digest.iter().take(20).map(|byte| format!("{byte:02x}")).collect()
        }
    }

    /// A Hub that answers search with `repositories`, their commit-pinned metadata, and range
    /// requests for their files, counting the file requests and the file bytes it serves.
    fn serve_repositories(
        repositories: Vec<StubRepository>,
    ) -> (
        String,
        Arc<std::sync::atomic::AtomicUsize>,
        Arc<std::sync::atomic::AtomicUsize>,
    ) {
        use std::io::{BufRead, BufReader, Write};
        use std::sync::atomic::Ordering;
        let search = serde_json::to_vec(&repositories
            .iter()
            .map(|repository| {
                let mut gguf = serde_json::json!({});
                if let Some(architecture) = repository.architecture {
                    gguf["architecture"] = serde_json::json!(architecture);
                }
                serde_json::json!({
                    "id": repository.id,
                    "sha": repository.commit(),
                    "downloads": 7,
                    "gated": if repository.gated { serde_json::json!("auto") } else { serde_json::json!(false) },
                    "tags": ["gguf"],
                    "gguf": gguf,
                    "siblings": repository.files.iter().map(|(path, ..)| serde_json::json!({ "rfilename": path })).collect::<Vec<_>>(),
                })
            })
            .collect::<Vec<_>>())
        .unwrap();
        let mut routes = BTreeMap::<String, Vec<u8>>::new();
        let mut files = BTreeMap::<String, Vec<u8>>::new();
        let mut throttled = BTreeMap::<String, usize>::new();
        for repository in &repositories {
            let commit = repository.commit();
            let siblings = repository
                .files
                .iter()
                .map(|(path, size, _)| {
                    let digest = Sha256::digest(format!("{}/{path}", repository.id).as_bytes());
                    serde_json::json!({
                        "rfilename": path,
                        "size": size,
                        "lfs": {
                            "sha256": digest.iter().map(|byte| format!("{byte:02x}")).collect::<String>(),
                            "size": size,
                        },
                    })
                })
                .collect::<Vec<_>>();
            throttled.insert(
                format!("/api/models/{}/revision/{commit}", repository.id),
                repository.listing_429s,
            );
            routes.insert(
                format!("/api/models/{}/revision/{commit}", repository.id),
                serde_json::to_vec(&serde_json::json!({
                    "id": repository.id,
                    "sha": commit,
                    "tags": ["gguf"],
                    "siblings": siblings,
                }))
                .unwrap(),
            );
            for (path, _, bytes) in &repository.files {
                throttled.insert(format!("/{}/resolve/{commit}/{path}", repository.id), repository.file_429s);
                if let Some(bytes) = bytes {
                    files.insert(format!("/{}/resolve/{commit}/{path}", repository.id), bytes.clone());
                }
            }
        }
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("hub listener");
        let endpoint = format!("http://{}", listener.local_addr().expect("hub address"));
        let requests = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let sent = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let (counted, sent_counted) = (Arc::clone(&requests), Arc::clone(&sent));
        let shared = Arc::new((search, routes, files, std::sync::Mutex::new(throttled)));
        std::thread::spawn(move || {
            for connection in listener.incoming() {
                let Ok(mut connection) = connection else {
                    return;
                };
                let (shared, counted, sent) = (
                    Arc::clone(&shared),
                    Arc::clone(&counted),
                    Arc::clone(&sent_counted),
                );
                std::thread::spawn(move || {
                    let (search, routes, files, throttled) = &*shared;
                    let mut reader = BufReader::new(connection.try_clone().expect("stream"));
                    let mut request_line = String::new();
                    let _ = reader.read_line(&mut request_line);
                    let target = request_line
                        .split_whitespace()
                        .nth(1)
                        .unwrap_or("/")
                        .to_owned();
                    let path = target.split_once('?').map_or(target.as_str(), |(path, _)| path);
                    let mut range = None;
                    loop {
                        let mut line = String::new();
                        if reader.read_line(&mut line).unwrap_or(0) == 0 || line == "\r\n" {
                            break;
                        }
                        if let Some((header, value)) = line.split_once(':')
                            && header.eq_ignore_ascii_case("range")
                        {
                            let (start, end) = value
                                .trim()
                                .trim_start_matches("bytes=")
                                .split_once('-')
                                .unwrap();
                            range = Some((
                                start.parse::<usize>().unwrap(),
                                end.parse::<usize>().unwrap(),
                            ));
                        }
                    }
                    let json = |connection: &mut std::net::TcpStream, body: &[u8]| {
                        let _ = write!(
                            connection,
                            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            body.len()
                        );
                        let _ = connection.write_all(body);
                    };
                    if path == "/api/models" {
                        return json(&mut connection, search);
                    }
                    if !path.starts_with("/api/") {
                        counted.fetch_add(1, Ordering::SeqCst);
                    }
                    let refused = throttled
                        .lock()
                        .unwrap()
                        .get_mut(path)
                        .is_some_and(|left| *left > 0 && { *left -= 1; true });
                    if refused {
                        // A refusal whose body is a well-formed GGUF header of a projector: read as
                        // file bytes, it would make the repository look like it holds no model.
                        let body = crate::test_gguf::gguf_bytes(
                            &[("general.architecture", crate::test_gguf::Meta::Str("clip"))],
                            &["v.patch_embd.weight"],
                        );
                        let _ = write!(
                            connection,
                            "HTTP/1.1 429 Too Many Requests\r\nRetry-After: 1\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            body.len()
                        );
                        let _ = connection.write_all(&body);
                        return;
                    }
                    if let Some(body) = routes.get(path) {
                        return json(&mut connection, body);
                    }
                    if path.starts_with("/api/") {
                        let _ = write!(connection, "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
                        return;
                    }
                    let Some(file) = files.get(path) else {
                        let _ = write!(connection, "HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
                        return;
                    };
                    let (start, end) = range.unwrap_or((0, file.len() - 1));
                    let end = end.min(file.len() - 1);
                    let body = &file[start..=end];
                    sent.fetch_add(body.len(), Ordering::SeqCst);
                    let _ = write!(
                        connection,
                        "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: bytes {start}-{end}/{}\r\nConnection: close\r\n\r\n",
                        body.len(),
                        file.len()
                    );
                    let _ = connection.write_all(body);
                });
            }
        });
        (endpoint, requests, sent)
    }

    /// A discovery over `store` running `targets` (and `dflash` as a drafter), whose catalog is
    /// `catalog`.
    fn shape_discovery(
        store: &Arc<ManagedModelStore>,
        targets: &[&str],
        catalog: Vec<(ModelId, ModelFingerprint)>,
    ) -> HuggingFaceDiscovery {
        HuggingFaceDiscovery::new(
            Arc::clone(store),
            RecognizedArchitectures::new(targets.iter().copied(), ["dflash"]),
            CatalogLineage::new(
                Arc::clone(store),
                catalog_with_target("catalog/Llama-GGUF", &"a".repeat(40)),
            ),
            Arc::new(StubCatalog(catalog)),
        )
    }

    async fn search_all(discovery: &HuggingFaceDiscovery) -> BTreeMap<String, HuggingFaceModelSearchResult> {
        discovery
            .search(HuggingFaceModelSearchRequest {
                query: "model".to_owned(),
                limit: 10,
            })
            .await
            .unwrap()
            .results
            .into_iter()
            .map(|result| (result.repository.clone(), result))
            .collect()
    }

    fn catalog_fingerprint(fixture: &str) -> ModelFingerprint {
        crate::search_shape::ModelShape::from_header_prefix(&crate::test_gguf::fixture_header(fixture))
            .unwrap()
            .fingerprint()
    }

    /// An importance matrix: quantization data with no architecture.
    fn imatrix() -> Vec<u8> {
        crate::test_gguf::gguf_bytes(
            &[("general.type", crate::test_gguf::Meta::Str("imatrix"))],
            &["blk.0.attn_q.weight.in_sum2"],
        )
    }

    fn architecture_only(architecture: &str) -> Vec<u8> {
        crate::test_gguf::gguf_bytes(
            &[("general.architecture", crate::test_gguf::Meta::Str(architecture))],
            &["token_embd.weight"],
        )
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_result_is_a_variant_of_the_catalog_model_whose_shape_its_header_has() {
        use crate::test_gguf::fixture_header;
        use std::sync::atomic::Ordering;
        let (endpoint, requests, sent) = serve_repositories(vec![
            StubRepository::new("HauhauCS/Gemma-4-E2B-Uncensored-HauhauCS-Aggressive", Some("gemma4"))
                .file("mmproj-Gemma-4-E2B-f16.gguf", 9_000_000_000, None)
                .file(
                    "Gemma-4-E2B-Uncensored-HauhauCS-Aggressive-Q2_K_P.gguf",
                    2_000_000_000,
                    Some(fixture_header("Gemma-4-E2B-Uncensored-HauhauCS-Aggressive-Q2_K_P")),
                ),
            StubRepository::new("someone/Qwen3.5-4B-GGUF", Some("qwen35")).file(
                "Qwen3.5-4B-Q3_K_M.gguf",
                2_000_000_000,
                Some(fixture_header("Qwen3.5-4B-Q3_K_M")),
            ),
            StubRepository::new("broken/Gemma-4-GGUF", Some("gemma4"))
                .file("missing-Q4_K_M.gguf", 2_000_000_000, None),
        ]);
        let (_temporary, store) = stub_store(endpoint).await;
        let gemma: ModelId = "gemma-test:gguf:q4".parse().unwrap();
        let catalog = vec![(gemma.clone(), catalog_fingerprint("gemma-4-E2B-it-qat-UD-Q4_K_XL"))];
        let variants = |results: BTreeMap<String, HuggingFaceModelSearchResult>| {
            results
                .into_iter()
                .map(|(repository, result)| (repository, result.variant_of, result.unsupported.is_empty()))
                .collect::<Vec<_>>()
        };
        let expected = vec![
            ("HauhauCS/Gemma-4-E2B-Uncensored-HauhauCS-Aggressive".to_owned(), Some(gemma.clone()), true),
            ("broken/Gemma-4-GGUF".to_owned(), None, true),
            ("someone/Qwen3.5-4B-GGUF".to_owned(), None, true),
        ];
        let discovery = shape_discovery(&store, &["gemma4", "qwen35"], catalog.clone());
        assert_eq!(
            variants(search_all(&discovery).await),
            expected,
            "a repository declaring no lineage is a variant by its header's shape; an unknown \
             shape and a failed read are variants of none, and a failed read never fails a row the \
             Hub names a runnable architecture for"
        );
        assert_eq!(requests.load(Ordering::SeqCst), 3, "one read per result");
        assert!(
            sent.load(Ordering::SeqCst) <= 2 * crate::search_shape::SHAPE_PREFIX_BYTES,
            "each read is one prefix"
        );

        let sent_before = sent.load(Ordering::SeqCst);
        let discovery = shape_discovery(&store, &["gemma4", "qwen35"], catalog);
        assert_eq!(variants(search_all(&discovery).await), expected);
        assert_eq!(
            requests.load(Ordering::SeqCst),
            4,
            "what a repository holds is cached; only the failed read is tried again"
        );
        assert_eq!(sent.load(Ordering::SeqCst), sent_before);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn the_model_file_is_the_largest_gguf_whose_header_is_a_model() {
        use crate::test_gguf::fixture_header;
        use std::sync::atomic::Ordering;
        let (endpoint, requests, _) = serve_repositories(vec![
            StubRepository::new("someone/Gemma-4-E2B-GGUF", Some("gemma4"))
                .file("mmproj-F16.gguf", 20_000_000_000, None)
                .file("imatrix_unsloth.gguf", 9_000_000_000, Some(imatrix()))
                .file(
                    "MTP/mtp-Gemma-4-E2B-BF16.gguf",
                    8_000_000_000,
                    Some(fixture_header("Qwen3.5-4B-DFlash-Q8_0")),
                )
                .file(
                    "BF16/Gemma-4-E2B-BF16-00001-of-00002.gguf",
                    3_500_000_000,
                    Some(fixture_header("Gemma-4-E2B-Uncensored-HauhauCS-Aggressive-Q2_K_P")),
                )
                .file("BF16/Gemma-4-E2B-BF16-00002-of-00002.gguf", 3_500_000_000, None)
                .file(
                    "Gemma-4-E2B-Q4_K_M.gguf",
                    5_000_000_000,
                    Some(fixture_header("Qwen3.5-4B-Q3_K_M")),
                ),
        ]);
        let (_temporary, store) = stub_store(endpoint).await;
        let gemma: ModelId = "gemma-test:gguf:q4".parse().unwrap();
        let discovery = shape_discovery(
            &store,
            &["gemma4", "qwen35"],
            vec![(gemma.clone(), catalog_fingerprint("gemma-4-E2B-it-qat-UD-Q4_K_XL"))],
        );
        let result = &search_all(&discovery).await["someone/Gemma-4-E2B-GGUF"];
        assert_eq!(
            result.variant_of,
            Some(gemma),
            "the projector is never read, the importance matrix and the drafter are skipped by \
             their headers, and a sharded set counts its total size (7 GB, above the 5 GB file)"
        );
        assert_eq!(result.architecture.as_deref(), Some("gemma4"));
        assert_eq!(requests.load(Ordering::SeqCst), 3);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_missing_hub_architecture_comes_from_the_model_file() {
        use crate::test_gguf::fixture_header;
        let mut refused = StubRepository::new("someone/Refused-GGUF", None)
            .file("Refused-Q4_K_M.gguf", 2_000_000_000, None);
        refused.gated = true;
        let (endpoint, _, _) = serve_repositories(vec![
            StubRepository::new("someone/Qwen3.5-4B-GGUF", None).file(
                "Qwen3.5-4B-Q3_K_M.gguf",
                2_000_000_000,
                Some(fixture_header("Qwen3.5-4B-Q3_K_M")),
            ),
            StubRepository::new("someone/Image-GGUF", None)
                .file("image-Q8_0.gguf", 2_000_000_000, Some(architecture_only("qwen_image"))),
            StubRepository::new("someone/Matrix-GGUF", None)
                .file("imatrix.gguf", 13_000_000, Some(imatrix())),
            StubRepository::new("someone/Unreachable-GGUF", None)
                .file("model-Q4_K_M.gguf", 2_000_000_000, None),
            StubRepository::new("someone/Mistral-7B-GGUF", None)
                .file("Mistral-7B-Q8_0.gguf", 2_000_000_000, Some(fixture_header("Qwen3.5-4B-Q3_K_M")))
                .file("Mistral-7B-bf16.gguf", 0, None),
            refused,
        ]);
        let (_temporary, store) = stub_store(endpoint).await;
        let results = search_all(&shape_discovery(&store, &["qwen35"], Vec::new())).await;
        let state = |repository: &str| {
            let result = &results[repository];
            (result.architecture.clone(), result.unsupported.clone(), result.gated)
        };
        let unreadable = vec![UnsupportedReason::Known(KnownUnsupportedReason::Unreadable)];
        assert_eq!(state("someone/Qwen3.5-4B-GGUF"), (Some("qwen35".to_owned()), Vec::new(), false));
        assert_eq!(
            state("someone/Image-GGUF"),
            (
                Some("qwen_image".to_owned()),
                vec![UnsupportedReason::Known(KnownUnsupportedReason::Architecture {
                    architecture: "qwen_image".to_owned()
                })],
                false
            )
        );
        assert_eq!(
            state("someone/Matrix-GGUF"),
            (None, unreadable, false),
            "headers read that name no model architecture are unreadable"
        );
        assert_eq!(
            state("someone/Unreachable-GGUF"),
            (None, Vec::new(), false),
            "a failed read leaves the row as the Hub describes it"
        );
        assert_eq!(
            state("someone/Mistral-7B-GGUF"),
            (None, Vec::new(), false),
            "a listing that can't be resolved (an empty file in it) is a failed read, not an \
             unreadable repository"
        );
        assert_eq!(
            state("someone/Refused-GGUF"),
            (None, Vec::new(), true),
            "a gated repository whose files are refused stays listed; the client says it needs access"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_near_miss_is_read_from_the_whole_header_where_later_keys_settle_it() {
        use crate::header_layout::{Layout, walk};
        use crate::search_shape::{ModelShape, SHAPE_PREFIX_BYTES};
        use crate::test_gguf::fixture_header;
        let fixture = "Qwen3.5-4B-Q3_K_M";
        let header = fixture_header(fixture);
        // The MTP head count moved after the token list, as tools that append MTP heads write it.
        let Ok(Layout::Complete { entries, .. }) = walk(&header) else {
            panic!("complete header");
        };
        let entry = |key: &str| entries.iter().find(|entry| entry.key == key).unwrap();
        let (nextn, tokens) = (entry("qwen35.nextn_predict_layers"), entry("tokenizer.ggml.tokens"));
        let mut moved = header[..nextn.start].to_vec();
        moved.extend_from_slice(&header[nextn.end..tokens.end]);
        moved.extend_from_slice(&header[nextn.start..nextn.end]);
        moved.extend_from_slice(&header[tokens.end..]);
        let prefix_shape = ModelShape::from_header_prefix(&moved[..SHAPE_PREFIX_BYTES]).unwrap();
        let catalog_shape = ModelShape::from_header_prefix(&header).unwrap();
        assert_ne!(prefix_shape, catalog_shape, "the prefix counts the MTP layer as a block");
        assert_eq!(ModelShape::from_header_prefix(&moved), Some(catalog_shape.clone()));
        let length = moved.len() as u64;
        let (endpoint, _, sent) = serve_repositories(vec![
            StubRepository::new("someone/Qwen3.5-4B-MTP-GGUF", Some("qwen35"))
                .file("Qwen3.5-4B-MTP-Q3_K_M.gguf", length, Some(moved)),
        ]);
        let (_temporary, store) = stub_store(endpoint).await;
        let qwen: ModelId = "qwen-test:gguf:q4".parse().unwrap();
        let results = search_all(&shape_discovery(
            &store,
            &["qwen35"],
            vec![(qwen.clone(), catalog_shape.fingerprint())],
        ))
        .await;
        assert_eq!(results["someone/Qwen3.5-4B-MTP-GGUF"].variant_of, Some(qwen));
        assert!(sent.load(std::sync::atomic::Ordering::SeqCst) > SHAPE_PREFIX_BYTES);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn an_unknown_tensor_type_never_affects_the_shape() {
        use crate::search_shape::{ModelShape, SHAPE_PREFIX_BYTES};
        use crate::test_gguf::fixture_header;
        let fixture = "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-IQ3_XS";
        let mut header = fixture_header(fixture);
        let catalog_shape = ModelShape::from_header_prefix(&header).unwrap();
        // The output head's type becomes one no parser knows, as a new quantization's would be.
        let name = b"output.weight";
        let at = header
            .windows(8 + name.len())
            .rposition(|window| window[..8] == (name.len() as u64).to_le_bytes() && &window[8..] == name)
            .unwrap()
            + 8
            + name.len();
        let rank = u32::from_le_bytes(header[at..at + 4].try_into().unwrap()) as usize;
        let encoding = at + 4 + 8 * rank;
        header[encoding..encoding + 4].copy_from_slice(&200_u32.to_le_bytes());
        assert!(magnitude_artifacts::gguf::read_header(&header).is_err());
        assert_eq!(ModelShape::from_header_prefix(&header[..SHAPE_PREFIX_BYTES]), None);
        let length = header.len() as u64;
        let (endpoint, _, _) = serve_repositories(vec![
            StubRepository::new("someone/Bonsai-27B-GGUF", Some("qwen35"))
                .file("Bonsai-27B-PTQ1_0.gguf", length, Some(header)),
        ]);
        let (_temporary, store) = stub_store(endpoint).await;
        let qwen: ModelId = "qwen-test:gguf:q4".parse().unwrap();
        let results = search_all(&shape_discovery(
            &store,
            &["qwen35"],
            vec![(qwen.clone(), catalog_shape.fingerprint())],
        ))
        .await;
        let result = &results["someone/Bonsai-27B-GGUF"];
        assert_eq!(result.variant_of, Some(qwen), "the shape comes from metadata alone");
        assert!(result.unsupported.is_empty(), "weight support is decided at inspection");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_pasted_hugging_face_link_searches_for_its_repository() {
        let (endpoint, _) = serve_hub(|path, query| {
            (path == "/api/models").then(|| {
                if query.contains("search=unsloth%2Fqwen3.5-9b-gguf&") {
                    serde_json::json!([{
                        "id": "unsloth/Qwen3.5-9B-GGUF",
                        "sha": "a".repeat(40),
                        "downloads": 7,
                        "tags": ["gguf"],
                        "gguf": { "architecture": "qwen35" },
                    }])
                } else {
                    serde_json::json!([])
                }
            })
        });
        let (_temporary, store) = stub_store(endpoint).await;
        let discovery = shape_discovery(&store, &["qwen35"], Vec::new());
        for query in [
            "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF",
            "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/blob/main/Qwen3.5-9B-Q4_K_M.gguf",
            "hf.co/unsloth/Qwen3.5-9B-GGUF",
            "unsloth/Qwen3.5-9B-GGUF",
        ] {
            let results = discovery
                .search(HuggingFaceModelSearchRequest {
                    query: query.to_owned(),
                    limit: 10,
                })
                .await
                .unwrap();
            assert_eq!(
                results
                    .results
                    .iter()
                    .map(|result| result.repository.as_str())
                    .collect::<Vec<_>>(),
                ["unsloth/Qwen3.5-9B-GGUF"],
                "{query}"
            );
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn repositories_without_a_model_file_are_left_out_or_unreadable() {
        let piece = crate::test_gguf::gguf_bytes(
            &[
                ("general.architecture", crate::test_gguf::Meta::Str("llama")),
                ("llama.block_count", crate::test_gguf::Meta::U32(32)),
            ],
            &["output.weight"],
        );
        let head = crate::test_gguf::gguf_bytes(
            &[
                ("general.architecture", crate::test_gguf::Meta::Str("qwen4exp")),
                ("qwen4exp.block_count", crate::test_gguf::Meta::U32(49)),
                ("qwen4exp.nextn_predict_layers", crate::test_gguf::Meta::U32(1)),
            ],
            &["blk.48.nextn.eh_proj.weight"],
        );
        let (endpoint, requests, _) = serve_repositories(vec![
            StubRepository::new("someone/Roleplay-1B", None),
            StubRepository::new("paperniuk/Qwen3.8-Flash-Next-MTP-block", Some("qwen4exp"))
                .file("Qwen3.8-Flash-Next-MTP-block.gguf", 1_519_036_352, Some(head)),
            StubRepository::new("meshllm/Llama-8B-layers", Some("llama"))
                .file("projectors/projector-00000.gguf", 3_000_000_000, Some(architecture_only("clip")))
                .file("shared/output.gguf", 900_000_000, Some(piece.clone()))
                .file("layers/layer-00000.gguf", 280_000_000, Some(piece.clone())),
            // A split set's first shard holds only some of the set's tensors, fewer than its
            // blocks too, and is still the model.
            StubRepository::new("someone/Llama-405B-GGUF", Some("llama"))
                .file("BF16/Llama-405B-BF16-00001-of-00002.gguf", 50_000_000_000, Some(piece))
                .file("BF16/Llama-405B-BF16-00002-of-00002.gguf", 50_000_000_000, None),
        ]);
        let (_temporary, store) = stub_store(endpoint).await;
        let results = search_all(&shape_discovery(&store, &["llama", "qwen4exp"], Vec::new())).await;
        assert!(
            !results.contains_key("someone/Roleplay-1B"),
            "a repository holding no GGUF at all is no result: nothing in it can be added"
        );
        assert_eq!(
            results["meshllm/Llama-8B-layers"].unsupported,
            [UnsupportedReason::Known(KnownUnsupportedReason::Unreadable)],
            "pieces of a layer-split model repeat its metadata over fewer tensors than its blocks: \
             no file holds the model"
        );
        assert_eq!(
            results["someone/Llama-405B-GGUF"].unsupported,
            Vec::new(),
            "a split set's shards are never taken for layer-split pieces"
        );
        assert_eq!(results["someone/Llama-405B-GGUF"].architecture.as_deref(), Some("llama"));
        assert!(
            !results.contains_key("paperniuk/Qwen3.8-Flash-Next-MTP-block"),
            "a prediction head alone (prediction layers over fewer tensors than blocks) is a \
             drafter repository, offered only in drafter lists"
        );
        assert_eq!(requests.load(std::sync::atomic::Ordering::SeqCst), 5);
    }

    /// A rate limit (HTTP 429) on a file listing or a header read is unknown, never a verdict: the
    /// row stays as the Hub's search describes it, its refusal body is never read as a GGUF, the
    /// outcome isn't cached, and no further request goes to the Hub until its `Retry-After` passes.
    #[tokio::test(flavor = "multi_thread")]
    async fn a_rate_limit_leaves_rows_as_the_hub_describes_them_and_is_not_cached() {
        use crate::test_gguf::fixture_header;
        use std::sync::atomic::Ordering;
        let gemma = || Some(fixture_header("Gemma-4-E2B-Uncensored-HauhauCS-Aggressive-Q2_K_P"));
        let (endpoint, requests, _) = serve_repositories(vec![
            StubRepository::new("someone/Listing-Refused-GGUF", Some("gemma4"))
                .file("model-Q4_K_M.gguf", 2_000_000_000, gemma())
                .throttled(1, 0),
            StubRepository::new("someone/Header-Refused-GGUF", Some("gemma4"))
                .file("model-Q4_K_M.gguf", 2_000_000_000, gemma())
                .throttled(0, 1),
        ]);
        let (_temporary, store) = stub_store(endpoint).await;
        let catalog = vec![(
            "gemma-test:gguf:q4".parse::<ModelId>().unwrap(),
            catalog_fingerprint("gemma-4-E2B-it-qat-UD-Q4_K_XL"),
        )];
        let state = |results: &BTreeMap<String, HuggingFaceModelSearchResult>, repository: &str| {
            let result = &results[repository];
            (result.architecture.clone(), result.unsupported.clone(), result.variant_of.clone())
        };
        let plain = (Some("gemma4".to_owned()), Vec::new(), None);
        let results = search_all(&shape_discovery(&store, &["gemma4"], catalog.clone())).await;
        assert_eq!(state(&results, "someone/Listing-Refused-GGUF"), plain, "a refused listing keeps the row");
        assert_eq!(
            state(&results, "someone/Header-Refused-GGUF"),
            plain,
            "a refused header read keeps the row, and its body is never read as a projector's header"
        );
        assert!(
            requests.load(Ordering::SeqCst) <= 2,
            "after the first refusal no request goes out until Retry-After passes"
        );

        // A refusal paused the host, so a read the pause stopped was never sent; its stub refusal
        // may still be pending and pause the next search once more. Once each refusal is spent,
        // every read happens: none was cached.
        let mut variants = Vec::new();
        for _ in 0..3 {
            tokio::time::sleep(Duration::from_millis(1_100)).await;
            let results = search_all(&shape_discovery(&store, &["gemma4"], catalog.clone())).await;
            variants = ["someone/Listing-Refused-GGUF", "someone/Header-Refused-GGUF"]
                .map(|repository| state(&results, repository).2)
                .to_vec();
            if variants.iter().all(Option::is_some) {
                break;
            }
        }
        assert_eq!(
            variants,
            vec![Some("gemma-test:gguf:q4".parse::<ModelId>().unwrap()); 2],
            "a refusal was never cached, so each read happens once allowed"
        );
    }

    /// The listing shapes of real large repositories: shard sets in subfolders whose first shard
    /// holds the metadata and only a few tensors (AtomicChat's Qwen3.8 Flash Next sets, 693 MB of
    /// 85 GB), a 19-shard set at the root (a 405B llama), and large single files beside them
    /// (projectors, an importance matrix). Each set's first shard is the model.
    #[tokio::test(flavor = "multi_thread")]
    async fn shard_sets_with_a_metadata_first_shard_are_the_model() {
        let metadata_shard = |architecture: &'static str| {
            crate::test_gguf::gguf_bytes(
                &[
                    ("general.architecture", crate::test_gguf::Meta::Str(architecture)),
                    ("llama.block_count", crate::test_gguf::Meta::U32(126)),
                    ("qwen4exp.block_count", crate::test_gguf::Meta::U32(48)),
                ],
                &["token_embd.weight"],
            )
        };
        let mut flash = StubRepository::new("AtomicChat/Qwen3.8-Flash-Next-GGUF", Some("qwen4exp"))
            .file("mmproj-Qwen3.8-Flash-Next-BF16.gguf", 907_542_784, Some(architecture_only("clip")))
            .file("imatrix.gguf", 580_038_688, Some(imatrix()));
        const FLASH: [&str; 28] = [
            "Q4/Flash-Q4-00001-of-00028.gguf", "Q4/Flash-Q4-00002-of-00028.gguf", "Q4/Flash-Q4-00003-of-00028.gguf",
            "Q4/Flash-Q4-00004-of-00028.gguf", "Q4/Flash-Q4-00005-of-00028.gguf", "Q4/Flash-Q4-00006-of-00028.gguf",
            "Q4/Flash-Q4-00007-of-00028.gguf", "Q4/Flash-Q4-00008-of-00028.gguf", "Q4/Flash-Q4-00009-of-00028.gguf",
            "Q4/Flash-Q4-00010-of-00028.gguf", "Q4/Flash-Q4-00011-of-00028.gguf", "Q4/Flash-Q4-00012-of-00028.gguf",
            "Q4/Flash-Q4-00013-of-00028.gguf", "Q4/Flash-Q4-00014-of-00028.gguf", "Q4/Flash-Q4-00015-of-00028.gguf",
            "Q4/Flash-Q4-00016-of-00028.gguf", "Q4/Flash-Q4-00017-of-00028.gguf", "Q4/Flash-Q4-00018-of-00028.gguf",
            "Q4/Flash-Q4-00019-of-00028.gguf", "Q4/Flash-Q4-00020-of-00028.gguf", "Q4/Flash-Q4-00021-of-00028.gguf",
            "Q4/Flash-Q4-00022-of-00028.gguf", "Q4/Flash-Q4-00023-of-00028.gguf", "Q4/Flash-Q4-00024-of-00028.gguf",
            "Q4/Flash-Q4-00025-of-00028.gguf", "Q4/Flash-Q4-00026-of-00028.gguf", "Q4/Flash-Q4-00027-of-00028.gguf",
            "Q4/Flash-Q4-00028-of-00028.gguf",
        ];
        for (index, path) in FLASH.iter().enumerate() {
            flash = if index == 0 {
                flash.file(path, 693_000_000, Some(metadata_shard("qwen4exp")))
            } else {
                flash.file(path, 3_000_000_000, None)
            };
        }
        let mut llama = StubRepository::new("etemiz/Llama-3.1-405B-Inst-GGUF", Some("llama"));
        const LLAMA: [&str; 19] = [
            "llama-3.1-405b-IQ1_M-00001-of-00019.gguf", "llama-3.1-405b-IQ1_M-00002-of-00019.gguf",
            "llama-3.1-405b-IQ1_M-00003-of-00019.gguf", "llama-3.1-405b-IQ1_M-00004-of-00019.gguf",
            "llama-3.1-405b-IQ1_M-00005-of-00019.gguf", "llama-3.1-405b-IQ1_M-00006-of-00019.gguf",
            "llama-3.1-405b-IQ1_M-00007-of-00019.gguf", "llama-3.1-405b-IQ1_M-00008-of-00019.gguf",
            "llama-3.1-405b-IQ1_M-00009-of-00019.gguf", "llama-3.1-405b-IQ1_M-00010-of-00019.gguf",
            "llama-3.1-405b-IQ1_M-00011-of-00019.gguf", "llama-3.1-405b-IQ1_M-00012-of-00019.gguf",
            "llama-3.1-405b-IQ1_M-00013-of-00019.gguf", "llama-3.1-405b-IQ1_M-00014-of-00019.gguf",
            "llama-3.1-405b-IQ1_M-00015-of-00019.gguf", "llama-3.1-405b-IQ1_M-00016-of-00019.gguf",
            "llama-3.1-405b-IQ1_M-00017-of-00019.gguf", "llama-3.1-405b-IQ1_M-00018-of-00019.gguf",
            "llama-3.1-405b-IQ1_M-00019-of-00019.gguf",
        ];
        for (index, path) in LLAMA.iter().enumerate() {
            llama = if index == 0 {
                llama.file(path, 5_000_000_000, Some(metadata_shard("llama")))
            } else {
                llama.file(path, 5_000_000_000, None)
            };
        }
        let (endpoint, requests, _) = serve_repositories(vec![flash, llama]);
        let (_temporary, store) = stub_store(endpoint).await;
        let results = search_all(&shape_discovery(&store, &["qwen4exp", "llama"], Vec::new())).await;
        for (repository, architecture) in [
            ("AtomicChat/Qwen3.8-Flash-Next-GGUF", "qwen4exp"),
            ("etemiz/Llama-3.1-405B-Inst-GGUF", "llama"),
        ] {
            assert_eq!(results[repository].unsupported, Vec::new(), "{repository}");
            assert_eq!(results[repository].architecture.as_deref(), Some(architecture));
        }
        assert_eq!(
            requests.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "each set's first shard is read once, the largest group first"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn shape_keys_after_the_token_list_are_read_from_the_whole_header() {
        use crate::search_shape::{ModelShape, SHAPE_PREFIX_BYTES};
        use crate::test_gguf::fixture_header;
        use std::sync::atomic::Ordering;
        let fixture = "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-IQ3_XS";
        let header = fixture_header(fixture);
        assert_eq!(
            ModelShape::from_header_prefix(&header[..SHAPE_PREFIX_BYTES]),
            None,
            "this writer puts the embedding length and head counts after the token list"
        );
        let header_length = header.len();
        let (endpoint, requests, sent) = serve_repositories(vec![
            StubRepository::new("HauhauCS/Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-MTP-GGUF", Some("qwen35"))
                .file(
                    "Qwen3.8-27B-Uncensored-HauhauCS-Aggressive-IQ3_XS.gguf",
                    header_length as u64,
                    Some(header),
                ),
        ]);
        let (_temporary, store) = stub_store(endpoint).await;
        let qwen: ModelId = "qwen-test:gguf:q4".parse().unwrap();
        let catalog = vec![(qwen.clone(), catalog_fingerprint(fixture))];
        let variant = || async {
            search_all(&shape_discovery(&store, &["qwen35"], catalog.clone()))
                .await
                .into_values()
                .next()
                .unwrap()
                .variant_of
        };
        assert_eq!(variant().await, Some(qwen.clone()));
        let (read_requests, read_bytes) =
            (requests.load(Ordering::SeqCst), sent.load(Ordering::SeqCst));
        assert!(
            read_bytes <= header_length + 2 * SHAPE_PREFIX_BYTES,
            "the fallback reads header bytes only: {read_bytes} for {header_length}"
        );
        assert_eq!(variant().await, Some(qwen));
        assert_eq!(
            requests.load(Ordering::SeqCst),
            read_requests,
            "a shape read from the whole header is cached like any other"
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn search_lists_exact_names_first_then_by_downloads_and_unsupported_last() {
        let (endpoint, _) = serve_hub(|path, query| {
            let result = |id: &str, seed: char, architecture: &str, downloads: u64| {
                serde_json::json!({
                    "id": id,
                    "sha": seed.to_string().repeat(40),
                    "downloads": downloads,
                    "tags": ["gguf"],
                    "gguf": { "architecture": architecture },
                })
            };
            (path == "/api/models" && query.contains("search=")).then(|| {
                serde_json::json!([
                    result("unsupported/Llama-8B-GGUF", 'a', "deepseek4", 5_000),
                    result("someone/Llama-8B-Uncensored-GGUF", 'b', "llama", 900),
                    result("community/Llama-8B-GGUF", 'c', "llama", 100),
                    result("bartowski/Meta_Llama-8B-GGUF", 'd', "llama", 50),
                    result("other/Llama-8B-Instruct-GGUF", 'e', "llama", 70),
                ])
            })
        });
        let (_temporary, store) = stub_store(endpoint).await;
        let discovery = HuggingFaceDiscovery::new(
            Arc::clone(&store),
            RecognizedArchitectures::new(["llama"], ["dflash"]),
            CatalogLineage::new(
                Arc::clone(&store),
                catalog_with_target("catalog/Llama-GGUF", &"a".repeat(40)),
            ),
            StubCatalog::empty(),
        );
        let results = discovery
            .search(HuggingFaceModelSearchRequest {
                query: "llama 8b".to_owned(),
                limit: 10,
            })
            .await
            .unwrap();
        assert_eq!(
            results
                .results
                .iter()
                .map(|result| result.repository.as_str())
                .collect::<Vec<_>>(),
            [
                "community/Llama-8B-GGUF",
                "bartowski/Meta_Llama-8B-GGUF",
                "someone/Llama-8B-Uncensored-GGUF",
                "other/Llama-8B-Instruct-GGUF",
                "unsupported/Llama-8B-GGUF",
            ]
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn an_undecodable_cached_snapshot_is_a_miss_and_a_pinned_one_needs_no_request() {
        let (endpoint, requests) = search_hub();
        let (_temporary, store) = stub_store(endpoint).await;
        let request = HuggingFaceRepositoryRequest {
            repository: "catalog/Llama-GGUF".to_owned(),
            revision: "a".repeat(40),
        };
        let key = repository_cache_key(store.client.endpoint(), &request);
        store.cache.write_index(
            ModelIndexKind::HuggingFaceRepositorySnapshot,
            &key,
            &serde_json::json!({
                "captured_at": 1,
                "snapshot": {
                    "repository": "catalog/Llama-GGUF",
                    "commit": "a".repeat(40),
                    "license_url": null,
                    "base_models": [],
                    "gguf_files": [],
                },
            }),
        );
        let snapshot = pinned_hugging_face_repository(&store, request.clone())
            .await
            .expect("an old cache entry is refetched");
        assert_eq!(snapshot.base_models, ["base/Llama"]);
        let observed = requests.load(std::sync::atomic::Ordering::SeqCst);
        pinned_hugging_face_repository(&store, request)
            .await
            .unwrap();
        assert_eq!(requests.load(std::sync::atomic::Ordering::SeqCst), observed);
    }

    #[tokio::test]
    async fn an_undecodable_cached_assessment_is_a_miss() {
        let temporary = tempfile::tempdir().unwrap();
        let mut config = crate::inventory::InventoryConfig::with_roots(
            temporary.path().join("model-store"),
            temporary.path().join("cache"),
        )
        .unwrap();
        config.local_models.roots.clear();
        let store = ManagedModelStore::open(config).await.unwrap();
        store.cache.write_index(
            ModelIndexKind::ModelAssessment,
            "old evidence",
            &serde_json::json!({
                "capabilities": {
                    "vision": false,
                    "tools": false,
                    "structuredOutput": false,
                    "reasoning": { "supported": false, "efforts": [] },
                },
                "template_fingerprint": "",
                "profile": {
                    "_tag": "Unsupported",
                    "profile": { "contextLength": 8192 },
                    "failure": { "code": "unsupported", "message": "old", "retryable": false },
                },
            }),
        );
        assert!(store.read_model_assessment("old evidence").is_none());
        store.cache.write_index(
            ModelIndexKind::ModelAssessment,
            "fits without a drafter verdict",
            &serde_json::json!({
                "capabilities": {
                    "vision": false,
                    "tools": false,
                    "structuredOutput": false,
                    "reasoning": { "supported": false, "efforts": [] },
                },
                "template_fingerprint": "",
                "profile": {
                    "_tag": "Fits",
                    "profile": { "contextLength": 8192 },
                    "assessmentId": "assessment_old",
                    "memory": [],
                    "performance": [],
                },
            }),
        );
        assert!(
            store
                .read_model_assessment("fits without a drafter verdict")
                .is_none()
        );
    }
}
