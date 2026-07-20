use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Weak};

use icn_contracts::{
    ComponentRelationship, ComponentRole, ContentIdentity, HardwareProvider, Integrity,
    InventoryError, ModelComponent, ModelHardwareAssessor, ModelId, ModelLocation, ModelPreview,
    ModelPreviewAssessment, ModelPreviewRequest, ModelPreviewSource, ModelPreviewer, ModelSource,
    ResolvedComponent, ResolvedModel,
};
use reqwest::header::{CONTENT_RANGE, RANGE};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::cache::{ModelBlobKind, ModelCacheWorkspace, ModelIndexKind};
use crate::identity::content_id;
use crate::inventory::{ModelManager, build_model, now};

const INITIAL_HEADER_BYTES: usize = 1024 * 1024;
const MAX_HEADER_BYTES: usize = 128 * 1024 * 1024;
const MAX_HUB_METADATA_BYTES: usize = 16 * 1024 * 1024;
const MAX_HUB_SIBLINGS: usize = 100_000;
const MAX_MODEL_COMPONENTS: u32 = 256;
const MAX_COMPONENT_LOGICAL_BYTES: u64 = 1024 * 1024 * 1024 * 1024;
const MAX_PREVIEW_CONTEXT_TOKENS: u32 = 16 * 1024 * 1024;
const MAX_PREVIEW_PARALLEL_SEQUENCES: u32 = 64;

pub struct ModelPreviewService {
    models: Arc<ModelManager>,
    assessor: Arc<dyn ModelHardwareAssessor>,
    work_gates: tokio::sync::Mutex<BTreeMap<String, Weak<tokio::sync::Mutex<()>>>>,
}

impl ModelPreviewService {
    #[must_use]
    pub fn new(models: Arc<ModelManager>, assessor: Arc<dyn ModelHardwareAssessor>) -> Self {
        Self {
            models,
            assessor,
            work_gates: tokio::sync::Mutex::new(BTreeMap::new()),
        }
    }

    async fn work_gate(&self, key: &str) -> Arc<tokio::sync::Mutex<()>> {
        let mut gates = self.work_gates.lock().await;
        gates.retain(|_, gate| gate.strong_count() > 0);
        if let Some(gate) = gates.get(key).and_then(Weak::upgrade) {
            return gate;
        }
        let gate = Arc::new(tokio::sync::Mutex::new(()));
        gates.insert(key.to_owned(), Arc::downgrade(&gate));
        gate
    }
}

impl ModelPreviewer for ModelPreviewService {
    fn preview(
        &self,
        request: ModelPreviewRequest,
    ) -> futures_util::future::BoxFuture<'_, Result<ModelPreview, InventoryError>> {
        Box::pin(async move {
            validate_preview_request(&request)?;
            let source_key = serde_json::to_string(&request.source)
                .map_err(|error| InventoryError::Internal(error.to_string()))?;
            let artifact_gate = self.work_gate(&format!("artifact:{source_key}")).await;
            let prepared = {
                let _guard = artifact_gate.lock().await;
                self.models.prepare_preview(&request.source).await?
            };
            let snapshot = HardwareProvider::snapshot(self.assessor.as_ref()).await?;
            let mut assessments = Vec::with_capacity(request.profiles.len());
            for profile in &request.profiles {
                let hardware_key = self.assessor.cache_key(Some(profile), &snapshot)?;
                let content_id = &prepared.model.model.content_id;
                let assessment = match self
                    .models
                    .cache
                    .read_hardware_assessment(content_id, &hardware_key)
                {
                    Some(assessment) => assessment,
                    None => {
                        let gate = self
                            .work_gate(&format!("assessment:{}:{hardware_key}", content_id.0))
                            .await;
                        let _guard = gate.lock().await;
                        match self
                            .models
                            .cache
                            .read_hardware_assessment(content_id, &hardware_key)
                        {
                            Some(assessment) => assessment,
                            None => {
                                let assessment = self
                                    .assessor
                                    .assess_profile(prepared.model.clone(), Some(profile.clone()))
                                    .await?;
                                self.models.cache.write_hardware_assessment(
                                    content_id,
                                    &hardware_key,
                                    &assessment,
                                );
                                assessment
                            }
                        }
                    }
                };
                assessments.push(ModelPreviewAssessment {
                    profile_id: profile.id.clone(),
                    artifact_fingerprint: prepared.artifact_fingerprint.clone(),
                    execution_policy: self.assessor.policy_identity().to_owned(),
                    hardware_topology: snapshot.topology_fingerprint.clone(),
                    assessment,
                });
            }
            Ok(ModelPreview {
                repository: prepared.repository,
                commit: prepared.commit,
                components: prepared.components,
                properties: prepared.model.model.properties,
                assessments,
            })
        })
    }
}

fn validate_preview_request(request: &ModelPreviewRequest) -> Result<(), InventoryError> {
    if request.profiles.is_empty() || request.profiles.len() > 16 {
        return Err(InventoryError::InvalidRequest(
            "preview requires between one and sixteen execution profiles".to_owned(),
        ));
    }
    if request.profiles.iter().any(|profile| {
        profile.id.is_empty()
            || profile.policy.is_empty()
            || profile.context_length == 0
            || profile.context_length > MAX_PREVIEW_CONTEXT_TOKENS
            || profile.parallel_sequences == 0
            || profile.parallel_sequences > MAX_PREVIEW_PARALLEL_SEQUENCES
    }) {
        return Err(InventoryError::InvalidRequest(
            "preview profiles require IDs, versioned policies, context, and parallelism".to_owned(),
        ));
    }
    let mut ids = std::collections::BTreeSet::new();
    if request
        .profiles
        .iter()
        .any(|profile| !ids.insert(&profile.id))
    {
        return Err(InventoryError::InvalidRequest(
            "preview profile IDs must be unique within one request".to_owned(),
        ));
    }
    Ok(())
}

pub struct PreparedPreview {
    pub model: ResolvedModel,
    pub repository: String,
    pub commit: String,
    pub components: Vec<ModelComponent>,
    pub artifact_fingerprint: String,
    _workspace: ModelCacheWorkspace,
}

#[derive(Debug, Serialize, Deserialize)]
struct CachedArtifact {
    repository: String,
    commit: String,
    primary_gguf: PathBuf,
    components: Vec<CachedComponent>,
}

#[derive(Debug, Serialize, Deserialize)]
struct CachedComponent {
    component: ModelComponent,
    header_digest: String,
    #[serde(skip)]
    acquired_header: Option<Vec<u8>>,
}

#[derive(Debug, Deserialize)]
struct HubModel {
    sha: Option<String>,
    #[serde(default)]
    siblings: Vec<HubSibling>,
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

struct SelectedComponent<'a> {
    sibling: &'a HubSibling,
    role: ComponentRole,
    shard_index: Option<u32>,
    relationship: Option<ComponentRelationship>,
}

impl ModelManager {
    pub async fn prepare_preview(
        &self,
        source: &ModelPreviewSource,
    ) -> Result<PreparedPreview, InventoryError> {
        validate_source(source)?;
        let evidence = serde_json::to_string(source)
            .map_err(|error| InventoryError::Internal(error.to_string()))?;
        let cached = self
            .cache
            .read_index::<CachedArtifact>(ModelIndexKind::Artifact, &evidence)
            .filter(|artifact| artifact_matches_source(artifact, source))
            .filter(|artifact| {
                artifact.components.iter().all(|component| {
                    self.cache
                        .read_blob(ModelBlobKind::GgufHeader, &component.header_digest)
                        .is_some()
                })
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

    async fn acquire_preview_artifact(
        &self,
        source: &ModelPreviewSource,
    ) -> Result<CachedArtifact, InventoryError> {
        let http = reqwest::Client::builder()
            .build()
            .map_err(|error| InventoryError::Upstream(error.to_string()))?;
        let url = format!(
            "{}/api/models/{}",
            self.client.endpoint(),
            source.repository
        );
        let mut request = http
            .get(url)
            .query(&[("revision", source.revision.as_str()), ("blobs", "true")]);
        if let Some(token) = hub_token() {
            request = request.bearer_auth(token);
        }
        let response = request
            .send()
            .await
            .map_err(|error| InventoryError::Upstream(error.to_string()))?;
        if !response.status().is_success() {
            return Err(InventoryError::Upstream(format!(
                "Hugging Face metadata returned HTTP {}",
                response.status()
            )));
        }
        let metadata: HubModel = serde_json::from_slice(
            &bounded_response_bytes(response, MAX_HUB_METADATA_BYTES).await?,
        )
        .map_err(|error| InventoryError::Upstream(error.to_string()))?;
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
                &http,
                self.client.endpoint(),
                &source.repository,
                &commit,
                &component.path,
                size,
            )
            .await?;
            let header_digest = format!("{:x}", Sha256::digest(&header));
            self.cache
                .write_blob(ModelBlobKind::GgufHeader, &header_digest, &header);
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
                .read_blob(ModelBlobKind::GgufHeader, &cached.header_digest)
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
                .and_then(|()| file.set_len(cached.component.size_bytes))
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
        let id = ModelId(format!(
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
            self.template_assessor.as_deref(),
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
            artifact_fingerprint: source_fingerprint,
            _workspace: workspace,
        })
    }
}

fn validate_source(source: &ModelPreviewSource) -> Result<(), InventoryError> {
    let mut repository_parts = source.repository.split('/');
    let valid_repository = repository_parts.next().is_some_and(|part| !part.is_empty())
        && repository_parts.next().is_some_and(|part| !part.is_empty())
        && repository_parts.next().is_none();
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
                && matches!(
                    component.role,
                    ComponentRole::Projector | ComponentRole::Draft | ComponentRole::Mtp
                )
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
                && cached.component.role == expected.role
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
    let primary_name = primary.to_string_lossy();
    if let Some((prefix, total)) = split_parts(primary) {
        let parent = primary.parent().unwrap_or_else(|| Path::new(""));
        return (1..=total)
            .map(|index| {
                let path = parent.join(format!("{prefix}-{index:05}-of-{total:05}.gguf"));
                let name = path.to_string_lossy();
                siblings
                    .iter()
                    .find(|sibling| sibling.rfilename == name)
                    .ok_or_else(|| {
                        InventoryError::Integrity(format!("missing preview shard {name}"))
                    })
            })
            .collect();
    }
    siblings
        .iter()
        .find(|sibling| sibling.rfilename == primary_name)
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
            role: additional.role.clone(),
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
    component: &icn_contracts::ModelPreviewComponentSource,
    primary: &Path,
) -> ComponentRelationship {
    match component.role {
        ComponentRole::Projector => ComponentRelationship::ProjectorFor {
            projector: component.path.clone(),
            model: primary.to_path_buf(),
        },
        ComponentRole::Draft => ComponentRelationship::DraftFor {
            draft: component.path.clone(),
            model: primary.to_path_buf(),
        },
        ComponentRole::Mtp => ComponentRelationship::MtpFor {
            mtp: component.path.clone(),
            model: primary.to_path_buf(),
        },
        _ => unreachable!("preview source validation restricts auxiliary roles"),
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

async fn bounded_response_bytes(
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

async fn fetch_header(
    http: &reqwest::Client,
    endpoint: &str,
    repository: &str,
    commit: &str,
    path: &Path,
    logical_size: u64,
) -> Result<Vec<u8>, InventoryError> {
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

    let mut requested =
        INITIAL_HEADER_BYTES.min(usize::try_from(logical_size).unwrap_or(usize::MAX));
    loop {
        let mut request = http
            .get(url.clone())
            .header(RANGE, format!("bytes=0-{}", requested.saturating_sub(1)));
        if let Some(token) = hub_token() {
            request = request.bearer_auth(token);
        }
        let mut response = request
            .send()
            .await
            .map_err(|error| InventoryError::Upstream(error.to_string()))?;
        if !response.status().is_success() {
            return Err(InventoryError::Upstream(format!(
                "GGUF header request returned HTTP {}",
                response.status()
            )));
        }
        let expected = u64::try_from(requested)
            .map_err(|_| InventoryError::Internal("range length overflow".to_owned()))?;
        let complete_response = expected == logical_size && response.status().as_u16() == 200;
        let valid_partial = response.status().as_u16() == 206
            && response
                .headers()
                .get(CONTENT_RANGE)
                .and_then(|value| value.to_str().ok())
                .is_some_and(|value| valid_content_range(value, expected, logical_size));
        if !complete_response && !valid_partial {
            return Err(InventoryError::Upstream(
                "GGUF source did not honor the exact bounded range request".to_owned(),
            ));
        }
        if response.content_length() != Some(expected) {
            return Err(InventoryError::Upstream(
                "GGUF header response length did not match its content range".to_owned(),
            ));
        }
        let mut bytes = Vec::with_capacity(requested);
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|error| InventoryError::Upstream(error.to_string()))?
        {
            if bytes.len().saturating_add(chunk.len()) > requested {
                return Err(InventoryError::Upstream(
                    "GGUF header response exceeded its requested range".to_owned(),
                ));
            }
            bytes.extend_from_slice(&chunk);
        }
        let temporary = tempfile::NamedTempFile::new()
            .map_err(|error| InventoryError::Io(error.to_string()))?;
        fs::write(temporary.path(), &bytes)
            .map_err(|error| InventoryError::Io(error.to_string()))?;
        match crate::gguf::inspect(temporary.path()) {
            Ok(inspection) => {
                let header_length = usize::try_from(inspection.header_bytes).map_err(|_| {
                    InventoryError::Integrity("GGUF header is too large".to_owned())
                })?;
                if header_length > bytes.len() {
                    return Err(InventoryError::Integrity(
                        "GGUF header response ended before its aligned data offset".to_owned(),
                    ));
                }
                bytes.truncate(header_length);
                return Ok(bytes);
            }
            Err(_)
                if requested < MAX_HEADER_BYTES
                    && u64::try_from(requested).is_ok_and(|value| value < logical_size) =>
            {
                requested = requested
                    .saturating_mul(2)
                    .min(MAX_HEADER_BYTES)
                    .min(usize::try_from(logical_size).unwrap_or(usize::MAX));
            }
            Err(error) => return Err(InventoryError::Integrity(error.to_string())),
        }
    }
}

fn valid_content_range(value: &str, expected_bytes: u64, logical_size: u64) -> bool {
    let Some(value) = value.strip_prefix("bytes ") else {
        return false;
    };
    let Some((range, total)) = value.split_once('/') else {
        return false;
    };
    let Some((start, end)) = range.split_once('-') else {
        return false;
    };
    start == "0"
        && end.parse::<u64>().ok() == expected_bytes.checked_sub(1)
        && total.parse::<u64>().ok() == Some(logical_size)
}

fn hub_token() -> Option<String> {
    std::env::var("HF_TOKEN")
        .ok()
        .filter(|token| !token.trim().is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use icn_contracts::{
        CapabilityEvidence, EffectiveTemplateInputs, HardwareAssessment, LocalDeclaration,
        ReasoningCapability, ReasoningControlDomain, ReasoningDelimiters, ReasoningVisibility,
        TemplateAssessment, TemplateAssessor, TemplateCapabilities,
    };

    struct TestTemplateAssessor;

    impl TemplateAssessor for TestTemplateAssessor {
        fn cache_identity(&self) -> &str {
            "preview-template-assessor:test"
        }

        fn assess(&self, _inputs: &EffectiveTemplateInputs) -> Result<TemplateAssessment, String> {
            Ok(TemplateAssessment {
                capabilities: TemplateCapabilities {
                    string_content: true,
                    typed_content: false,
                    tools: false,
                    tool_calls: false,
                    parallel_tool_calls: false,
                    system_role: true,
                    preserve_reasoning: false,
                    object_arguments: false,
                    enable_thinking: false,
                },
                reasoning: ReasoningCapability::Supported {
                    control: ReasoningControlDomain::Effort {
                        levels: vec!["none".to_owned()],
                        default: Some("none".to_owned()),
                    },
                    visibility: ReasoningVisibility::Hidden,
                    delimiters: ReasoningDelimiters::Unavailable,
                    evidence: CapabilityEvidence::BoundedTemplateProbe {
                        fingerprint: "preview-test-template".to_owned(),
                    },
                },
                fingerprint: "preview-test-template".to_owned(),
            })
        }
    }

    struct CountingProfileAssessor(AtomicUsize);

    impl HardwareProvider for CountingProfileAssessor {
        fn snapshot(
            &self,
        ) -> futures_util::future::BoxFuture<
            '_,
            Result<icn_contracts::HardwareSnapshot, InventoryError>,
        > {
            Box::pin(async {
                Ok(icn_contracts::HardwareSnapshot {
                    captured_at: 1,
                    platform: "test".to_owned(),
                    architecture: "test".to_owned(),
                    cpu_model: None,
                    logical_cores: 1,
                    native_build: "native".to_owned(),
                    enabled_backends: vec!["cpu".to_owned()],
                    assessment_policy: "test-policy".to_owned(),
                    capacity_policy: "stable".to_owned(),
                    topology_fingerprint: "topology".to_owned(),
                    memory_domains: Vec::new(),
                })
            })
        }
    }

    impl ModelHardwareAssessor for CountingProfileAssessor {
        fn policy_identity(&self) -> &str {
            "test-policy"
        }

        fn cache_key(
            &self,
            profile: Option<&icn_contracts::ModelPreviewProfile>,
            _snapshot: &icn_contracts::HardwareSnapshot,
        ) -> Result<String, InventoryError> {
            Ok(format!(
                "profile:{}",
                profile.map_or(0, |profile| profile.context_length)
            ))
        }

        fn assess_profile(
            &self,
            _model: ResolvedModel,
            profile: Option<icn_contracts::ModelPreviewProfile>,
        ) -> futures_util::future::BoxFuture<'_, Result<HardwareAssessment, InventoryError>>
        {
            Box::pin(async move {
                self.0.fetch_add(1, Ordering::SeqCst);
                tokio::time::sleep(std::time::Duration::from_millis(25)).await;
                Ok(HardwareAssessment::Fits {
                    profile: icn_contracts::HardwareProfile {
                        context_length: profile.map_or(0, |profile| profile.context_length),
                        acceleration: "cpu".to_owned(),
                        device: "system".to_owned(),
                    },
                    memory: icn_contracts::HardwareMemory {
                        required_bytes: 1,
                        available_bytes: 2,
                        headroom_bytes: 1,
                    },
                    recommendation: icn_contracts::HardwareRecommendation::Recommended,
                })
            })
        }
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

    #[test]
    fn preview_profiles_are_bounded_and_uniquely_correlated() {
        let source = ModelPreviewSource {
            repository: "owner/repository".to_owned(),
            revision: "a".repeat(40),
            primary_gguf: PathBuf::from("model.gguf"),
            additional_components: Vec::new(),
        };
        let profile = icn_contracts::ModelPreviewProfile {
            id: "profile".to_owned(),
            policy: "policy".to_owned(),
            context_length: 4096,
            parallel_sequences: 1,
        };
        assert!(
            validate_preview_request(&ModelPreviewRequest {
                source: source.clone(),
                profiles: vec![profile.clone()],
            })
            .is_ok()
        );
        assert!(
            validate_preview_request(&ModelPreviewRequest {
                source: source.clone(),
                profiles: vec![profile.clone(), profile.clone()],
            })
            .is_err()
        );
        assert!(
            validate_preview_request(&ModelPreviewRequest {
                source,
                profiles: vec![icn_contracts::ModelPreviewProfile {
                    context_length: MAX_PREVIEW_CONTEXT_TOKENS + 1,
                    ..profile
                }],
            })
            .is_err()
        );
    }

    #[tokio::test]
    async fn fresh_preview_survives_an_unusable_persistent_cache() {
        let temporary = tempfile::tempdir().unwrap();
        let store = temporary.path().join("model-store");
        let mut config = crate::inventory::InventoryConfig::with_root(store.clone()).unwrap();
        config.hf_cache_dirs.clear();
        let manager =
            ModelManager::open_with_template_assessor(config, Some(Arc::new(TestTemplateAssessor)))
                .await
                .unwrap();
        let cache_root = store.join("cache");
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
            icn_contracts::InventoryProperties::Inspected { .. }
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
                icn_contracts::ModelPreviewComponentSource {
                    path: PathBuf::from("projector.gguf"),
                    role: ComponentRole::Projector,
                },
                icn_contracts::ModelPreviewComponentSource {
                    path: PathBuf::from("draft.gguf"),
                    role: ComponentRole::Draft,
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
            additional_components: vec![icn_contracts::ModelPreviewComponentSource {
                path: PathBuf::from("model.gguf"),
                role: ComponentRole::Mtp,
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
        assert!(valid_content_range("bytes 0-1023/4096", 1024, 4096));
        assert!(!valid_content_range("bytes 1-1024/4096", 1024, 4096));
        assert!(!valid_content_range("bytes 0-1024/4096", 1024, 4096));
        assert!(!valid_content_range("bytes 0-1023/8192", 1024, 4096));
    }

    #[tokio::test]
    async fn preview_assessments_cache_only_terminal_results_by_complete_evidence() {
        let temporary = tempfile::tempdir().unwrap();
        let mut config =
            crate::inventory::InventoryConfig::with_root(temporary.path().join("model-store"))
                .unwrap();
        config.hf_cache_dirs.clear();
        let manager = ModelManager::open(config).await.unwrap();
        let assessment = HardwareAssessment::Fits {
            profile: icn_contracts::HardwareProfile {
                context_length: 4096,
                acceleration: "cpu".to_owned(),
                device: "system".to_owned(),
            },
            memory: icn_contracts::HardwareMemory {
                required_bytes: 10,
                available_bytes: 20,
                headroom_bytes: 10,
            },
            recommendation: icn_contracts::HardwareRecommendation::Recommended,
        };
        let content_id = icn_contracts::ContentId("artifact".to_owned());
        manager
            .cache
            .write_hardware_assessment(&content_id, "profile:hardware", &assessment);
        assert_eq!(
            manager
                .cache
                .read_hardware_assessment(&content_id, "profile:hardware"),
            Some(assessment)
        );
        assert!(
            manager
                .cache
                .read_hardware_assessment(&content_id, "other-profile:hardware")
                .is_none()
        );

        manager.cache.write_hardware_assessment(
            &content_id,
            "operational-failure",
            &HardwareAssessment::NotAssessed {
                reason: "temporary".to_owned(),
            },
        );
        assert!(
            manager
                .cache
                .read_hardware_assessment(&content_id, "operational-failure")
                .is_none()
        );
    }

    #[tokio::test]
    async fn concurrent_identical_previews_coalesce_native_assessment() {
        let temporary = tempfile::tempdir().unwrap();
        let mut config =
            crate::inventory::InventoryConfig::with_root(temporary.path().join("model-store"))
                .unwrap();
        config.hf_cache_dirs.clear();
        let manager = Arc::new(
            ModelManager::open_with_template_assessor(config, Some(Arc::new(TestTemplateAssessor)))
                .await
                .unwrap(),
        );
        let source = ModelPreviewSource {
            repository: "owner/repository".to_owned(),
            revision: "a".repeat(40),
            primary_gguf: PathBuf::from("model.gguf"),
            additional_components: Vec::new(),
        };
        let mut header = Vec::new();
        header.extend_from_slice(b"GGUF");
        header.extend_from_slice(&3_u32.to_le_bytes());
        header.extend_from_slice(&0_u64.to_le_bytes());
        header.extend_from_slice(&0_u64.to_le_bytes());
        header.resize(32, 0);
        let header_digest = format!("{:x}", Sha256::digest(&header));
        manager
            .cache
            .write_blob(ModelBlobKind::GgufHeader, &header_digest, &header);
        let component = ModelComponent {
            path: source.primary_gguf.clone(),
            role: ComponentRole::Weights,
            size_bytes: header.len() as u64,
            content: ContentIdentity::Sha256 {
                value: format!("{:x}", Sha256::digest(&header)),
            },
            shard_index: None,
            relationship: None,
        };
        let source_evidence = serde_json::to_string(&source).unwrap();
        manager.cache.write_index(
            ModelIndexKind::Artifact,
            &source_evidence,
            &CachedArtifact {
                repository: source.repository.clone(),
                commit: source.revision.clone(),
                primary_gguf: source.primary_gguf.clone(),
                components: vec![CachedComponent {
                    component,
                    header_digest,
                    acquired_header: None,
                }],
            },
        );

        let assessor = Arc::new(CountingProfileAssessor(AtomicUsize::new(0)));
        let service = ModelPreviewService::new(manager, assessor.clone());
        let request = ModelPreviewRequest {
            source,
            profiles: vec![icn_contracts::ModelPreviewProfile {
                id: "profile".to_owned(),
                policy: "test-policy".to_owned(),
                context_length: 4096,
                parallel_sequences: 1,
            }],
        };
        let (first, second) = tokio::join!(
            service.preview(request.clone()),
            service.preview(request.clone())
        );
        assert_eq!(first.unwrap(), second.unwrap());
        assert_eq!(assessor.0.load(Ordering::SeqCst), 1);
        service.preview(request).await.unwrap();
        assert_eq!(assessor.0.load(Ordering::SeqCst), 1);
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
        let mut config =
            crate::inventory::InventoryConfig::with_root(temporary.path().join("model-store"))
                .unwrap();
        config.hf_cache_dirs.clear();
        let manager =
            ModelManager::open_with_template_assessor(config, Some(Arc::new(TestTemplateAssessor)))
                .await
                .unwrap();

        for (index, fixture) in fixtures.into_iter().enumerate() {
            let inspection = crate::gguf::inspect(&fixture).unwrap();
            let mut header = vec![0_u8; usize::try_from(inspection.header_bytes).unwrap()];
            std::fs::File::open(&fixture)
                .unwrap()
                .read_exact(&mut header)
                .unwrap();
            let header_digest = format!("{:x}", Sha256::digest(&header));
            manager
                .cache
                .write_blob(ModelBlobKind::GgufHeader, &header_digest, &header);
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
                ModelId(format!("mdl_{index:064x}")),
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
                manager.template_assessor.as_deref(),
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
}
