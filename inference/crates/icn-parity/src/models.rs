use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component as PathComponent, Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, bail};
use futures_util::StreamExt;
use reqwest::Url;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;

use crate::digest::{sha256_file, valid_sha256};
use crate::model::valid_primitive;

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelRegistry {
    pub schema_version: u32,
    pub artifact_root_env: String,
    pub models: Vec<ModelRecord>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelRecord {
    pub id: String,
    pub status: String,
    pub kind: String,
    pub display_name: String,
    pub valid_for: Vec<String>,
    pub source: ModelSource,
    pub access: ModelAccess,
    pub license: ModelLicense,
    pub attributes: ModelAttributes,
    pub files: Vec<ModelFile>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSource {
    pub provider: String,
    pub repository: String,
    pub revision: String,
    pub integrity: String,
    #[serde(default)]
    pub note: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelAccess {
    pub public: bool,
    pub gated: bool,
    pub authentication_required: bool,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelLicense {
    pub identifier: String,
    pub redistribution: String,
    #[serde(default)]
    pub note: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelAttributes {
    pub architecture_tags: Vec<String>,
    pub tokenizer_family: String,
    pub template_family: String,
    pub quantization: String,
    pub parameter_scale: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ModelFile {
    pub role: String,
    pub path: PathBuf,
    pub bytes: u64,
    #[serde(default)]
    pub max_bytes: Option<u64>,
    pub sha256: String,
    pub url: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelFileStatus {
    pub model_id: String,
    pub path: PathBuf,
    pub state: ModelFileState,
    pub expected_bytes: u64,
    pub actual_bytes: Option<u64>,
    pub expected_sha256: String,
    pub actual_sha256: Option<String>,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum ModelFileState {
    Verified,
    Missing,
    SizeMismatch,
    DigestMismatch,
}

impl ModelRegistry {
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read model registry {}", path.display()))?;
        let registry: Self = toml::from_str(&text)
            .with_context(|| format!("invalid model registry {}", path.display()))?;
        registry.validate()?;
        Ok(registry)
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        if self.schema_version != 1 {
            bail!(
                "unsupported model registry schema_version {}",
                self.schema_version
            );
        }
        if self.artifact_root_env.trim().is_empty() {
            bail!("model registry artifact_root_env must be non-empty");
        }
        if self.models.is_empty() {
            bail!("model registry must contain at least one accepted model");
        }
        let mut ids = BTreeSet::new();
        for model in &self.models {
            if !ids.insert(&model.id) {
                bail!("duplicate model registry id {}", model.id);
            }
            if model.files.is_empty() {
                bail!("model {} has no files", model.id);
            }
            if model.status != "accepted" {
                bail!("model {} is not accepted", model.id);
            }
            let known_primitives = model.valid_for.iter().collect::<BTreeSet<_>>();
            if known_primitives.len() != model.valid_for.len()
                || model.valid_for.iter().any(|value| !valid_primitive(value))
            {
                bail!(
                    "model {} has invalid or duplicate valid_for entries",
                    model.id
                );
            }
            if model.source.revision.trim().is_empty()
                || model.source.repository.trim().is_empty()
                || model.source.integrity != "digest-pinned"
            {
                bail!(
                    "model {} source must be digest-pinned and identify a repository/revision",
                    model.id
                );
            }
            let mut roles = BTreeSet::new();
            let mut paths = BTreeSet::new();
            for file in &model.files {
                validate_relative_artifact_path(&file.path)
                    .with_context(|| format!("model {} file path", model.id))?;
                if file.role.trim().is_empty()
                    || !roles.insert(&file.role)
                    || !paths.insert(&file.path)
                    || file.bytes == 0
                    || file.max_bytes.is_some_and(|maximum| maximum < file.bytes)
                    || !valid_sha256(&file.sha256)
                    || validate_https_url(&file.url).is_err()
                {
                    bail!(
                        "model {} contains an invalid file record for {}",
                        model.id,
                        file.path.display()
                    );
                }
            }
        }
        Ok(())
    }

    pub fn by_id(&self, id: &str) -> anyhow::Result<&ModelRecord> {
        self.models
            .iter()
            .find(|model| model.id == id)
            .with_context(|| format!("unknown model registry id {id}"))
    }

    pub fn artifact_root(&self, parity_root: &Path, override_root: Option<&Path>) -> PathBuf {
        override_root
            .map(Path::to_path_buf)
            .or_else(|| std::env::var_os(&self.artifact_root_env).map(PathBuf::from))
            .unwrap_or_else(|| {
                parity_root
                    .parent()
                    .unwrap_or(parity_root)
                    .join("target/parity-models")
            })
    }

    pub fn model_paths(
        &self,
        id: &str,
        artifact_root: &Path,
    ) -> anyhow::Result<BTreeMap<String, PathBuf>> {
        let model = self.by_id(id)?;
        model
            .files
            .iter()
            .map(|file| {
                Ok((
                    file.role.clone(),
                    safe_artifact_path(artifact_root, &file.path)?,
                ))
            })
            .collect()
    }

    pub async fn verify(
        &self,
        ids: &[String],
        artifact_root: &Path,
    ) -> anyhow::Result<Vec<ModelFileStatus>> {
        let artifact_root = if artifact_root.exists() {
            std::fs::canonicalize(artifact_root).with_context(|| {
                format!(
                    "failed to canonicalize artifact root {}",
                    artifact_root.display()
                )
            })?
        } else {
            artifact_root.to_path_buf()
        };
        let selected = self.select(ids)?;
        let mut statuses = Vec::new();
        for model in selected {
            for file in &model.files {
                let path = safe_artifact_path(&artifact_root, &file.path)?;
                if !path.is_file() {
                    statuses.push(ModelFileStatus {
                        model_id: model.id.clone(),
                        path,
                        state: ModelFileState::Missing,
                        expected_bytes: file.bytes,
                        actual_bytes: None,
                        expected_sha256: file.sha256.clone(),
                        actual_sha256: None,
                    });
                    continue;
                }
                let (digest, bytes) = sha256_file(&path).await?;
                statuses.push(ModelFileStatus {
                    model_id: model.id.clone(),
                    path,
                    state: if bytes != file.bytes {
                        ModelFileState::SizeMismatch
                    } else if digest != file.sha256 {
                        ModelFileState::DigestMismatch
                    } else {
                        ModelFileState::Verified
                    },
                    expected_bytes: file.bytes,
                    actual_bytes: Some(bytes),
                    expected_sha256: file.sha256.clone(),
                    actual_sha256: Some(digest),
                });
            }
        }
        Ok(statuses)
    }

    pub async fn fetch(
        &self,
        ids: &[String],
        artifact_root: &Path,
    ) -> anyhow::Result<Vec<ModelFileStatus>> {
        if ids.is_empty() {
            bail!("model fetch requires at least one explicit model id");
        }
        tokio::fs::create_dir_all(artifact_root)
            .await
            .with_context(|| {
                format!("failed to create artifact root {}", artifact_root.display())
            })?;
        let artifact_root = std::fs::canonicalize(artifact_root).with_context(|| {
            format!(
                "failed to canonicalize artifact root {}",
                artifact_root.display()
            )
        })?;
        let selected = self.select(ids)?;
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(20))
            .timeout(Duration::from_secs(30 * 60))
            .https_only(true)
            .redirect(reqwest::redirect::Policy::custom(|attempt| {
                if attempt.url().scheme() != "https" {
                    attempt.error("model fetch redirect attempted to leave HTTPS")
                } else if attempt.previous().len() >= 5 {
                    attempt.error("too many model fetch redirects")
                } else {
                    attempt.follow()
                }
            }))
            .build()
            .context("failed to construct model fetch client")?;
        for model in selected {
            if model.access.authentication_required || model.access.gated || !model.access.public {
                bail!(
                    "model {} is not eligible for unattended public fetch",
                    model.id
                );
            }
            for file in &model.files {
                let destination = ensure_safe_artifact_parent(&artifact_root, &file.path)?;
                if let Ok(metadata) = std::fs::symlink_metadata(&destination) {
                    if !metadata.is_file() {
                        bail!(
                            "refusing to replace non-file model artifact {}",
                            destination.display()
                        );
                    }
                    let (digest, bytes) = sha256_file(&destination).await?;
                    if bytes == file.bytes && digest == file.sha256 {
                        continue;
                    }
                    bail!(
                        "refusing to replace unverified existing artifact {}; remove or relocate it explicitly",
                        destination.display()
                    );
                }
                let parent = destination.parent().context("model file has no parent")?;
                let temporary = tempfile::Builder::new()
                    .prefix(".icn-parity-download-")
                    .tempfile_in(parent)
                    .with_context(|| {
                        format!("failed to create temporary file in {}", parent.display())
                    })?;
                let request_url = validate_https_url(&file.url)?;
                let response = client
                    .get(request_url)
                    .send()
                    .await
                    .with_context(|| format!("failed to fetch model artifact from {}", file.url))?
                    .error_for_status()
                    .with_context(|| format!("model artifact request failed for {}", file.url))?;
                if response.url().scheme() != "https" {
                    bail!("model artifact response left HTTPS for {}", file.url);
                }
                let maximum = file.max_bytes.unwrap_or(file.bytes);
                if response
                    .content_length()
                    .is_some_and(|length| length > maximum)
                {
                    bail!("model response exceeds declared maximum of {maximum} bytes");
                }
                // Retain NamedTempFile's original handle and clone that handle
                // for async writes. Never close and reopen its pathname.
                let std_file = temporary
                    .as_file()
                    .try_clone()
                    .context("failed to clone temporary model file handle")?;
                let mut output = tokio::fs::File::from_std(std_file);
                let mut received = 0_u64;
                let mut digest = Sha256::new();
                let mut stream = response.bytes_stream();
                while let Some(chunk) = stream.next().await {
                    let chunk = chunk.context("failed while reading model response")?;
                    received = received.saturating_add(chunk.len() as u64);
                    if received > maximum {
                        bail!("model response exceeded declared maximum of {maximum} bytes");
                    }
                    output
                        .write_all(&chunk)
                        .await
                        .context("failed to write model artifact")?;
                    digest.update(&chunk);
                }
                output
                    .flush()
                    .await
                    .context("failed to flush model artifact")?;
                output
                    .sync_all()
                    .await
                    .context("failed to sync model artifact")?;
                drop(output);
                if received != file.bytes {
                    bail!(
                        "downloaded {} bytes for {}, expected {}",
                        received,
                        model.id,
                        file.bytes
                    );
                }
                let digest = format!("{:x}", digest.finalize());
                if digest != file.sha256 {
                    bail!(
                        "downloaded artifact for {} failed digest verification",
                        model.id
                    );
                }
                match temporary.persist_noclobber(&destination) {
                    Ok(file) => file.sync_all().with_context(|| {
                        format!("failed to sync model artifact {}", destination.display())
                    })?,
                    Err(error) if error.error.kind() == std::io::ErrorKind::AlreadyExists => {
                        drop(error.file);
                        let safe_destination = safe_artifact_path(&artifact_root, &file.path)?;
                        let (winner_digest, winner_bytes) = sha256_file(&safe_destination)
                            .await
                            .context("failed to verify concurrent model fetch winner")?;
                        if winner_bytes != file.bytes || winner_digest != file.sha256 {
                            bail!(
                                "concurrent model fetch produced an unverified artifact at {}",
                                safe_destination.display()
                            );
                        }
                    }
                    Err(error) => {
                        return Err(error.error).with_context(|| {
                            format!(
                                "failed to atomically place model artifact at {}",
                                destination.display()
                            )
                        });
                    }
                }
                sync_parent_directory(parent)?;
            }
        }
        self.verify(ids, &artifact_root).await
    }

    fn select(&self, ids: &[String]) -> anyhow::Result<Vec<&ModelRecord>> {
        if ids.is_empty() {
            return Ok(self.models.iter().collect());
        }
        let mut seen = BTreeSet::new();
        ids.iter()
            .map(|id| {
                if !seen.insert(id) {
                    bail!("duplicate selected model id {id}");
                }
                self.by_id(id)
            })
            .collect()
    }
}

fn validate_https_url(value: &str) -> anyhow::Result<Url> {
    let url = Url::parse(value).with_context(|| format!("invalid model artifact URL {value:?}"))?;
    if url.scheme() != "https"
        || !url.has_host()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.fragment().is_some()
    {
        bail!("model artifact URL must be credential-free HTTPS without a fragment");
    }
    Ok(url)
}

fn ensure_safe_artifact_parent(root: &Path, relative: &Path) -> anyhow::Result<PathBuf> {
    validate_relative_artifact_path(relative)?;
    let parent = relative.parent().unwrap_or_else(|| Path::new(""));
    let mut current = root.to_path_buf();
    for component in parent.components() {
        let PathComponent::Normal(component) = component else {
            bail!("artifact parent path contains an invalid component");
        };
        current.push(component);
        match std::fs::symlink_metadata(&current) {
            Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => {}
            Ok(_) => bail!(
                "artifact parent is not a real directory: {}",
                current.display()
            ),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                match std::fs::create_dir(&current) {
                    Ok(()) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                        let metadata = std::fs::symlink_metadata(&current)?;
                        if !metadata.is_dir() || metadata.file_type().is_symlink() {
                            bail!(
                                "artifact parent became unsafe during creation: {}",
                                current.display()
                            );
                        }
                    }
                    Err(error) => {
                        return Err(error).with_context(|| {
                            format!("failed to create artifact parent {}", current.display())
                        });
                    }
                }
            }
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("failed to inspect artifact parent {}", current.display())
                });
            }
        }
    }
    let canonical_parent = std::fs::canonicalize(&current)
        .with_context(|| format!("failed to resolve artifact parent {}", current.display()))?;
    if !canonical_parent.starts_with(root) {
        bail!("artifact parent escapes model root: {}", current.display());
    }
    safe_artifact_path(root, relative)
}

#[cfg(unix)]
fn sync_parent_directory(parent: &Path) -> anyhow::Result<()> {
    std::fs::File::open(parent)
        .and_then(|directory| directory.sync_all())
        .with_context(|| {
            format!(
                "failed to sync model artifact directory {}",
                parent.display()
            )
        })
}

#[cfg(not(unix))]
fn sync_parent_directory(_parent: &Path) -> anyhow::Result<()> {
    Ok(())
}

fn safe_artifact_path(root: &Path, relative: &Path) -> anyhow::Result<PathBuf> {
    validate_relative_artifact_path(relative)?;
    let mut current = root.to_path_buf();
    for component in relative.components() {
        let PathComponent::Normal(component) = component else {
            bail!("artifact path contains an invalid component");
        };
        current.push(component);
        if let Ok(metadata) = std::fs::symlink_metadata(&current)
            && metadata.file_type().is_symlink()
        {
            bail!("artifact path traverses symlink {}", current.display());
        }
    }
    Ok(current)
}

fn validate_relative_artifact_path(path: &Path) -> anyhow::Result<()> {
    if path.as_os_str().is_empty()
        || path.is_absolute()
        || path
            .components()
            .any(|component| !matches!(component, PathComponent::Normal(_)))
    {
        bail!("artifact path must be relative and cannot contain parent traversal");
    }
    Ok(())
}

pub fn all_verified(statuses: &[ModelFileStatus]) -> bool {
    !statuses.is_empty()
        && statuses
            .iter()
            .all(|status| status.state == ModelFileState::Verified)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_traversal() {
        assert!(validate_relative_artifact_path(Path::new("../model.gguf")).is_err());
        assert!(validate_relative_artifact_path(Path::new("")).is_err());
        assert!(validate_relative_artifact_path(Path::new("./model.gguf")).is_err());
        assert!(validate_relative_artifact_path(Path::new("models/model.gguf")).is_ok());
    }

    #[test]
    fn validates_https_artifact_urls_structurally() {
        assert!(validate_https_url("https://example.com/model.gguf").is_ok());
        assert!(validate_https_url("http://example.com/model.gguf").is_err());
        assert!(validate_https_url("https://user@example.com/model.gguf").is_err());
        assert!(validate_https_url("https://example.com/model.gguf#fragment").is_err());
        assert!(validate_https_url("https-not-really://example.com").is_err());
    }

    #[test]
    fn empty_status_collection_is_not_verified() {
        assert!(!all_verified(&[]));
    }

    #[tokio::test]
    async fn fetch_requires_an_explicit_selection_before_network_access() {
        let registry = ModelRegistry {
            schema_version: 1,
            artifact_root_env: "TEST_MODELS".to_owned(),
            models: Vec::new(),
        };
        let root = tempfile::tempdir().unwrap();
        assert!(registry.fetch(&[], root.path()).await.is_err());
    }

    #[cfg(unix)]
    #[test]
    fn rejects_artifact_paths_that_traverse_symlinks() {
        use std::os::unix::fs::symlink;
        let root = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        symlink(outside.path(), root.path().join("escape")).unwrap();
        assert!(safe_artifact_path(root.path(), Path::new("escape/model.gguf")).is_err());
    }
}
