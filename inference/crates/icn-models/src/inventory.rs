use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use fs2::FileExt;
use hf_hub::HFClient;
use icn_contracts::{
    CapabilitySupport, ComponentRole, ContentIdentity, HardwareAssessment, Integrity,
    InventoryError, InventoryModel, InventoryProperties, LocalDeclaration, ModelComponent, ModelId,
    ModelLocation, ModelOperation, ModelSource, ModelStatus, ReasoningCapability, TemplateAssessor,
};
use sha2::{Digest, Sha256};

use crate::download::blob_key;
use crate::gguf;
use crate::identity::{content_id, fingerprint, model_id};
use crate::manifest::{MANIFEST_VERSION, ManagedManifest, OperationManifest};

const MAX_SCAN_ENTRIES: usize = 100_000;
const MAX_SCAN_DEPTH: usize = 8;

#[derive(Debug, Clone)]
pub struct InventoryConfig {
    pub root: PathBuf,
    /// Optional v1 TypeScript-owned Hugging Face store imported once at startup.
    pub legacy_store: Option<PathBuf>,
    pub hf_cache_dirs: Vec<PathBuf>,
    pub model_sources: Vec<PathBuf>,
    pub max_concurrent_downloads: usize,
    pub disk_reserve_bytes: u64,
}

impl InventoryConfig {
    pub fn default_root() -> Result<PathBuf, InventoryError> {
        let home = std::env::var_os("HOME").ok_or_else(|| {
            InventoryError::InvalidRequest(
                "cannot determine the user home directory for the model store".to_owned(),
            )
        })?;
        Ok(PathBuf::from(home).join(".magnitude/models"))
    }

    pub fn with_root(root: PathBuf) -> Result<Self, InventoryError> {
        if !root.is_absolute() {
            return Err(InventoryError::InvalidRequest(
                "model store root must be absolute".to_owned(),
            ));
        }
        let client =
            HFClient::new().map_err(|error| InventoryError::Upstream(error.to_string()))?;
        let external_hf_cache = std::env::var_os("HF_HUB_CACHE")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HF_HOME").map(|home| PathBuf::from(home).join("hub")))
            .unwrap_or_else(|| client.cache_dir().to_path_buf());
        Ok(Self {
            root,
            legacy_store: None,
            hf_cache_dirs: vec![external_hf_cache],
            model_sources: Vec::new(),
            max_concurrent_downloads: 2,
            disk_reserve_bytes: 2 * 1024 * 1024 * 1024,
        })
    }
}

pub struct ModelManager {
    pub(crate) config: InventoryConfig,
    pub(crate) client: HFClient,
    pub(crate) models: Arc<RwLock<BTreeMap<ModelId, InventoryModel>>>,
    pub(crate) operations:
        Arc<tokio::sync::Mutex<BTreeMap<String, Arc<crate::download::DownloadOperation>>>>,
    pub(crate) download_slots: Arc<tokio::sync::Semaphore>,
    pub(crate) template_assessor: Option<Arc<dyn TemplateAssessor>>,
}

impl Clone for ModelManager {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            client: self.client.clone(),
            models: Arc::clone(&self.models),
            operations: Arc::clone(&self.operations),
            download_slots: Arc::clone(&self.download_slots),
            template_assessor: self.template_assessor.clone(),
        }
    }
}

impl ModelManager {
    pub async fn open(config: InventoryConfig) -> Result<Self, InventoryError> {
        Self::open_with_template_assessor(config, None).await
    }

    pub async fn open_with_template_assessor(
        config: InventoryConfig,
        template_assessor: Option<Arc<dyn TemplateAssessor>>,
    ) -> Result<Self, InventoryError> {
        validate_config(&config)?;
        create_layout(&config.root).await?;
        if let Some(legacy_store) = config.legacy_store.clone() {
            let destination = config.root.clone();
            tokio::task::spawn_blocking(move || {
                crate::legacy::import_v1_store(&legacy_store, &destination)
            })
            .await
            .map_err(|error| InventoryError::Internal(error.to_string()))??;
        }
        let reconciliation_root = config.root.clone();
        tokio::task::spawn_blocking(move || {
            crate::service::reconcile_tombstones(&reconciliation_root)
        })
        .await
        .map_err(|error| InventoryError::Internal(error.to_string()))??;
        let client = HFClient::builder()
            .cache_dir(config.root.join("hub"))
            .build()
            .map_err(|error| InventoryError::Upstream(error.to_string()))?;
        let manager = Self {
            download_slots: Arc::new(tokio::sync::Semaphore::new(config.max_concurrent_downloads)),
            config,
            client,
            models: Arc::new(RwLock::new(BTreeMap::new())),
            operations: Arc::new(tokio::sync::Mutex::new(BTreeMap::new())),
            template_assessor,
        };
        manager.refresh().await?;
        Ok(manager)
    }

    pub async fn refresh(&self) -> Result<(), InventoryError> {
        let config = self.config.clone();
        let assessor = self.template_assessor.clone();
        let discovered = tokio::task::spawn_blocking(move || scan(&config, assessor.as_deref()))
            .await
            .map_err(|error| InventoryError::Internal(error.to_string()))??;
        *self
            .models
            .write()
            .map_err(|_| InventoryError::Internal("inventory lock poisoned".to_owned()))? =
            discovered;
        Ok(())
    }

    pub fn root(&self) -> &Path {
        &self.config.root
    }

    /// Ensure the model currently selected by the process has an inventory
    /// identity, even when its file is outside configured discovery roots.
    pub async fn register_active_model(
        &self,
        path: &Path,
        display_name: Option<&str>,
    ) -> Result<ModelId, InventoryError> {
        let canonical = path.canonicalize().map_err(io_error)?;
        if !canonical.is_file() {
            return Err(InventoryError::InvalidRequest(format!(
                "active model is not a regular file: {}",
                path.display()
            )));
        }
        let existing = self
            .models
            .read()
            .map_err(|_| InventoryError::Internal("inventory lock poisoned".to_owned()))?
            .values()
            .find(|model| model_primary_path(&self.config.root, model).as_ref() == Some(&canonical))
            .map(|model| model.id.clone());
        if let Some(id) = existing {
            return Ok(id);
        }

        let metadata = canonical.metadata().map_err(io_error)?;
        let component = ModelComponent {
            path: canonical.file_name().map(PathBuf::from).ok_or_else(|| {
                InventoryError::InvalidRequest("active model has no filename".to_owned())
            })?,
            role: ComponentRole::Weights,
            size_bytes: metadata.len(),
            content: ContentIdentity::FileIdentity {
                value: file_identity(&canonical, &metadata),
            },
            shard_index: None,
            relationship: None,
        };
        let content = content_id(std::slice::from_ref(&component));
        let id = model_id("active-file", &canonical, &content);
        let timestamp = now();
        let mut model = build_model(
            id.clone(),
            content,
            timestamp,
            timestamp,
            ModelSource::Local {
                declared_by: LocalDeclaration::ActiveProcess,
            },
            ModelLocation::File {
                path: canonical.clone(),
                component,
                integrity: Integrity::Unverified {
                    reason: "active_process".to_owned(),
                },
            },
            &canonical,
            false,
            self.template_assessor.as_deref(),
        );
        if let Some(name) = display_name {
            model.name = name.to_owned();
        }
        self.models
            .write()
            .map_err(|_| InventoryError::Internal("inventory lock poisoned".to_owned()))?
            .insert(id.clone(), model);
        Ok(id)
    }

    pub fn update_hardware(
        &self,
        id: &ModelId,
        assessment: HardwareAssessment,
    ) -> Result<(), InventoryError> {
        let mut models = self
            .models
            .write()
            .map_err(|_| InventoryError::Internal("inventory lock poisoned".to_owned()))?;
        let model = models
            .get_mut(id)
            .ok_or_else(|| InventoryError::NotFound(id.0.clone()))?;
        model.hardware = assessment;
        model.updated_at = now();
        Ok(())
    }
}

fn model_primary_path(root: &Path, model: &InventoryModel) -> Option<PathBuf> {
    let component = model.location.components().iter().find(|component| {
        matches!(
            component.role,
            ComponentRole::Weights | ComponentRole::Shard
        )
    })?;
    let path = match (&model.location, &model.source) {
        (
            ModelLocation::MagnitudeCache { .. },
            ModelSource::HuggingFace {
                repository, commit, ..
            },
        ) => root
            .join("hub")
            .join(hf_repo_dir(repository))
            .join("snapshots")
            .join(commit)
            .join(&component.path),
        (ModelLocation::HuggingFaceCache { cache_root, .. }, _) => cache_root.join(&component.path),
        (ModelLocation::Directory { root, .. }, _) => root.join(&component.path),
        (ModelLocation::File { path, .. }, _) => path.clone(),
        _ => return None,
    };
    path.canonicalize().ok()
}

fn validate_config(config: &InventoryConfig) -> Result<(), InventoryError> {
    if !config.root.is_absolute() {
        return Err(InventoryError::InvalidRequest(
            "model store root must be absolute".to_owned(),
        ));
    }
    if config.max_concurrent_downloads == 0 {
        return Err(InventoryError::InvalidRequest(
            "max_concurrent_downloads must be positive".to_owned(),
        ));
    }
    for root in config
        .hf_cache_dirs
        .iter()
        .chain(config.model_sources.iter())
        .chain(config.legacy_store.iter())
    {
        if !root.is_absolute() {
            return Err(InventoryError::InvalidRequest(format!(
                "configured model source must be absolute: {}",
                root.display()
            )));
        }
    }
    Ok(())
}

async fn create_layout(root: &Path) -> Result<(), InventoryError> {
    for relative in [
        "hub",
        "installations",
        "operations",
        "locks",
        "trash",
        "quarantine",
    ] {
        let path = root.join(relative);
        tokio::fs::create_dir_all(&path).await.map_err(io_error)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            tokio::fs::set_permissions(&path, fs::Permissions::from_mode(0o700))
                .await
                .map_err(io_error)?;
        }
    }
    let schema = root.join("schema-version");
    if !schema.exists() {
        tokio::fs::write(&schema, format!("{MANIFEST_VERSION}\n"))
            .await
            .map_err(io_error)?;
    }
    Ok(())
}

fn scan(
    config: &InventoryConfig,
    assessor: Option<&dyn TemplateAssessor>,
) -> Result<BTreeMap<ModelId, InventoryModel>, InventoryError> {
    let mut discovered = Vec::new();
    scan_managed(config, assessor, &mut discovered)?;

    let mut distinct_hf = BTreeSet::new();
    for cache in &config.hf_cache_dirs {
        let canonical = cache.canonicalize().unwrap_or_else(|_| cache.clone());
        if canonical != config.root.join("hub") && distinct_hf.insert(canonical.clone()) {
            scan_hf_cache(&canonical, assessor, &mut discovered)?;
        }
    }
    for source in &config.model_sources {
        let canonical = source.canonicalize().unwrap_or_else(|_| source.clone());
        let source_id = format!(
            "configured-{}",
            fingerprint(canonical.to_string_lossy().as_bytes())
        );
        scan_directory(source, &source_id, assessor, &mut discovered)?;
    }
    scan_interrupted(config, &mut discovered)?;

    // Earlier sources have higher precedence. A canonical model path is projected once.
    let mut seen_paths = BTreeSet::new();
    let mut models = BTreeMap::new();
    for (path, model) in discovered {
        let canonical = path.canonicalize().unwrap_or(path);
        if seen_paths.insert(canonical) {
            models.insert(model.id.clone(), model);
        }
    }
    apply_persisted_discovery_times(config, &mut models)?;
    Ok(models)
}

fn apply_persisted_discovery_times(
    config: &InventoryConfig,
    models: &mut BTreeMap<ModelId, InventoryModel>,
) -> Result<(), InventoryError> {
    let lock_path = config.root.join("locks/discovery.lock");
    let lock = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(lock_path)
        .map_err(io_error)?;
    FileExt::lock_exclusive(&lock).map_err(io_error)?;
    let path = config.root.join("discovery.json");
    let mut times = fs::read(&path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<BTreeMap<String, u64>>(&bytes).ok())
        .unwrap_or_default();
    let mut changed = false;
    let discovered_at = now();
    for model in models.values_mut() {
        if matches!(model.location, ModelLocation::MagnitudeCache { .. }) {
            continue;
        }
        let created = *times.entry(model.id.0.clone()).or_insert_with(|| {
            changed = true;
            discovered_at
        });
        model.created = created;
        model.updated_at = created;
        if let ModelStatus::Available { ready_at } = &mut model.status {
            *ready_at = created;
        }
    }
    if changed {
        let bytes = serde_json::to_vec_pretty(&times)
            .map_err(|error| InventoryError::Internal(error.to_string()))?;
        let temporary = path.with_extension(format!("tmp-{}", std::process::id()));
        fs::write(&temporary, bytes).map_err(io_error)?;
        fs::rename(temporary, path).map_err(io_error)?;
    }
    FileExt::unlock(&lock).map_err(io_error)?;
    Ok(())
}

fn scan_managed(
    config: &InventoryConfig,
    assessor: Option<&dyn TemplateAssessor>,
    output: &mut Vec<(PathBuf, InventoryModel)>,
) -> Result<(), InventoryError> {
    let manifests = config.root.join("installations");
    for entry in read_dir_sorted(&manifests)? {
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let bytes = match fs::read(&path) {
            Ok(bytes) => bytes,
            Err(_) => continue,
        };
        let manifest: ManagedManifest = match serde_json::from_slice::<ManagedManifest>(&bytes) {
            Ok(manifest) if manifest.validate().is_ok() => manifest,
            _ => continue,
        };
        let snapshot = config
            .root
            .join("hub")
            .join(hf_repo_dir(&manifest.repository))
            .join("snapshots")
            .join(&manifest.commit);
        let repository_root = config
            .root
            .join("hub")
            .join(hf_repo_dir(&manifest.repository));
        if !components_exist_contained(&snapshot, &repository_root, &manifest.components) {
            continue;
        }
        let primary = match primary_path(&snapshot, &manifest.components) {
            Some(path) => path,
            None => continue,
        };
        let model = build_model(
            manifest.model_id,
            manifest.content_id,
            manifest.created_at,
            manifest.ready_at,
            ModelSource::HuggingFace {
                repository: manifest.repository,
                requested_revision: manifest.requested_revision,
                commit: manifest.commit,
                metadata: None,
            },
            ModelLocation::MagnitudeCache {
                total_bytes: manifest.components.iter().map(|item| item.size_bytes).sum(),
                components: manifest.components,
                integrity: Integrity::Verified {
                    method: "manifest".to_owned(),
                },
            },
            &primary,
            true,
            assessor,
        );
        output.push((primary, model));
    }
    Ok(())
}

fn scan_hf_cache(
    cache: &Path,
    assessor: Option<&dyn TemplateAssessor>,
    output: &mut Vec<(PathBuf, InventoryModel)>,
) -> Result<(), InventoryError> {
    if !cache.is_dir() {
        return Ok(());
    }
    let mut count = 0;
    for repo_entry in read_dir_sorted(cache)? {
        let repo_name = repo_entry.file_name().to_string_lossy().into_owned();
        let Some(repository) = parse_hf_repo_dir(&repo_name) else {
            continue;
        };
        let repo_root = repo_entry.path();
        let snapshots = repo_root.join("snapshots");
        for snapshot_entry in read_dir_sorted(&snapshots)? {
            count += 1;
            if count > MAX_SCAN_ENTRIES {
                return Err(InventoryError::Io(
                    "Hugging Face cache scan exceeded entry bound".to_owned(),
                ));
            }
            let commit = snapshot_entry.file_name().to_string_lossy().into_owned();
            let snapshot = snapshot_entry.path();
            let groups = discover_groups(&snapshot, &repo_root)?;
            for group in groups {
                let components = components_for_group(&snapshot, &group)?;
                let primary = match primary_path(&snapshot, &components) {
                    Some(path) => path,
                    None => continue,
                };
                let content = content_id(&components);
                let id = model_id("hugging-face-cache", &snapshot, &content);
                let created = modified_seconds(&snapshot).unwrap_or_else(now);
                let model = build_model(
                    id,
                    content,
                    created,
                    created,
                    ModelSource::HuggingFace {
                        repository: repository.clone(),
                        requested_revision: commit.clone(),
                        commit: commit.clone(),
                        metadata: None,
                    },
                    ModelLocation::HuggingFaceCache {
                        cache_root: snapshot.clone(),
                        repository: repository.clone(),
                        commit: commit.clone(),
                        total_bytes: components.iter().map(|item| item.size_bytes).sum(),
                        components,
                        integrity: Integrity::Unverified {
                            reason: "external_cache".to_owned(),
                        },
                    },
                    &primary,
                    false,
                    assessor,
                );
                output.push((primary, model));
            }
        }
    }
    Ok(())
}

fn scan_directory(
    root: &Path,
    source_id: &str,
    assessor: Option<&dyn TemplateAssessor>,
    output: &mut Vec<(PathBuf, InventoryModel)>,
) -> Result<(), InventoryError> {
    if !root.is_dir() {
        return Ok(());
    }
    let canonical_root = root.canonicalize().map_err(io_error)?;
    let groups = discover_groups(&canonical_root, &canonical_root)?;
    for group in groups {
        let components = components_for_group(&canonical_root, &group)?;
        let primary = match primary_path(&canonical_root, &components) {
            Some(path) => path,
            None => continue,
        };
        let content = content_id(&components);
        let id = model_id("directory", &canonical_root, &content);
        let created = modified_seconds(&primary).unwrap_or_else(now);
        let model = build_model(
            id,
            content,
            created,
            created,
            ModelSource::Local {
                declared_by: LocalDeclaration::Configuration,
            },
            ModelLocation::Directory {
                source_id: source_id.to_owned(),
                root: canonical_root.clone(),
                total_bytes: components.iter().map(|item| item.size_bytes).sum(),
                components,
                integrity: Integrity::Unverified {
                    reason: "configured_directory".to_owned(),
                },
            },
            &primary,
            false,
            assessor,
        );
        output.push((primary, model));
    }
    Ok(())
}

fn scan_interrupted(
    config: &InventoryConfig,
    output: &mut Vec<(PathBuf, InventoryModel)>,
) -> Result<(), InventoryError> {
    for entry in read_dir_sorted(&config.root.join("operations"))? {
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("json") {
            continue;
        }
        let manifest: OperationManifest = match fs::read(&path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<OperationManifest>(&bytes).ok())
        {
            Some(manifest)
                if manifest.version == MANIFEST_VERSION
                    && !manifest.components.is_empty()
                    && manifest.components.iter().all(|component| {
                        component.expected_size > 0
                            && component.content_key == blob_key(&component.content)
                            && crate::validation::validate_relative_path(&component.path).is_ok()
                    }) =>
            {
                manifest
            }
            _ => continue,
        };
        let snapshot = config
            .root
            .join("hub")
            .join(hf_repo_dir(&manifest.repository))
            .join("snapshots")
            .join(&manifest.commit);
        let components = manifest
            .components
            .iter()
            .map(|component| ModelComponent {
                path: component.path.clone(),
                role: component.role.clone(),
                size_bytes: component.expected_size,
                content: component.content.clone(),
                shard_index: component.shard_index,
                relationship: component.relationship.clone(),
            })
            .collect::<Vec<_>>();
        let total_bytes = manifest
            .components
            .iter()
            .map(|component| component.expected_size)
            .sum();
        let blobs = config
            .root
            .join("hub")
            .join(hf_repo_dir(&manifest.repository))
            .join("blobs");
        let mut completed_bytes = 0_u64;
        for component in &manifest.components {
            let blob = blobs.join(&component.content_key);
            if blob.metadata().is_ok_and(|metadata| {
                metadata.is_file() && metadata.len() == component.expected_size
            }) {
                completed_bytes = completed_bytes.saturating_add(component.expected_size);
                continue;
            }
            let partial = blobs.join(format!("{}.incomplete", component.content_key));
            if let Ok(metadata) = partial.symlink_metadata() {
                if !metadata.is_file() || metadata.len() > component.expected_size {
                    let quarantine = config.root.join("quarantine").join(format!(
                        "{}-{}-{}",
                        manifest.model_id.0,
                        component.content_key,
                        now()
                    ));
                    let _ = fs::rename(&partial, quarantine);
                } else {
                    completed_bytes = completed_bytes.saturating_add(metadata.len());
                }
            }
        }
        let primary = snapshot.join(
            manifest
                .components
                .first()
                .map(|item| item.path.as_path())
                .unwrap_or_else(|| Path::new("model.gguf")),
        );
        let model = InventoryModel {
            id: manifest.model_id.clone(),
            content_id: manifest.content_id,
            created: manifest.started_at,
            name: manifest.repository.clone(),
            supported_parameters: Vec::new(),
            status: ModelStatus::Interrupted {
                completed_bytes,
                total_bytes,
                resumable: true,
                reason: None,
                last_error: manifest
                    .last_error
                    .unwrap_or_else(|| "daemon_restarted".to_owned()),
                updated_at: manifest.updated_at,
            },
            source: ModelSource::HuggingFace {
                repository: manifest.repository,
                requested_revision: manifest.requested_revision,
                commit: manifest.commit,
                metadata: None,
            },
            location: ModelLocation::MagnitudeCache {
                components,
                total_bytes,
                integrity: Integrity::Unverified {
                    reason: "interrupted".to_owned(),
                },
            },
            properties: InventoryProperties::Pending,
            hardware: HardwareAssessment::NotAssessed {
                reason: "model_not_ready".to_owned(),
            },
            operations: Vec::new(),
            updated_at: manifest.updated_at,
        };
        output.push((primary, model));
    }
    Ok(())
}

// This construction boundary intentionally lists every independently acquired inventory field;
// grouping them would introduce an otherwise meaningless intermediate domain type.
#[allow(clippy::too_many_arguments)]
fn build_model(
    id: ModelId,
    content_id: icn_contracts::ContentId,
    created: u64,
    ready_at: u64,
    source: ModelSource,
    location: ModelLocation,
    primary: &Path,
    deletable: bool,
    assessor: Option<&dyn TemplateAssessor>,
) -> InventoryModel {
    let inspection = gguf::inspect(primary);
    let (name, properties, supported_parameters) = match inspection {
        Ok(inspection) => {
            let evidence = fingerprint(&inspection.fingerprint_material);
            let template = inspection.chat_template.as_deref().and_then(|template| {
                assessor.map(|assessor| assessor.assess(template, None, None))
            });
            let (tools, reasoning, template_evidence) = match template {
                Some(Ok(assessment)) => (
                    if assessment.capabilities.tools || assessment.capabilities.tool_calls {
                        CapabilitySupport::Supported {
                            parallel: Some(assessment.capabilities.parallel_tool_calls),
                        }
                    } else {
                        CapabilitySupport::Unsupported
                    },
                    assessment.reasoning,
                    Some(assessment.fingerprint),
                ),
                Some(Err(error)) => (
                    CapabilitySupport::Unknown {
                        reason: error.clone(),
                    },
                    ReasoningCapability::Unknown { reason: error },
                    None,
                ),
                None => (
                    CapabilitySupport::Unknown {
                        reason: "template_not_inspected".to_owned(),
                    },
                    ReasoningCapability::Unknown {
                        reason: "template_not_inspected".to_owned(),
                    },
                    None,
                ),
            };
            let name = inspection.name.clone().unwrap_or_else(|| {
                primary
                    .file_stem()
                    .and_then(|value| value.to_str())
                    .unwrap_or("local model")
                    .to_owned()
            });
            let mut supported_parameters = Vec::new();
            if matches!(tools, CapabilitySupport::Supported { .. }) {
                supported_parameters.push("tools".to_owned());
            }
            if matches!(
                reasoning,
                ReasoningCapability::Supported {
                    control: icn_contracts::ReasoningControlDomain::Effort { .. }
                        | icn_contracts::ReasoningControlDomain::EffortAndBudget { .. },
                    ..
                }
            ) {
                supported_parameters.push("reasoning_effort".to_owned());
            }
            let mut modalities = inspection.modalities;
            if location
                .components()
                .iter()
                .any(|component| component.role == ComponentRole::Projector)
                && !modalities.iter().any(|modality| modality == "image")
            {
                modalities.push("image".to_owned());
            }
            (
                name,
                InventoryProperties::Inspected {
                    architecture: inspection.architecture,
                    quantization: inspection.quantization,
                    parameter_count: inspection.parameter_count,
                    active_parameter_count: inspection.active_parameter_count,
                    training_context_length: inspection.training_context_length,
                    tokenizer: inspection.tokenizer,
                    modalities,
                    base_models: inspection.base_models,
                    tools,
                    structured_output: CapabilitySupport::Unknown {
                        reason: "template_not_inspected".to_owned(),
                    },
                    reasoning,
                    evidence_fingerprint: template_evidence.map_or(evidence.clone(), |template| {
                        format!("{evidence}+{template}")
                    }),
                },
                supported_parameters,
            )
        }
        Err(error) => (
            primary
                .file_stem()
                .and_then(|value| value.to_str())
                .unwrap_or("local model")
                .to_owned(),
            InventoryProperties::Unavailable {
                reason: error.to_string(),
            },
            Vec::new(),
        ),
    };
    let mut operations = vec![ModelOperation::Load, ModelOperation::Assess];
    if deletable {
        operations.push(ModelOperation::Delete);
    }
    InventoryModel {
        id,
        content_id,
        created,
        name,
        supported_parameters,
        status: ModelStatus::Available { ready_at },
        source,
        location,
        properties,
        hardware: HardwareAssessment::NotAssessed {
            reason: "not_requested".to_owned(),
        },
        operations,
        updated_at: ready_at,
    }
}

#[derive(Debug)]
struct ModelGroup {
    paths: Vec<PathBuf>,
    projector: Option<PathBuf>,
}

fn discover_groups(
    root: &Path,
    containment_root: &Path,
) -> Result<Vec<ModelGroup>, InventoryError> {
    let mut files = Vec::new();
    collect_gguf(root, containment_root, 0, &mut files)?;
    let mut groups: BTreeMap<PathBuf, Vec<(u32, u32, PathBuf)>> = BTreeMap::new();
    let mut standalone = Vec::new();
    let mut projectors = Vec::new();
    for path in files {
        let name = path
            .file_name()
            .and_then(|value| value.to_str())
            .unwrap_or_default()
            .to_ascii_lowercase();
        if name.contains("mmproj") || name.contains("projector") {
            projectors.push(path);
        } else if let Some((prefix, index, total)) = split_shard_name(&path) {
            groups.entry(prefix).or_default().push((index, total, path));
        } else {
            standalone.push(path);
        }
    }
    let mut output = standalone
        .into_iter()
        .map(|path| ModelGroup {
            projector: unique_projector_for(&path, &projectors),
            paths: vec![path],
        })
        .collect::<Vec<_>>();
    for (_prefix, mut shards) in groups {
        shards.sort_by_key(|(index, _, _)| *index);
        let total = shards.first().map(|(_, total, _)| *total).unwrap_or(0);
        if total == 0
            || shards.len() != total as usize
            || shards
                .iter()
                .enumerate()
                .any(|(offset, (index, candidate_total, _))| {
                    *index != offset as u32 + 1 || *candidate_total != total
                })
        {
            continue;
        }
        let first = shards[0].2.clone();
        output.push(ModelGroup {
            projector: unique_projector_for(&first, &projectors),
            paths: shards.into_iter().map(|(_, _, path)| path).collect(),
        });
    }
    Ok(output)
}

fn components_for_group(
    root: &Path,
    group: &ModelGroup,
) -> Result<Vec<ModelComponent>, InventoryError> {
    let mut components = Vec::new();
    for (offset, path) in group.paths.iter().enumerate() {
        let relative = path.strip_prefix(root).map_err(|_| {
            InventoryError::Io("discovered model escaped its configured root".to_owned())
        })?;
        let metadata = path.metadata().map_err(io_error)?;
        components.push(ModelComponent {
            path: relative.to_path_buf(),
            role: if group.paths.len() == 1 {
                ComponentRole::Weights
            } else {
                ComponentRole::Shard
            },
            size_bytes: metadata.len(),
            content: content_identity_for_file(path, &metadata),
            shard_index: (group.paths.len() > 1).then_some(offset as u32 + 1),
            relationship: None,
        });
    }
    if let Some(projector) = group.projector.as_ref() {
        let relative = projector.strip_prefix(root).map_err(|_| {
            InventoryError::Io("discovered projector escaped its configured root".to_owned())
        })?;
        let metadata = projector.metadata().map_err(io_error)?;
        components.push(ModelComponent {
            path: relative.to_path_buf(),
            role: ComponentRole::Projector,
            size_bytes: metadata.len(),
            content: content_identity_for_file(projector, &metadata),
            shard_index: None,
            relationship: components.first().map(|model| {
                icn_contracts::ComponentRelationship::ProjectorFor {
                    projector: relative.to_path_buf(),
                    model: model.path.clone(),
                }
            }),
        });
    }
    Ok(components)
}

fn collect_gguf(
    directory: &Path,
    containment_root: &Path,
    depth: usize,
    output: &mut Vec<PathBuf>,
) -> Result<(), InventoryError> {
    if depth > MAX_SCAN_DEPTH || output.len() >= MAX_SCAN_ENTRIES {
        return Ok(());
    }
    for entry in read_dir_sorted(directory)? {
        let path = entry.path();
        let file_type = entry.file_type().map_err(io_error)?;
        if file_type.is_symlink() {
            let canonical = match path.canonicalize() {
                Ok(canonical) if canonical.starts_with(containment_root) => canonical,
                _ => continue,
            };
            if canonical.is_file()
                && path.extension().and_then(|value| value.to_str()) == Some("gguf")
            {
                output.push(path);
            }
        } else if file_type.is_dir() {
            collect_gguf(&path, containment_root, depth + 1, output)?;
        } else if file_type.is_file()
            && path
                .extension()
                .and_then(|value| value.to_str())
                .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
        {
            output.push(path);
        }
        if output.len() >= MAX_SCAN_ENTRIES {
            break;
        }
    }
    Ok(())
}

fn split_shard_name(path: &Path) -> Option<(PathBuf, u32, u32)> {
    let name = path.file_name()?.to_str()?;
    let stem = name.strip_suffix(".gguf")?;
    let (left, total) = stem.rsplit_once("-of-")?;
    let (prefix, index) = left.rsplit_once('-')?;
    if index.len() != 5 || total.len() != 5 {
        return None;
    }
    let index = index.parse().ok()?;
    let total = total.parse().ok()?;
    Some((path.parent()?.join(prefix), index, total))
}

fn unique_projector_for(model: &Path, projectors: &[PathBuf]) -> Option<PathBuf> {
    let parent = model.parent()?;
    let matches = projectors
        .iter()
        .filter(|path| path.parent() == Some(parent))
        .collect::<Vec<_>>();
    (matches.len() == 1).then(|| matches[0].clone())
}

fn primary_path(root: &Path, components: &[ModelComponent]) -> Option<PathBuf> {
    components
        .iter()
        .find(|component| {
            matches!(
                component.role,
                ComponentRole::Weights | ComponentRole::Shard
            )
        })
        .map(|component| root.join(&component.path))
}

fn components_exist_contained(
    snapshot: &Path,
    repository_root: &Path,
    components: &[ModelComponent],
) -> bool {
    let canonical_repository = match repository_root.canonicalize() {
        Ok(root) => root,
        Err(_) => return false,
    };
    components.iter().all(|component| {
        let path = snapshot.join(&component.path);
        path.metadata()
            .is_ok_and(|metadata| metadata.is_file() && metadata.len() == component.size_bytes)
            && path
                .canonicalize()
                .is_ok_and(|canonical| canonical.starts_with(&canonical_repository))
    })
}

fn read_dir_sorted(path: &Path) -> Result<Vec<fs::DirEntry>, InventoryError> {
    if !path.is_dir() {
        return Ok(Vec::new());
    }
    let mut entries = fs::read_dir(path)
        .map_err(io_error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(io_error)?;
    entries.sort_by_key(fs::DirEntry::file_name);
    Ok(entries)
}

fn parse_hf_repo_dir(value: &str) -> Option<String> {
    let rest = value.strip_prefix("models--")?;
    let (owner, name) = rest.split_once("--")?;
    (!owner.is_empty() && !name.is_empty()).then(|| format!("{owner}/{name}"))
}

pub(crate) fn hf_repo_dir(repository: &str) -> String {
    format!("models--{}", repository.replace('/', "--"))
}

fn modified_seconds(path: &Path) -> Option<u64> {
    path.metadata()
        .ok()?
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|duration| duration.as_secs())
}

fn file_identity(path: &Path, metadata: &fs::Metadata) -> String {
    let mut digest = Sha256::new();
    digest.update(
        path.canonicalize()
            .unwrap_or_else(|_| path.to_path_buf())
            .to_string_lossy()
            .as_bytes(),
    );
    digest.update(metadata.len().to_le_bytes());
    if let Ok(modified) = metadata.modified()
        && let Ok(duration) = modified.duration_since(UNIX_EPOCH)
    {
        digest.update(duration.as_nanos().to_le_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn content_identity_for_file(path: &Path, metadata: &fs::Metadata) -> ContentIdentity {
    let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let in_blob_store = canonical
        .parent()
        .and_then(Path::file_name)
        .and_then(|value| value.to_str())
        == Some("blobs");
    let name = canonical.file_name().and_then(|value| value.to_str());
    if in_blob_store
        && let Some(value) = name
        && value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return ContentIdentity::Sha256 {
            value: value.to_ascii_lowercase(),
        };
    }
    if in_blob_store
        && let Some(value) = name
        && value.len() == 40
        && value.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return ContentIdentity::GitOid {
            value: value.to_ascii_lowercase(),
        };
    }
    ContentIdentity::FileIdentity {
        value: file_identity(path, metadata),
    }
}

pub(crate) fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn io_error(error: impl std::fmt::Display) -> InventoryError {
    InventoryError::Io(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_only_complete_split_names() {
        let first = Path::new("model-00001-of-00002.gguf");
        assert_eq!(
            split_shard_name(first).map(|(_, index, total)| (index, total)),
            Some((1, 2))
        );
        assert!(split_shard_name(Path::new("model-1-of-2.gguf")).is_none());
    }

    #[test]
    fn parses_hugging_face_cache_repository_directory() {
        assert_eq!(
            parse_hf_repo_dir("models--Qwen--Qwen3"),
            Some("Qwen/Qwen3".to_owned())
        );
        assert_eq!(parse_hf_repo_dir("datasets--owner--name"), None);
    }
}
