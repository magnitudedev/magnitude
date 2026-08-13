use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

use hf_hub::HFClient;
use icn_contracts::models::{
    InstalledModelPackage, InstalledModelPackagesResponse, ModelAssessment, ModelPackageId,
};
use icn_contracts::{
    CapabilitySupport, ComponentRole, ContentIdentity, DownloadFailure, EffectiveTemplateInputs,
    Integrity, InventoryError, InventoryModel, InventoryProperties, LocalDeclaration,
    ModelAvailability, ModelComponent, ModelId, ModelLocation, ModelOperation, ModelSource,
    ReasoningCapability, TemplateAssessor,
};
use icn_utils::file_cache::recover_map;
use sha2::{Digest, Sha256};

use crate::cache::{ModelCache, ModelIndexKind};
use crate::download::blob_key;
use crate::gguf;
use crate::identity::{content_id, fingerprint, model_id};
use crate::manifest::{MANIFEST_VERSION, ManagedManifest, OperationManifest};

const MAX_SCAN_ENTRIES: usize = 100_000;
const MAX_SCAN_DEPTH: usize = 8;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct CacheEvidence {
    content_id: String,
    observation_key: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct CachedModelInspection {
    pub(crate) name: String,
    pub(crate) properties: InventoryProperties,
    pub(crate) supported_parameters: Vec<String>,
}

#[derive(Debug, serde::Serialize)]
struct InventoryCache {
    models: BTreeMap<ModelId, InventoryModel>,
    evidence: BTreeMap<ModelId, CacheEvidence>,
    installed: InstalledPackageSnapshot,
}

type HydratedInventory = (
    BTreeMap<ModelId, InventoryModel>,
    BTreeMap<ModelId, CacheEvidence>,
    InstalledPackageSnapshot,
);

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct InstalledPackageRecord {
    pub(crate) installed: InstalledModelPackage,
    pub(crate) model: InventoryModel,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct InstalledPackageSnapshot {
    pub(crate) records: BTreeMap<ModelPackageId, InstalledPackageRecord>,
}

impl InstalledPackageSnapshot {
    pub(crate) fn response(
        &self,
        revision: u64,
        reconciliation_complete: bool,
    ) -> InstalledModelPackagesResponse {
        InstalledModelPackagesResponse {
            revision,
            reconciliation_complete,
            packages: self
                .records
                .values()
                .map(|record| record.installed.clone())
                .collect(),
        }
    }
}
#[derive(Debug, Clone)]
pub struct InventoryConfig {
    pub root: PathBuf,
    pub cache_root: PathBuf,
    pub hf_cache_dirs: Vec<PathBuf>,
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

    pub fn default_cache_root() -> Result<PathBuf, InventoryError> {
        let home = std::env::var_os("HOME").ok_or_else(|| {
            InventoryError::InvalidRequest(
                "cannot determine the user home directory for the cache".to_owned(),
            )
        })?;
        Ok(PathBuf::from(home).join(".magnitude/cache"))
    }

    pub fn with_roots(root: PathBuf, cache_root: PathBuf) -> Result<Self, InventoryError> {
        if !root.is_absolute() || !cache_root.is_absolute() {
            return Err(InventoryError::InvalidRequest(
                "model store and cache roots must be absolute".to_owned(),
            ));
        }
        Ok(Self {
            root,
            cache_root,
            hf_cache_dirs: Vec::new(),
            max_concurrent_downloads: 2,
            disk_reserve_bytes: 2 * 1024 * 1024 * 1024,
        })
    }
}

pub struct ModelManager {
    pub(crate) config: InventoryConfig,
    pub(crate) client: HFClient,
    pub(crate) http: reqwest::Client,
    pub(crate) models: Arc<RwLock<BTreeMap<ModelId, InventoryModel>>>,
    pub(crate) operations:
        Arc<tokio::sync::Mutex<BTreeMap<String, Arc<crate::download::DownloadOperation>>>>,
    pub(crate) download_slots: Arc<tokio::sync::Semaphore>,
    pub(crate) template_assessor: Option<Arc<dyn TemplateAssessor>>,
    pub(crate) cache: ModelCache,
    pub(crate) package_digests: Arc<RwLock<BTreeMap<PathBuf, (u64, SystemTime, String)>>>,
    pub(crate) installed_packages: Arc<RwLock<InstalledPackageSnapshot>>,
    pub(crate) model_assessments: Arc<RwLock<BTreeMap<String, ModelAssessment>>>,
    cache_evidence: Arc<RwLock<BTreeMap<ModelId, CacheEvidence>>>,
    ensure_gate: Arc<tokio::sync::Mutex<()>>,
    ensure_generation: Arc<AtomicU64>,
    reconciliation_running: Arc<AtomicBool>,
    reconciliation_complete: Arc<AtomicBool>,
}

struct ReconciliationLease(Arc<AtomicBool>);

impl Drop for ReconciliationLease {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

impl Clone for ModelManager {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            client: self.client.clone(),
            http: self.http.clone(),
            models: Arc::clone(&self.models),
            operations: Arc::clone(&self.operations),
            download_slots: Arc::clone(&self.download_slots),
            template_assessor: self.template_assessor.clone(),
            cache: self.cache.clone(),
            package_digests: Arc::clone(&self.package_digests),
            installed_packages: Arc::clone(&self.installed_packages),
            model_assessments: Arc::clone(&self.model_assessments),
            cache_evidence: Arc::clone(&self.cache_evidence),
            ensure_gate: Arc::clone(&self.ensure_gate),
            ensure_generation: Arc::clone(&self.ensure_generation),
            reconciliation_running: Arc::clone(&self.reconciliation_running),
            reconciliation_complete: Arc::clone(&self.reconciliation_complete),
        }
    }
}

impl ModelManager {
    pub(crate) fn installed_packages_response(
        &self,
    ) -> Result<InstalledModelPackagesResponse, InventoryError> {
        self.installed_packages
            .read()
            .map_err(|_| {
                InventoryError::Internal("installed package snapshot lock poisoned".to_owned())
            })
            .map(|snapshot| {
                snapshot.response(
                    self.ensure_generation.load(Ordering::Acquire),
                    self.reconciliation_complete.load(Ordering::Acquire),
                )
            })
    }

    #[must_use]
    pub fn derived_cache(&self) -> &ModelCache {
        &self.cache
    }

    pub async fn open(config: InventoryConfig) -> Result<Self, InventoryError> {
        Self::open_with_template_assessor(config, None).await
    }

    pub async fn open_with_template_assessor(
        config: InventoryConfig,
        template_assessor: Option<Arc<dyn TemplateAssessor>>,
    ) -> Result<Self, InventoryError> {
        validate_config(&config)?;
        create_layout(&config.root).await?;
        let reconciliation_root = config.root.clone();
        tokio::task::spawn_blocking(move || {
            crate::service::reconcile_tombstones(&reconciliation_root)
        })
        .await
        .map_err(|error| InventoryError::Internal(error.to_string()))??;
        let client_builder = HFClient::builder().cache_dir(config.root.join("hub"));
        let explicit_token = std::env::var("HF_TOKEN")
            .ok()
            .filter(|token| !token.trim().is_empty());
        let client_builder = match explicit_token {
            Some(token) => client_builder.token(token),
            None => client_builder,
        };
        let client = client_builder
            .build()
            .map_err(|error| InventoryError::Upstream(error.to_string()))?;
        let cache = ModelCache::new(&config.cache_root);
        let (models, cache_evidence, installed_packages) = load_inventory_index(&cache);
        let manager = Self {
            download_slots: Arc::new(tokio::sync::Semaphore::new(config.max_concurrent_downloads)),
            config,
            client,
            http: reqwest::Client::new(),
            models: Arc::new(RwLock::new(models)),
            operations: Arc::new(tokio::sync::Mutex::new(BTreeMap::new())),
            template_assessor,
            cache,
            package_digests: Arc::new(RwLock::new(BTreeMap::new())),
            installed_packages: Arc::new(RwLock::new(installed_packages)),
            model_assessments: Arc::new(RwLock::new(BTreeMap::new())),
            cache_evidence: Arc::new(RwLock::new(cache_evidence)),
            ensure_gate: Arc::new(tokio::sync::Mutex::new(())),
            ensure_generation: Arc::new(AtomicU64::new(0)),
            reconciliation_running: Arc::new(AtomicBool::new(false)),
            reconciliation_complete: Arc::new(AtomicBool::new(false)),
        };
        manager.request_installed_model_reconciliation();
        manager.start_installed_model_reconciler();
        Ok(manager)
    }

    fn start_installed_model_reconciler(&self) {
        let manager = self.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(5));
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            // Startup reconciliation was requested before this activity was admitted.
            interval.tick().await;
            loop {
                interval.tick().await;
                manager.request_installed_model_reconciliation();
            }
        });
    }

    pub(crate) fn request_installed_model_reconciliation(&self) {
        if self
            .reconciliation_running
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }
        let reconciliation = self.clone();
        tokio::spawn(async move {
            let _lease = ReconciliationLease(Arc::clone(&reconciliation.reconciliation_running));
            let result = reconciliation.ensure_installed_model_inventory().await;
            if let Err(error) = result {
                tracing::warn!(%error, "installed-model reconciliation failed");
            }
        });
    }

    pub(crate) fn cached_model_inspection(
        &self,
        content_id: &icn_contracts::ContentId,
        primary_name: &str,
        has_projector: bool,
    ) -> Option<CachedModelInspection> {
        let assessor = self.template_assessor.as_deref()?;
        let evidence = model_inspection_evidence(
            content_id,
            assessor.cache_identity(),
            primary_name,
            has_projector,
        )
        .ok()?;
        self.cache
            .read_index(ModelIndexKind::ArtifactInspection, &evidence)
    }

    pub async fn ensure_model_inventory(&self) -> Result<(), InventoryError> {
        self.reconcile_model_inventory().await
    }

    pub(crate) async fn ensure_installed_model_inventory(&self) -> Result<(), InventoryError> {
        self.reconcile_model_inventory().await
    }

    async fn reconcile_model_inventory(&self) -> Result<(), InventoryError> {
        let observed_generation = self.ensure_generation.load(Ordering::Acquire);
        let _guard = self.ensure_gate.lock().await;
        if self.ensure_generation.load(Ordering::Acquire) != observed_generation {
            return Ok(());
        }

        let live_models = self
            .models
            .read()
            .map_err(|_| InventoryError::Internal("inventory lock poisoned".to_owned()))?
            .clone();
        let mut attempt = 0_u8;
        let (discovered, next_evidence, installed_packages) = loop {
            let config = self.config.clone();
            let cache = self.cache.clone();
            let template_assessor = self.template_assessor.clone();
            let scan_live_models = live_models.clone();
            let scan_result = tokio::task::spawn_blocking(move || {
                scan(
                    &config,
                    &cache,
                    template_assessor.as_deref(),
                    &scan_live_models,
                )
            })
            .await
            .map_err(|error| InventoryError::Internal(error.to_string()))??;
            let discovered = scan_result.models;

            let mut next_evidence = BTreeMap::new();

            for model in discovered.values() {
                if !is_cacheable_model(model)? {
                    continue;
                }
                let evidence = CacheEvidence {
                    content_id: model.content_id.0.clone(),
                    observation_key: scan_result
                        .observations
                        .get(&model.id)
                        .cloned()
                        .ok_or_else(|| {
                            InventoryError::Internal(format!(
                                "ready model {} has no discovery observation",
                                model.id.0
                            ))
                        })?,
                };
                next_evidence.insert(model.id.clone(), evidence);
            }

            let package_manager = self.clone();
            let package_models = discovered.clone();
            let installed_packages = tokio::task::spawn_blocking(move || {
                package_manager.build_installed_package_snapshot(&package_models)
            })
            .await
            .map_err(|error| InventoryError::Internal(error.to_string()))??;

            if inventory_snapshot_is_current(
                &self.config.root,
                &discovered,
                &scan_result.observations,
            ) {
                break (discovered, next_evidence, installed_packages);
            }
            attempt += 1;
            if attempt >= 3 {
                return Err(InventoryError::ConcurrentMutation(
                    "model artifacts changed during three consecutive inventory attempts"
                        .to_owned(),
                ));
            }
        };

        persist_inventory_index(
            &self.cache,
            &discovered,
            &next_evidence,
            &installed_packages,
        );
        *self
            .models
            .write()
            .map_err(|_| InventoryError::Internal("inventory lock poisoned".to_owned()))? =
            discovered;
        *self
            .cache_evidence
            .write()
            .map_err(|_| InventoryError::Internal("inventory cache lock poisoned".to_owned()))? =
            next_evidence;
        *self.installed_packages.write().map_err(|_| {
            InventoryError::Internal("installed package snapshot lock poisoned".to_owned())
        })? = installed_packages;
        self.ensure_generation.fetch_add(1, Ordering::Release);
        self.reconciliation_complete.store(true, Ordering::Release);
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
            &self.cache,
            self.template_assessor.as_deref(),
        )?;
        if let Some(name) = display_name {
            model.name = name.to_owned();
        }
        self.complete_and_publish_model(model).await?;
        Ok(id)
    }

    pub(crate) async fn complete_and_publish_model(
        &self,
        model: InventoryModel,
    ) -> Result<InventoryModel, InventoryError> {
        let _guard = self.ensure_gate.lock().await;
        let evidence = is_cacheable_model(&model)?
            .then(|| {
                Ok::<_, InventoryError>(CacheEvidence {
                    content_id: model.content_id.0.clone(),
                    observation_key: model_observation_key(&self.config.root, &model)?,
                })
            })
            .transpose()?;
        let mut models = self
            .models
            .read()
            .map_err(|_| InventoryError::Internal("inventory lock poisoned".to_owned()))?
            .clone();
        let mut cache = self
            .cache_evidence
            .read()
            .map_err(|_| InventoryError::Internal("inventory cache lock poisoned".to_owned()))?
            .clone();
        models.insert(model.id.clone(), model.clone());
        if let Some(evidence) = evidence {
            cache.insert(model.id.clone(), evidence);
        } else {
            cache.remove(&model.id);
        }
        let installed = self.build_installed_package_snapshot(&models)?;
        persist_inventory_index(&self.cache, &models, &cache, &installed);
        *self
            .models
            .write()
            .map_err(|_| InventoryError::Internal("inventory lock poisoned".to_owned()))? = models;
        *self
            .cache_evidence
            .write()
            .map_err(|_| InventoryError::Internal("inventory cache lock poisoned".to_owned()))? =
            cache;
        *self.installed_packages.write().map_err(|_| {
            InventoryError::Internal("installed package snapshot lock poisoned".to_owned())
        })? = installed;
        self.ensure_generation.fetch_add(1, Ordering::Release);
        self.request_installed_model_reconciliation();
        Ok(model)
    }

    pub(crate) async fn remove_published_model(&self, id: &ModelId) -> Result<(), InventoryError> {
        let _guard = self.ensure_gate.lock().await;
        let mut models = self
            .models
            .read()
            .map_err(|_| InventoryError::Internal("inventory lock poisoned".to_owned()))?
            .clone();
        let mut cache = self
            .cache_evidence
            .read()
            .map_err(|_| InventoryError::Internal("inventory cache lock poisoned".to_owned()))?
            .clone();
        models.remove(id);
        cache.remove(id);
        let mut installed = self
            .installed_packages
            .read()
            .map_err(|_| {
                InventoryError::Internal("installed package snapshot lock poisoned".to_owned())
            })?
            .clone();
        installed.records.retain(|_, record| record.model.id != *id);
        persist_inventory_index(&self.cache, &models, &cache, &installed);
        *self
            .models
            .write()
            .map_err(|_| InventoryError::Internal("inventory lock poisoned".to_owned()))? = models;
        *self
            .cache_evidence
            .write()
            .map_err(|_| InventoryError::Internal("inventory cache lock poisoned".to_owned()))? =
            cache;
        *self.installed_packages.write().map_err(|_| {
            InventoryError::Internal("installed package snapshot lock poisoned".to_owned())
        })? = installed;
        self.ensure_generation.fetch_add(1, Ordering::Release);
        Ok(())
    }
}

fn is_cacheable_model(model: &InventoryModel) -> Result<bool, InventoryError> {
    match (&model.availability, &model.properties) {
        (ModelAvailability::Available { .. }, InventoryProperties::Inspected { .. }) => Ok(true),
        (ModelAvailability::InvalidArtifact { .. }, InventoryProperties::Unavailable { .. }) => {
            Ok(true)
        }
        (
            ModelAvailability::IncompatibleArtifact { .. },
            InventoryProperties::Unavailable { .. },
        ) => Ok(true),
        (ModelAvailability::Available { .. }, _) => Err(InventoryError::Internal(format!(
            "ready model {} has incomplete properties",
            model.id.0
        ))),
        (
            ModelAvailability::InvalidArtifact { .. }
            | ModelAvailability::IncompatibleArtifact { .. },
            _,
        ) => Err(InventoryError::Internal(format!(
            "unavailable model {} has inconsistent properties",
            model.id.0
        ))),
        _ => Ok(false),
    }
}

fn inventory_snapshot_is_current(
    root: &Path,
    models: &BTreeMap<ModelId, InventoryModel>,
    observations: &BTreeMap<ModelId, String>,
) -> bool {
    models
        .values()
        .filter(|model| {
            matches!(
                model.availability,
                ModelAvailability::Available { .. }
                    | ModelAvailability::InvalidArtifact { .. }
                    | ModelAvailability::IncompatibleArtifact { .. }
            )
        })
        .all(|model| {
            observations.get(&model.id).is_some_and(|observed| {
                model_observation_key(root, model).is_ok_and(|current| current == *observed)
            })
        })
}

fn load_inventory_index(cache: &ModelCache) -> HydratedInventory {
    let Some(mut index) = cache.read_inventory() else {
        return (
            BTreeMap::new(),
            BTreeMap::new(),
            InstalledPackageSnapshot::default(),
        );
    };
    let raw_models = recover_map::<InventoryModel>(index.remove("models"), MAX_SCAN_ENTRIES);
    let raw_evidence = recover_map::<CacheEvidence>(index.remove("evidence"), MAX_SCAN_ENTRIES);
    let installed = index
        .remove("installed")
        .and_then(|value| serde_json::from_value(value).ok())
        .unwrap_or_default();
    let mut models = BTreeMap::new();
    for (raw_id, model) in raw_models {
        let Ok(id) = ModelId::parse(raw_id) else {
            continue;
        };
        if model.id != id {
            continue;
        }
        models.insert(id, model);
    }
    let mut evidence = BTreeMap::new();
    for (raw_id, entry) in raw_evidence {
        let Ok(id) = ModelId::parse(raw_id) else {
            continue;
        };
        if !models.contains_key(&id) {
            continue;
        }
        evidence.insert(id, entry);
    }
    (models, evidence, installed)
}

fn persist_inventory_index(
    cache: &ModelCache,
    models: &BTreeMap<ModelId, InventoryModel>,
    evidence: &BTreeMap<ModelId, CacheEvidence>,
    installed: &InstalledPackageSnapshot,
) {
    cache.write_inventory(&InventoryCache {
        models: models.clone(),
        evidence: evidence.clone(),
        installed: installed.clone(),
    });
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
    if !config.cache_root.is_absolute() {
        return Err(InventoryError::InvalidRequest(
            "cache root must be absolute".to_owned(),
        ));
    }
    if config.max_concurrent_downloads == 0 {
        return Err(InventoryError::InvalidRequest(
            "max_concurrent_downloads must be positive".to_owned(),
        ));
    }
    for root in &config.hf_cache_dirs {
        if !root.is_absolute() {
            return Err(InventoryError::InvalidRequest(format!(
                "configured Hugging Face cache root must be absolute: {}",
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
    Ok(())
}

struct InventoryScan {
    models: BTreeMap<ModelId, InventoryModel>,
    observations: BTreeMap<ModelId, String>,
}

fn scan(
    config: &InventoryConfig,
    cache: &ModelCache,
    assessor: Option<&dyn TemplateAssessor>,
    live_models: &BTreeMap<ModelId, InventoryModel>,
) -> Result<InventoryScan, InventoryError> {
    let mut discovered = Vec::new();
    scan_managed(config, &mut discovered)?;

    let mut distinct_hf = BTreeSet::new();
    let mut roots = Vec::new();
    for cache in &config.hf_cache_dirs {
        let canonical = cache.canonicalize().unwrap_or_else(|_| cache.clone());
        if canonical != config.root.join("hub") && distinct_hf.insert(canonical.clone()) {
            roots.push(canonical);
        }
    }
    let concurrency = std::thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(1)
        .clamp(1, 8);
    for roots in roots.chunks(concurrency) {
        let root_results = std::thread::scope(|scope| {
            roots
                .iter()
                .map(|root| {
                    scope.spawn(move || {
                        let mut models = Vec::new();
                        scan_hf_cache(root, &mut models)?;
                        Ok::<_, InventoryError>(models)
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| {
                    handle.join().map_err(|_| {
                        InventoryError::Internal("model discovery worker panicked".to_owned())
                    })?
                })
                .collect::<Result<Vec<_>, InventoryError>>()
        })?;
        for models in root_results {
            discovered.extend(models);
        }
    }
    scan_interrupted(config, &mut discovered)?;

    let (mut cached_models, cached_evidence, _) =
        load_inventory_index(&ModelCache::new(&config.cache_root));
    // The durable entry controls cache validity. Overlay only transient runtime state for an entry
    // that independently survived durable schema validation.
    for (id, durable) in &mut cached_models {
        if let Some(live) = live_models
            .get(id)
            .filter(|live| live.content_id == durable.content_id)
        {
            *durable = live.clone();
        }
    }

    // Earlier external sources have higher precedence for the same canonical path. Managed
    // manifests retain their exact model identity even when two packages intentionally share a
    // primary weights file (for example, a weights-only package and a weights-plus-MTP package).
    let mut seen_paths = BTreeSet::new();
    let mut models = BTreeMap::new();
    let mut observations = BTreeMap::new();
    let mut stale = Vec::new();
    for candidate in discovered {
        let path = candidate.primary_path().to_path_buf();
        let canonical = path.canonicalize().unwrap_or(path);
        let path_is_distinct = match &candidate {
            DiscoveryCandidate::Artifact(candidate) => {
                if matches!(candidate.location, ModelLocation::MagnitudeCache { .. }) {
                    // Managed manifests are distinct by model identity, but still claim their
                    // canonical path so a later external-source scan cannot duplicate them.
                    seen_paths.insert(canonical);
                    true
                } else {
                    seen_paths.insert(canonical)
                }
            }
            DiscoveryCandidate::Record { .. } => seen_paths.insert(canonical),
        };
        if path_is_distinct {
            match candidate {
                DiscoveryCandidate::Artifact(candidate) => {
                    let observation_key = artifact_observation_key(
                        &config.root,
                        &candidate.source,
                        &candidate.location,
                    )?;
                    observations.insert(candidate.id.clone(), observation_key.clone());
                    if let Some(model) = reuse_inspection(
                        &candidate,
                        &observation_key,
                        &cached_models,
                        &cached_evidence,
                    ) {
                        models.insert(model.id.clone(), model);
                    } else {
                        stale.push(candidate);
                    }
                }
                DiscoveryCandidate::Record { model, .. } => {
                    models.insert(model.id.clone(), *model);
                }
            }
        }
    }

    // Enrichment is bounded across candidates, rather than being serialized by directory.
    for candidates in stale.chunks(concurrency) {
        let enriched = std::thread::scope(|scope| {
            candidates
                .iter()
                .map(|candidate| scope.spawn(move || enrich_candidate(candidate, cache, assessor)))
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| {
                    handle.join().map_err(|_| {
                        InventoryError::Internal("model enrichment worker panicked".to_owned())
                    })?
                })
                .collect::<Result<Vec<_>, InventoryError>>()
        })?;
        for model in enriched {
            models.insert(model.id.clone(), model);
        }
    }
    Ok(InventoryScan {
        models,
        observations,
    })
}

fn scan_managed(
    config: &InventoryConfig,
    output: &mut Vec<DiscoveryCandidate>,
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
        output.push(DiscoveryCandidate::Artifact(Box::new(ArtifactCandidate {
            id: manifest.model_id,
            content_id: manifest.content_id,
            created: manifest.created_at,
            ready_at: manifest.ready_at,
            source: ModelSource::HuggingFace {
                repository: manifest.repository,
                requested_revision: manifest.requested_revision,
                commit: manifest.commit,
                metadata: None,
            },
            location: ModelLocation::MagnitudeCache {
                total_bytes: manifest.components.iter().map(|item| item.size_bytes).sum(),
                components: manifest.components,
                integrity: Integrity::Verified {
                    method: "manifest".to_owned(),
                },
            },
            primary,
            deletable: true,
        })));
    }
    Ok(())
}

fn scan_hf_cache(cache: &Path, output: &mut Vec<DiscoveryCandidate>) -> Result<(), InventoryError> {
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
                output.push(DiscoveryCandidate::Artifact(Box::new(ArtifactCandidate {
                    id,
                    content_id: content,
                    created,
                    ready_at: created,
                    source: ModelSource::HuggingFace {
                        repository: repository.clone(),
                        requested_revision: commit.clone(),
                        commit: commit.clone(),
                        metadata: None,
                    },
                    location: ModelLocation::HuggingFaceCache {
                        cache_root: snapshot.clone(),
                        repository: repository.clone(),
                        commit: commit.clone(),
                        total_bytes: components.iter().map(|item| item.size_bytes).sum(),
                        components,
                        integrity: Integrity::Unverified {
                            reason: "external_cache".to_owned(),
                        },
                    },
                    primary,
                    deletable: false,
                })));
            }
        }
    }
    Ok(())
}

fn scan_interrupted(
    config: &InventoryConfig,
    output: &mut Vec<DiscoveryCandidate>,
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
            availability: ModelAvailability::Interrupted {
                completed_bytes,
                total_bytes,
                resumable: true,
                failure: manifest.failure.unwrap_or(DownloadFailure::Interrupted),
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
            operations: Vec::new(),
            updated_at: manifest.updated_at,
        };
        output.push(DiscoveryCandidate::Record {
            primary,
            model: Box::new(model),
        });
    }
    Ok(())
}

#[derive(Debug)]
struct ArtifactCandidate {
    id: ModelId,
    content_id: icn_contracts::ContentId,
    created: u64,
    ready_at: u64,
    source: ModelSource,
    location: ModelLocation,
    primary: PathBuf,
    deletable: bool,
}

#[derive(Debug)]
enum DiscoveryCandidate {
    Artifact(Box<ArtifactCandidate>),
    Record {
        primary: PathBuf,
        model: Box<InventoryModel>,
    },
}

impl DiscoveryCandidate {
    fn primary_path(&self) -> &Path {
        match self {
            Self::Artifact(candidate) => &candidate.primary,
            Self::Record { primary, .. } => primary,
        }
    }
}

// Discovery resolves stable identity and location before enrichment. An unchanged artifact can
// therefore reuse its persisted terminal inspection without reopening or reprobeing the GGUF.
fn reuse_inspection(
    candidate: &ArtifactCandidate,
    observation_key: &str,
    cached_models: &BTreeMap<ModelId, InventoryModel>,
    cached_evidence: &BTreeMap<ModelId, CacheEvidence>,
) -> Option<InventoryModel> {
    let reusable = cached_evidence
        .get(&candidate.id)
        .filter(|evidence| evidence.content_id == candidate.content_id.0)
        .filter(|evidence| evidence.observation_key == observation_key)
        .and_then(|_| cached_models.get(&candidate.id))
        .filter(|model| {
            matches!(
                (&model.availability, &model.properties),
                (
                    ModelAvailability::Available { .. },
                    InventoryProperties::Inspected { .. }
                ) | (
                    ModelAvailability::InvalidArtifact { .. }
                        | ModelAvailability::IncompatibleArtifact { .. },
                    InventoryProperties::Unavailable { .. },
                )
            )
        });
    reusable.map(|cached| {
        let mut model = cached.clone();
        model.content_id = candidate.content_id.clone();
        model.source = candidate.source.clone();
        model.location = candidate.location.clone();
        model.operations = match &model.availability {
            ModelAvailability::Available { .. } => {
                let mut operations = vec![ModelOperation::Load, ModelOperation::Unload];
                if candidate.deletable {
                    operations.push(ModelOperation::Delete);
                }
                operations
            }
            ModelAvailability::InvalidArtifact { .. }
            | ModelAvailability::IncompatibleArtifact { .. } => candidate
                .deletable
                .then_some(ModelOperation::Delete)
                .into_iter()
                .collect(),
            _ => unreachable!("only terminal discovery records are reusable"),
        };
        model
    })
}

fn enrich_candidate(
    candidate: &ArtifactCandidate,
    cache: &ModelCache,
    assessor: Option<&dyn TemplateAssessor>,
) -> Result<InventoryModel, InventoryError> {
    let model = build_model(
        candidate.id.clone(),
        candidate.content_id.clone(),
        candidate.created,
        candidate.ready_at,
        candidate.source.clone(),
        candidate.location.clone(),
        &candidate.primary,
        candidate.deletable,
        cache,
        assessor,
    )?;
    Ok(model)
}

// This construction boundary intentionally lists every independently acquired inventory field;
// grouping them would introduce an otherwise meaningless intermediate domain type.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_model(
    id: ModelId,
    content_id: icn_contracts::ContentId,
    created: u64,
    ready_at: u64,
    source: ModelSource,
    location: ModelLocation,
    primary: &Path,
    deletable: bool,
    cache: &ModelCache,
    assessor: Option<&dyn TemplateAssessor>,
) -> Result<InventoryModel, InventoryError> {
    let assessor = assessor.ok_or_else(|| {
        InventoryError::Internal("the model inventory has no template assessor".to_owned())
    })?;
    let primary_name = primary
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("local model");
    let has_projector = location
        .components()
        .iter()
        .any(|component| component.role == ComponentRole::Projector);
    let inspection_evidence = model_inspection_evidence(
        &content_id,
        assessor.cache_identity(),
        primary_name,
        has_projector,
    )?;
    let cached_inspection = cache.read_index::<CachedModelInspection>(
        ModelIndexKind::ArtifactInspection,
        &inspection_evidence,
    );
    let inspected = match cached_inspection {
        Some(inspection) => inspection,
        None => match gguf::inspect(primary) {
            Ok(inspection) => {
                let evidence = fingerprint(&inspection.fingerprint_material);
                let template = assessor.assess(&EffectiveTemplateInputs {
                    model_path: primary.to_path_buf(),
                });
                let (tools, reasoning, template_evidence) = match template {
                    Ok(assessment) => (
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
                    Err(error) => {
                        return Ok(unavailable_model(
                            id,
                            content_id,
                            created,
                            ready_at,
                            source,
                            location,
                            primary,
                            deletable,
                            "template_inspection_failed",
                            format!(
                                "template inspection failed for {}: {error}",
                                primary.display()
                            ),
                            false,
                        ));
                    }
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
                if has_projector && !modalities.iter().any(|modality| modality == "image") {
                    modalities.push("image".to_owned());
                }
                let inspected = CachedModelInspection {
                    name,
                    properties: InventoryProperties::Inspected {
                        architecture: inspection.architecture,
                        quantization: inspection.quantization,
                        quantization_name: inspection.quantization_name,
                        parameter_count: inspection.parameter_count,
                        active_parameter_count: inspection.active_parameter_count,
                        training_context_length: inspection.training_context_length,
                        tokenizer: inspection.tokenizer,
                        modalities,
                        base_models: inspection.base_models,
                        tools,
                        structured_output: CapabilitySupport::Supported { parallel: None },
                        reasoning,
                        evidence_fingerprint: template_evidence
                            .map_or(evidence.clone(), |template| {
                                format!("{evidence}+{template}")
                            }),
                    },
                    supported_parameters,
                };
                cache.write_index(
                    ModelIndexKind::ArtifactInspection,
                    &inspection_evidence,
                    &inspected,
                );
                inspected
            }
            Err(error) => {
                let incompatible = matches!(error, gguf::GgufError::UnsupportedVersion(_));
                return Ok(unavailable_model(
                    id,
                    content_id,
                    created,
                    ready_at,
                    source,
                    location,
                    primary,
                    deletable,
                    if incompatible {
                        "unsupported_gguf_version"
                    } else {
                        "invalid_gguf"
                    },
                    error.to_string(),
                    incompatible,
                ));
            }
        },
    };
    let mut operations = vec![ModelOperation::Load, ModelOperation::Unload];
    if deletable {
        operations.push(ModelOperation::Delete);
    }
    Ok(InventoryModel {
        id,
        content_id,
        created,
        name: inspected.name,
        supported_parameters: inspected.supported_parameters,
        availability: ModelAvailability::Available { ready_at },
        source,
        location,
        properties: inspected.properties,
        operations,
        updated_at: ready_at,
    })
}

fn model_inspection_evidence(
    content_id: &icn_contracts::ContentId,
    assessor_identity: &str,
    primary_name: &str,
    has_projector: bool,
) -> Result<String, InventoryError> {
    serde_json::to_string(&(
        &content_id.0,
        assessor_identity,
        primary_name,
        has_projector,
    ))
    .map_err(|error| InventoryError::Internal(error.to_string()))
}

#[allow(clippy::too_many_arguments)]
fn unavailable_model(
    id: ModelId,
    content_id: icn_contracts::ContentId,
    created: u64,
    detected_at: u64,
    source: ModelSource,
    location: ModelLocation,
    primary: &Path,
    deletable: bool,
    code: &str,
    message: String,
    incompatible: bool,
) -> InventoryModel {
    let availability = if incompatible {
        ModelAvailability::IncompatibleArtifact {
            detected_at,
            code: code.to_owned(),
            message: message.clone(),
        }
    } else {
        ModelAvailability::InvalidArtifact {
            detected_at,
            code: code.to_owned(),
            message: message.clone(),
        }
    };
    InventoryModel {
        id,
        content_id,
        created,
        name: primary
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("local model")
            .to_owned(),
        supported_parameters: Vec::new(),
        availability,
        source,
        location,
        properties: InventoryProperties::Unavailable { reason: message },
        operations: deletable
            .then_some(ModelOperation::Delete)
            .into_iter()
            .collect(),
        updated_at: detected_at,
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
        } else if is_execution_companion(&path, &name) {
            // Draft and MTP artifacts are not independently loadable chat models. They remain
            // available at their source path for an explicit paired configuration.
            continue;
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

fn is_execution_companion(path: &Path, name: &str) -> bool {
    gguf::inspect(path).is_ok_and(|inspection| inspection.execution_role.is_some())
        || is_execution_companion_name(name)
}

fn is_execution_companion_name(name: &str) -> bool {
    name.trim_end_matches(".gguf")
        .split(|character: char| !character.is_ascii_alphanumeric())
        .any(|part| matches!(part, "dflash" | "draft" | "eagle3" | "mtp"))
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

fn artifact_observation_key(
    inventory_root: &Path,
    source: &ModelSource,
    location: &ModelLocation,
) -> Result<String, InventoryError> {
    let base = match (location, source) {
        (
            ModelLocation::MagnitudeCache { .. },
            ModelSource::HuggingFace {
                repository, commit, ..
            },
        ) => inventory_root
            .join("hub")
            .join(hf_repo_dir(repository))
            .join("snapshots")
            .join(commit),
        (ModelLocation::HuggingFaceCache { cache_root, .. }, _) => cache_root.clone(),
        (ModelLocation::Directory { root, .. }, _) => root.clone(),
        (ModelLocation::File { path, .. }, _) => path
            .parent()
            .ok_or_else(|| InventoryError::Internal("ad-hoc model has no parent".to_owned()))?
            .to_path_buf(),
        _ => {
            return Err(InventoryError::Internal(
                "model source and location are inconsistent".to_owned(),
            ));
        }
    };
    let mut paths = location
        .components()
        .iter()
        .map(|component| match location {
            ModelLocation::File { path, .. } => path.clone(),
            _ => base.join(&component.path),
        })
        .collect::<Vec<_>>();
    paths.sort();
    let mut digest = Sha256::new();
    digest.update(b"magnitude-filesystem-observation-v1\0");
    for path in paths {
        let metadata = path.metadata().map_err(io_error)?;
        if !metadata.is_file() {
            return Err(InventoryError::Io(format!(
                "model component is not a regular file: {}",
                path.display()
            )));
        }
        digest.update(path.to_string_lossy().as_bytes());
        digest.update(b"\0");
        digest.update(file_identity(&path, &metadata).as_bytes());
        digest.update(b"\0");
    }
    Ok(format!("sha256:{:x}", digest.finalize()))
}

fn model_observation_key(
    inventory_root: &Path,
    model: &InventoryModel,
) -> Result<String, InventoryError> {
    artifact_observation_key(inventory_root, &model.source, &model.location)
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
    use std::io::Write;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

    use icn_contracts::models::{InstalledModelPackages, ModelPackageInspection};
    use icn_contracts::{
        CapabilityEvidence, ModelInventory, ReasoningControlDomain, ReasoningDelimiters,
        ReasoningVisibility, TemplateAssessment, TemplateCapabilities,
    };

    #[test]
    fn configured_store_does_not_adopt_host_hugging_face_caches() {
        let root = std::env::temp_dir().join("icn-owned-model-store");
        let config =
            InventoryConfig::with_roots(root, std::env::temp_dir().join("icn-owned-model-cache"))
                .expect("absolute model and cache roots");

        assert!(config.hf_cache_dirs.is_empty());
    }

    #[derive(Default)]
    struct CompleteTemplateAssessor {
        calls: AtomicUsize,
        active: AtomicUsize,
        max_active: AtomicUsize,
        delay: bool,
        reject_name: Option<&'static str>,
    }

    impl TemplateAssessor for CompleteTemplateAssessor {
        fn cache_identity(&self) -> &str {
            "complete-template-assessor:test"
        }

        fn assess(&self, inputs: &EffectiveTemplateInputs) -> Result<TemplateAssessment, String> {
            self.calls.fetch_add(1, AtomicOrdering::SeqCst);
            if self.reject_name.is_some_and(|name| {
                inputs
                    .model_path
                    .file_name()
                    .and_then(|value| value.to_str())
                    .is_some_and(|file_name| file_name.contains(name))
            }) {
                return Err("unsupported template".to_owned());
            }
            let active = self.active.fetch_add(1, AtomicOrdering::SeqCst) + 1;
            self.max_active.fetch_max(active, AtomicOrdering::SeqCst);
            if self.delay {
                std::thread::sleep(std::time::Duration::from_millis(30));
            }
            self.active.fetch_sub(1, AtomicOrdering::SeqCst);
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
                        fingerprint: "template-v1".to_owned(),
                    },
                },
                fingerprint: "template-v1".to_owned(),
            })
        }
    }

    fn write_minimal_gguf(path: &Path) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.resize(32, 0);
        fs::write(path, bytes).unwrap();
    }

    fn write_minimal_gguf_with_string_metadata(path: &Path, entries: &[(&str, &str)]) {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.extend_from_slice(&(entries.len() as u64).to_le_bytes());
        for (key, value) in entries {
            bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
            bytes.extend_from_slice(key.as_bytes());
            bytes.extend_from_slice(&8_u32.to_le_bytes());
            bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
            bytes.extend_from_slice(value.as_bytes());
        }
        let aligned = bytes.len().div_ceil(32) * 32;
        bytes.resize(aligned, 0);
        fs::write(path, bytes).unwrap();
    }

    fn create_hf_snapshot(root: &Path) -> (PathBuf, PathBuf) {
        let cache = root.join("hf-cache");
        let snapshot = cache
            .join("models--test--model")
            .join("snapshots")
            .join("0123456789abcdef");
        fs::create_dir_all(&snapshot).unwrap();
        (cache, snapshot)
    }

    #[test]
    fn managed_packages_keep_exact_identity_when_their_primary_path_is_shared() {
        let temporary = tempfile::tempdir().unwrap();
        let store = temporary.path().join("store");
        let cache = ModelCache::new(&temporary.path().join("cache"));
        let snapshot = store
            .join("hub")
            .join(hf_repo_dir("test/model"))
            .join("snapshots")
            .join("commit");
        fs::create_dir_all(store.join("installations")).unwrap();
        fs::create_dir_all(&snapshot).unwrap();
        write_minimal_gguf(&snapshot.join("weights.gguf"));
        write_minimal_gguf(&snapshot.join("mtp.gguf"));

        let weights = ModelComponent {
            path: PathBuf::from("weights.gguf"),
            role: ComponentRole::Weights,
            size_bytes: fs::metadata(snapshot.join("weights.gguf")).unwrap().len(),
            content: ContentIdentity::Unknown,
            shard_index: None,
            relationship: None,
        };
        let mtp = ModelComponent {
            path: PathBuf::from("mtp.gguf"),
            role: ComponentRole::Mtp,
            size_bytes: fs::metadata(snapshot.join("mtp.gguf")).unwrap().len(),
            content: ContentIdentity::Unknown,
            shard_index: None,
            relationship: Some(icn_contracts::ComponentRelationship::MtpFor {
                mtp: PathBuf::from("mtp.gguf"),
                model: PathBuf::from("weights.gguf"),
            }),
        };
        let manifest = |id: &str, content: &str, components: Vec<ModelComponent>| ManagedManifest {
            version: MANIFEST_VERSION,
            model_id: ModelId(id.to_owned()),
            content_id: icn_contracts::ContentId(content.to_owned()),
            repository: "test/model".to_owned(),
            requested_revision: "commit".to_owned(),
            commit: "commit".to_owned(),
            components,
            created_at: 1,
            ready_at: 2,
        };
        fs::write(
            store.join("installations/weights.json"),
            serde_json::to_vec(&manifest(
                "weights",
                "weights-content",
                vec![weights.clone()],
            ))
            .unwrap(),
        )
        .unwrap();
        fs::write(
            store.join("installations/weights-mtp.json"),
            serde_json::to_vec(&manifest(
                "weights-mtp",
                "weights-mtp-content",
                vec![weights, mtp],
            ))
            .unwrap(),
        )
        .unwrap();

        let config = InventoryConfig::with_roots(store, temporary.path().join("cache")).unwrap();
        let result = scan(
            &config,
            &cache,
            Some(&CompleteTemplateAssessor::default()),
            &BTreeMap::new(),
        )
        .unwrap();

        assert_eq!(result.models.len(), 2);
        assert!(result.models.contains_key(&ModelId("weights".to_owned())));
        assert!(
            result
                .models
                .contains_key(&ModelId("weights-mtp".to_owned()))
        );
    }

    #[tokio::test]
    async fn installed_package_listing_reports_discovered_package() {
        let temporary = tempfile::tempdir().unwrap();
        let store = temporary.path().join("store");
        let (hf_cache, source) = create_hf_snapshot(temporary.path());
        write_minimal_gguf(&source.join("model.gguf"));

        let mut config =
            InventoryConfig::with_roots(store, temporary.path().join("cache")).unwrap();
        config.hf_cache_dirs.push(hf_cache);
        let manager = ModelManager::open_with_template_assessor(
            config,
            Some(Arc::new(CompleteTemplateAssessor::default())),
        )
        .await
        .unwrap();
        manager.ensure_installed_model_inventory().await.unwrap();
        let installed = manager.list_installed().await.unwrap();

        assert_eq!(installed.packages.len(), 1);
        assert_eq!(
            installed.packages[0].origin,
            icn_contracts::models::ModelPackageInstallationOrigin::HuggingFaceCache
        );
        let assessed = manager.list().await.unwrap();
        assert_eq!(assessed.len(), 1);
    }

    #[tokio::test]
    async fn completed_publication_updates_installed_snapshot_before_returning() {
        let temporary = tempfile::tempdir().unwrap();
        let model_path = temporary.path().join("active.gguf");
        write_minimal_gguf(&model_path);
        let manager = ModelManager::open_with_template_assessor(
            InventoryConfig::with_roots(
                temporary.path().join("store"),
                temporary.path().join("cache"),
            )
            .unwrap(),
            Some(Arc::new(CompleteTemplateAssessor::default())),
        )
        .await
        .unwrap();

        manager
            .register_active_model(&model_path, None)
            .await
            .unwrap();

        assert_eq!(manager.list_installed().await.unwrap().packages.len(), 1);
    }

    #[tokio::test]
    async fn installed_package_reconciliation_refreshes_hugging_face_cache_after_startup() {
        let temporary = tempfile::tempdir().unwrap();
        let store = temporary.path().join("store");
        let (hf_cache, source) = create_hf_snapshot(temporary.path());
        write_minimal_gguf(&source.join("first.gguf"));

        let mut config =
            InventoryConfig::with_roots(store, temporary.path().join("cache")).unwrap();
        config.hf_cache_dirs.push(hf_cache);
        let manager = ModelManager::open_with_template_assessor(
            config,
            Some(Arc::new(CompleteTemplateAssessor::default())),
        )
        .await
        .unwrap();
        manager.ensure_installed_model_inventory().await.unwrap();
        assert_eq!(manager.list_installed().await.unwrap().packages.len(), 1);

        write_minimal_gguf_with_string_metadata(
            &source.join("second.gguf"),
            &[("general.name", "second")],
        );
        manager.ensure_installed_model_inventory().await.unwrap();
        let packages = manager.list_installed().await.unwrap().packages;

        assert_eq!(packages.len(), 2);
    }

    #[tokio::test]
    async fn installed_package_query_never_waits_for_reconciliation() {
        let temporary = tempfile::tempdir().unwrap();
        let manager = ModelManager::open(
            InventoryConfig::with_roots(
                temporary.path().join("store"),
                temporary.path().join("cache"),
            )
            .unwrap(),
        )
        .await
        .unwrap();
        let _reconciliation = manager.ensure_gate.lock().await;

        let snapshot = tokio::time::timeout(
            std::time::Duration::from_millis(100),
            manager.list_installed(),
        )
        .await
        .expect("snapshot query must not wait for reconciliation")
        .unwrap();

        assert!(snapshot.packages.is_empty());
    }

    #[tokio::test]
    async fn template_failure_isolated_to_the_affected_installed_model() {
        let temporary = tempfile::tempdir().unwrap();
        let store = temporary.path().join("store");
        let (hf_cache, source) = create_hf_snapshot(temporary.path());
        write_minimal_gguf(&source.join("working.gguf"));
        write_minimal_gguf(&source.join("broken.gguf"));
        fs::OpenOptions::new()
            .append(true)
            .open(source.join("broken.gguf"))
            .unwrap()
            .write_all(&[0])
            .unwrap();

        let mut config =
            InventoryConfig::with_roots(store, temporary.path().join("cache")).unwrap();
        config.hf_cache_dirs.push(hf_cache);
        let manager = ModelManager::open_with_template_assessor(
            config,
            Some(Arc::new(CompleteTemplateAssessor {
                reject_name: Some("broken"),
                ..CompleteTemplateAssessor::default()
            })),
        )
        .await
        .unwrap();

        manager.ensure_installed_model_inventory().await.unwrap();
        let installed = manager.list_installed().await.unwrap();

        assert_eq!(installed.packages.len(), 2);
        assert_eq!(
            installed
                .packages
                .iter()
                .filter(|package| {
                    matches!(package.inspection, ModelPackageInspection::Inspected { .. })
                })
                .count(),
            1,
        );
        assert_eq!(
            installed
                .packages
                .iter()
                .filter(|package| {
                    matches!(package.inspection, ModelPackageInspection::Invalid { .. })
                })
                .count(),
            1,
        );
    }

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
    fn excludes_execution_companions_from_standalone_model_groups() {
        let temporary = tempfile::tempdir().unwrap();
        write_minimal_gguf(&temporary.path().join("laguna-s-2.1-Q4_K_M.gguf"));
        write_minimal_gguf_with_string_metadata(
            &temporary.path().join("unlabelled-laguna-companion.gguf"),
            &[("dflash.decoder_arch", "laguna")],
        );
        write_minimal_gguf_with_string_metadata(
            &temporary.path().join("unlabelled-eagle-companion.gguf"),
            &[("general.architecture", "eagle3")],
        );
        write_minimal_gguf(&temporary.path().join("qwen-MTP-BF16.gguf"));

        let groups = discover_groups(temporary.path(), temporary.path()).unwrap();

        assert_eq!(groups.len(), 1);
        assert_eq!(
            groups[0].paths[0]
                .file_name()
                .and_then(|value| value.to_str()),
            Some("laguna-s-2.1-Q4_K_M.gguf"),
        );
        assert!(is_execution_companion_name("laguna-s-2.1-dflash-bf16.gguf"));
        assert!(is_execution_companion_name("eagle3-qwen-4b.gguf"));
        assert!(is_execution_companion_name("qwen-mtp-bf16.gguf"));
        assert!(!is_execution_companion_name("draftsmanship-q4.gguf"));
    }

    #[test]
    fn parses_hugging_face_cache_repository_directory() {
        assert_eq!(
            parse_hf_repo_dir("models--Qwen--Qwen3"),
            Some("Qwen/Qwen3".to_owned())
        );
        assert_eq!(parse_hf_repo_dir("datasets--owner--name"), None);
    }

    #[tokio::test]
    async fn list_is_complete_shared_and_reuses_inspection_evidence() {
        let temporary = tempfile::tempdir().unwrap();
        let store = temporary.path().join("store");
        let cache_root = temporary.path().join("cache");
        let (hf_cache, source) = create_hf_snapshot(temporary.path());
        write_minimal_gguf(&source.join("model.gguf"));

        let mut config = InventoryConfig::with_roots(store.clone(), cache_root.clone()).unwrap();
        config.hf_cache_dirs.push(hf_cache);
        let template = Arc::new(CompleteTemplateAssessor::default());
        let manager =
            ModelManager::open_with_template_assessor(config.clone(), Some(template.clone()))
                .await
                .unwrap();
        assert!(manager.models.read().unwrap().is_empty());
        manager.ensure_model_inventory().await.unwrap();
        let (first, second) = tokio::join!(manager.list(), manager.list());
        let first = first.unwrap();
        assert_eq!(first, second.unwrap());
        assert_eq!(first.len(), 1);
        assert!(matches!(
            first[0].availability,
            ModelAvailability::Available { .. }
        ));
        assert!(matches!(
            first[0].properties,
            InventoryProperties::Inspected { .. }
        ));
        assert_eq!(template.calls.load(AtomicOrdering::SeqCst), 1);
        let persisted_bytes = fs::read(cache_root.join("indexes/inventory.json")).unwrap();
        let persisted: serde_json::Value = serde_json::from_slice(&persisted_bytes).unwrap();
        assert!(persisted.get("version").is_none());

        let reopened =
            ModelManager::open_with_template_assessor(config.clone(), Some(template.clone()))
                .await
                .unwrap();
        let warm = reopened.list().await.unwrap();
        assert_eq!(warm.len(), 1);
        assert_eq!(template.calls.load(AtomicOrdering::SeqCst), 1);

        // A changed identity is the only candidate enriched on the next reconciliation.
        fs::OpenOptions::new()
            .append(true)
            .open(source.join("model.gguf"))
            .unwrap()
            .write_all(&[0])
            .unwrap();
        reopened.ensure_model_inventory().await.unwrap();
        let changed = reopened.list().await.unwrap();
        assert_eq!(changed.len(), 1);
        assert_eq!(template.calls.load(AtomicOrdering::SeqCst), 2);
    }

    #[tokio::test]
    async fn malformed_index_entry_is_isolated_and_stale_candidates_enrich_in_parallel() {
        let temporary = tempfile::tempdir().unwrap();
        let store = temporary.path().join("store");
        let cache_root = temporary.path().join("cache");
        let (hf_cache, source) = create_hf_snapshot(temporary.path());
        write_minimal_gguf(&source.join("first.gguf"));
        write_minimal_gguf(&source.join("second.gguf"));

        let mut config = InventoryConfig::with_roots(store.clone(), cache_root.clone()).unwrap();
        config.hf_cache_dirs.push(hf_cache);
        let template = Arc::new(CompleteTemplateAssessor {
            delay: true,
            ..CompleteTemplateAssessor::default()
        });
        let manager =
            ModelManager::open_with_template_assessor(config.clone(), Some(template.clone()))
                .await
                .unwrap();
        manager.ensure_installed_model_inventory().await.unwrap();
        assert_eq!(manager.list().await.unwrap().len(), 2);
        assert!(template.max_active.load(AtomicOrdering::SeqCst) > 1);

        let index_path = cache_root.join("indexes/inventory.json");
        let mut index: serde_json::Value =
            serde_json::from_slice(&fs::read(&index_path).unwrap()).unwrap();
        let models = index
            .get_mut("models")
            .and_then(serde_json::Value::as_object_mut)
            .unwrap();
        let malformed_id = models.keys().next().unwrap().clone();
        models.insert(malformed_id, serde_json::json!({ "invalid": true }));
        fs::write(&index_path, serde_json::to_vec_pretty(&index).unwrap()).unwrap();

        let reopened =
            ModelManager::open_with_template_assessor(config.clone(), Some(template.clone()))
                .await
                .unwrap();
        reopened.ensure_installed_model_inventory().await.unwrap();
        assert_eq!(reopened.list().await.unwrap().len(), 2);
        assert_eq!(template.calls.load(AtomicOrdering::SeqCst), 2);

        let inspection_dir = cache_root.join("indexes/inspections/artifacts");
        let one_inspection = fs::read_dir(&inspection_dir)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        fs::write(one_inspection, b"not json").unwrap();
        fs::write(&index_path, b"not json").unwrap();
        let corrupted =
            ModelManager::open_with_template_assessor(config.clone(), Some(template.clone()))
                .await
                .unwrap();
        corrupted.ensure_installed_model_inventory().await.unwrap();
        assert_eq!(corrupted.list().await.unwrap().len(), 2);
        assert_eq!(template.calls.load(AtomicOrdering::SeqCst), 3);

        fs::remove_file(&index_path).unwrap();
        fs::create_dir(&index_path).unwrap();
        let uncached = ModelManager::open_with_template_assessor(config, Some(template.clone()))
            .await
            .unwrap();
        uncached.ensure_installed_model_inventory().await.unwrap();
        assert_eq!(uncached.list().await.unwrap().len(), 2);
        assert_eq!(template.calls.load(AtomicOrdering::SeqCst), 3);
    }
}
