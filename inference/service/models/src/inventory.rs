use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, RwLock, Weak};
use std::time::{SystemTime, UNIX_EPOCH};

use hf_hub::HFClient;
use magnitude_service_contracts::models::{
    CatalogPackageRole, CustomModelDeclarationOrigin, InstalledModelPackage,
    InstalledModelPackagesResponse, LocalModelsConfiguration, ModelFileRelationship, ModelFileRole,
    ModelPackage, ModelPackageId, ModelPackageInstallationOrigin, ModelPackageSource, ModelRoot,
    ModelRootKind, RecommendableModel, ServableModelBundle, SpeculativeDraftSource,
};
use magnitude_service_contracts::{
    ComponentRole, ContentIdentity, HubCacheOwner, Integrity, InventoryEntryId, InventoryError,
    InventoryModel,
    InventoryProperties, LocalDeclaration, ModelAvailability, ModelComponent, ModelLocation,
    ModelOperation, ModelSource,
};
use magnitude_service_utils::file_cache::recover_map;
use sha2::{Digest, Sha256};

use crate::cache::{ModelCache, ModelIndexKind};
use crate::catalog_affiliations::CatalogAffiliations;
use crate::download::blob_key;
use crate::gguf;
use crate::hugging_face::HubCredential;
use crate::identity::{content_id, fingerprint, inventory_entry_id};
use crate::package_service::PrimaryEvidence;
use crate::snapshot_blobs::{RepositoryBlob, RepositoryBlobs};
use crate::store_fs::ensure_store_layout;

const MAX_SCAN_ENTRIES: usize = 100_000;
const MAX_SCAN_DEPTH: usize = 8;
const BACKGROUND_RECONCILIATION_RETRY_DELAY: std::time::Duration =
    std::time::Duration::from_secs(1);

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
struct CacheEvidence {
    content_id: String,
    observation_key: String,
    metadata_key: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct CachedModelMetadata {
    pub(crate) name: String,
    pub(crate) properties: InventoryProperties,
    pub(crate) supported_parameters: Vec<String>,
}

#[derive(Debug, serde::Serialize)]
struct InventoryCache {
    models: BTreeMap<InventoryEntryId, InventoryModel>,
    evidence: BTreeMap<InventoryEntryId, CacheEvidence>,
    installed: InstalledPackageSnapshot,
}

type HydratedInventory = (
    BTreeMap<InventoryEntryId, InventoryModel>,
    BTreeMap<InventoryEntryId, CacheEvidence>,
    InstalledPackageSnapshot,
);

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct InstalledPackageRecord {
    pub(crate) installed: InstalledModelPackage,
    pub(crate) model: InventoryModel,
    pub(crate) primary: Option<PrimaryEvidence>,
}

#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub(crate) struct InstalledPackageSnapshot {
    pub(crate) records: BTreeMap<InventoryEntryId, InstalledPackageRecord>,
}

impl InstalledPackageSnapshot {
    pub(crate) fn response(
        &self,
        revision: u64,
        reconciliation_complete: bool,
    ) -> InstalledModelPackagesResponse {
        let packages = self.records.values().fold(
            BTreeMap::<ModelPackageId, InstalledModelPackage>::new(),
            |mut packages, record| {
                packages
                    .entry(record.installed.package.id.clone())
                    .and_modify(|selected| {
                        if origin_precedence(record.installed.origin)
                            < origin_precedence(selected.origin)
                        {
                            *selected = record.installed.clone();
                        }
                    })
                    .or_insert_with(|| record.installed.clone());
                packages
            },
        );
        InstalledModelPackagesResponse {
            revision,
            reconciliation_complete,
            packages: packages.into_values().collect(),
        }
    }
}

/// Which occurrence of the same content represents it: the managed store, then the Hugging Face
/// cache, LM Studio, llama.cpp, Ollama, user folders, and imported files.
pub(crate) fn origin_precedence(origin: ModelPackageInstallationOrigin) -> u8 {
    match origin {
        ModelPackageInstallationOrigin::Magnitude => 0,
        ModelPackageInstallationOrigin::HuggingFaceCache => 1,
        ModelPackageInstallationOrigin::LmStudio => 2,
        ModelPackageInstallationOrigin::LlamaCpp => 3,
        ModelPackageInstallationOrigin::Ollama => 4,
        ModelPackageInstallationOrigin::Folder => 5,
        ModelPackageInstallationOrigin::Imported => 6,
    }
}

/// Inventory settings. `local_models` is the local-models configuration in effect before ACN
/// supplies one at runtime: the model roots to scan, declarations, hidden models and drafters.
#[derive(Debug, Clone)]
pub struct InventoryConfig {
    pub root: PathBuf,
    pub cache_root: PathBuf,
    pub local_models: LocalModelsConfiguration,
    pub max_concurrent_downloads: usize,
    pub disk_reserve_bytes: u64,
    pub catalog_models: Vec<RecommendableModel>,
    pub catalog_header_digests: BTreeMap<String, String>,
    pub hub_endpoint: Option<String>,
}

pub(crate) fn catalog_target(model: &RecommendableModel) -> &ModelPackage {
    match &model.configuration.bundle {
        ServableModelBundle::Standalone { package } => package,
        ServableModelBundle::SpeculativeDecoding { target, .. } => target,
    }
}

pub(crate) fn catalog_packages(
    model: &RecommendableModel,
) -> impl Iterator<Item = (&ModelPackage, CatalogPackageRole)> {
    let target = std::iter::once((catalog_target(model), CatalogPackageRole::Target));
    let dependency = match &model.configuration.bundle {
        ServableModelBundle::SpeculativeDecoding {
            draft_source: SpeculativeDraftSource::Separate { draft },
            ..
        } => Some((draft, CatalogPackageRole::Dependency)),
        ServableModelBundle::Standalone { .. }
        | ServableModelBundle::SpeculativeDecoding {
            draft_source: SpeculativeDraftSource::Embedded,
            ..
        } => None,
    };
    target.chain(dependency)
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

    /// Adds enabled Hugging Face cache locations to the initial configuration.
    pub fn add_hugging_face_caches(&mut self, paths: impl IntoIterator<Item = PathBuf>) {
        self.local_models
            .roots
            .extend(paths.into_iter().map(|path| ModelRoot {
                kind: ModelRootKind::HuggingFaceCache,
                path,
                enabled: true,
            }));
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
            local_models: LocalModelsConfiguration::default(),
            max_concurrent_downloads: 2,
            disk_reserve_bytes: 2 * 1024 * 1024 * 1024,
            catalog_models: Vec::new(),
            catalog_header_digests: BTreeMap::new(),
            hub_endpoint: None,
        })
    }
}

pub struct ManagedModelStore {
    pub(crate) config: InventoryConfig,
    pub(crate) client: HFClient,
    pub(crate) hub_credential: Option<HubCredential>,
    pub(crate) http: reqwest::Client,
    pub(crate) models: Arc<RwLock<BTreeMap<InventoryEntryId, InventoryModel>>>,
    pub(crate) operations:
        Arc<tokio::sync::Mutex<BTreeMap<String, Arc<crate::download::DownloadOperation>>>>,
    pub(crate) download_slots: Arc<tokio::sync::Semaphore>,
    pub(crate) cache: ModelCache,
    pub(crate) local_models: Arc<RwLock<LocalModelsConfiguration>>,
    root_scans: Arc<std::sync::Mutex<RootScans>>,
    pub(crate) installed_packages: Arc<RwLock<InstalledPackageSnapshot>>,
    pub(crate) model_assessments:
        Arc<RwLock<BTreeMap<String, crate::cache::CachedModelAssessment>>>,
    pub(crate) catalog_affiliations: Arc<RwLock<CatalogAffiliations>>,
    pub(crate) catalog_affiliations_dirty: Arc<AtomicBool>,
    cache_evidence: Arc<RwLock<BTreeMap<InventoryEntryId, CacheEvidence>>>,
    ensure_gate: Arc<tokio::sync::Mutex<()>>,
    ensure_generation: Arc<AtomicU64>,
    /// Advances only when the published installed packages change, so a rescan that finds
    /// nothing new leaves every revision derived from them current.
    packages_revision: Arc<AtomicU64>,
    reconciliation_running: Arc<AtomicBool>,
    reconciliation_complete: Arc<AtomicBool>,
    installed_packages_observer: Arc<RwLock<Option<Weak<dyn InstalledPackagesObserver>>>>,
    ollama_hub_resolution: Arc<AtomicBool>,
}

pub(crate) trait InstalledPackagesObserver: Send + Sync {
    fn installed_packages_changed(&self, revision: u64);
}

struct ReconciliationLease(Arc<AtomicBool>);

impl Drop for ReconciliationLease {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

impl Clone for ManagedModelStore {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            client: self.client.clone(),
            hub_credential: self.hub_credential.clone(),
            http: self.http.clone(),
            models: Arc::clone(&self.models),
            operations: Arc::clone(&self.operations),
            download_slots: Arc::clone(&self.download_slots),
            cache: self.cache.clone(),
            local_models: Arc::clone(&self.local_models),
            root_scans: Arc::clone(&self.root_scans),
            installed_packages: Arc::clone(&self.installed_packages),
            model_assessments: Arc::clone(&self.model_assessments),
            catalog_affiliations: Arc::clone(&self.catalog_affiliations),
            catalog_affiliations_dirty: Arc::clone(&self.catalog_affiliations_dirty),
            cache_evidence: Arc::clone(&self.cache_evidence),
            ensure_gate: Arc::clone(&self.ensure_gate),
            ensure_generation: Arc::clone(&self.ensure_generation),
            packages_revision: Arc::clone(&self.packages_revision),
            reconciliation_running: Arc::clone(&self.reconciliation_running),
            reconciliation_complete: Arc::clone(&self.reconciliation_complete),
            installed_packages_observer: Arc::clone(&self.installed_packages_observer),
            ollama_hub_resolution: Arc::clone(&self.ollama_hub_resolution),
        }
    }
}

impl ManagedModelStore {
    pub(crate) fn set_installed_packages_observer(
        &self,
        observer: Weak<dyn InstalledPackagesObserver>,
    ) -> Result<(), InventoryError> {
        *self.installed_packages_observer.write().map_err(|_| {
            InventoryError::Internal("installed-package observer lock poisoned".to_owned())
        })? = Some(observer);
        self.notify_installed_packages_changed();
        Ok(())
    }

    fn notify_installed_packages_changed(&self) {
        let observer = self
            .installed_packages_observer
            .read()
            .ok()
            .and_then(|observer| observer.as_ref().and_then(Weak::upgrade));
        if let Some(observer) = observer {
            observer.installed_packages_changed(self.packages_revision.load(Ordering::Acquire));
        }
    }

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
                    self.packages_revision.load(Ordering::Acquire),
                    self.reconciliation_complete.load(Ordering::Acquire),
                )
            })
    }

    #[must_use]
    pub fn derived_cache(&self) -> &ModelCache {
        &self.cache
    }

    pub async fn open(config: InventoryConfig) -> Result<Self, InventoryError> {
        validate_config(&config)?;
        ensure_store_layout(&config.root).await?;
        // The engine runs with implicit Hub credentials disabled, so this client is anonymous; only
        // an explicit HF_TOKEN is presented, through the credentialed client.
        let client_builder = || {
            let builder = HFClient::builder().cache_dir(config.root.join("hub"));
            match &config.hub_endpoint {
                Some(endpoint) => builder.endpoint(endpoint.clone()),
                None => builder,
            }
        };
        let client = client_builder()
            .build()
            .map_err(|error| InventoryError::Upstream(error.to_string()))?;
        let hub_credential = std::env::var("HF_TOKEN")
            .ok()
            .filter(|token| !token.trim().is_empty())
            .map(|token| {
                client_builder()
                    .token(token.clone())
                    .build()
                    .map(|client| HubCredential { client, token })
                    .map_err(|error| InventoryError::Upstream(error.to_string()))
            })
            .transpose()?;
        let cache = ModelCache::new(&config.cache_root);
        let catalog_affiliations = CatalogAffiliations::load(&config.root);
        let local_models = Arc::new(RwLock::new(config.local_models.clone()));
        let manager = Self {
            download_slots: Arc::new(tokio::sync::Semaphore::new(config.max_concurrent_downloads)),
            config,
            client,
            hub_credential,
            http: reqwest::Client::new(),
            models: Arc::new(RwLock::new(BTreeMap::new())),
            operations: Arc::new(tokio::sync::Mutex::new(BTreeMap::new())),
            cache,
            local_models,
            root_scans: Arc::default(),
            installed_packages: Arc::new(RwLock::new(InstalledPackageSnapshot::default())),
            model_assessments: Arc::new(RwLock::new(BTreeMap::new())),
            catalog_affiliations: Arc::new(RwLock::new(catalog_affiliations)),
            catalog_affiliations_dirty: Arc::new(AtomicBool::new(false)),
            cache_evidence: Arc::new(RwLock::new(BTreeMap::new())),
            ensure_gate: Arc::new(tokio::sync::Mutex::new(())),
            ensure_generation: Arc::new(AtomicU64::new(0)),
            packages_revision: Arc::new(AtomicU64::new(0)),
            reconciliation_running: Arc::new(AtomicBool::new(false)),
            reconciliation_complete: Arc::new(AtomicBool::new(false)),
            installed_packages_observer: Arc::new(RwLock::new(None)),
            ollama_hub_resolution: Arc::new(AtomicBool::new(false)),
        };
        manager.request_installed_model_reconciliation();
        Ok(manager)
    }

    /// Resolves the Hub files of unresolved `hf.co` Ollama pulls in the background, and rescans the
    /// Ollama locations when any resolved, so their models take their Hub identity.
    fn request_ollama_hub_resolution(&self) {
        if self
            .ollama_hub_resolution
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return;
        }
        let store = self.clone();
        tokio::spawn(async move {
            let _lease = ReconciliationLease(Arc::clone(&store.ollama_hub_resolution));
            let roots = store.resolve_ollama_hub_files().await;
            if !roots.is_empty()
                && let Err(error) = store.rescan_model_roots(roots).await
            {
                tracing::warn!(%error, "rescanning Ollama after resolving Hub files failed");
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
            while !reconciliation
                .reconciliation_complete
                .load(Ordering::Acquire)
            {
                match reconciliation.ensure_installed_model_inventory().await {
                    Ok(()) => break,
                    Err(error) => {
                        tracing::warn!(
                            %error,
                            retry_delay_ms = BACKGROUND_RECONCILIATION_RETRY_DELAY.as_millis(),
                            "installed-model reconciliation failed; retrying"
                        );
                        tokio::time::sleep(BACKGROUND_RECONCILIATION_RETRY_DELAY).await;
                    }
                }
            }
        });
    }

    pub async fn ensure_model_inventory(&self) -> Result<(), InventoryError> {
        self.reconcile_model_inventory(false, &ScanScope::All).await
    }

    pub(crate) async fn ensure_installed_model_inventory(&self) -> Result<(), InventoryError> {
        self.reconcile_model_inventory(false, &ScanScope::All).await
    }

    /// Reconciles even when another reconciliation completed meanwhile, because the filesystem or
    /// the configuration changed after it started.
    pub(crate) async fn rescan_model_inventory(&self) -> Result<(), InventoryError> {
        self.reconcile_model_inventory(true, &ScanScope::All).await
    }

    /// Rescans only the named locations; every other location keeps its last scan.
    pub(crate) async fn rescan_model_roots(
        &self,
        roots: BTreeSet<PathBuf>,
    ) -> Result<(), InventoryError> {
        self.reconcile_model_inventory(true, &ScanScope::Roots(roots))
            .await
    }

    pub(crate) fn local_models_configuration(
        &self,
    ) -> Result<LocalModelsConfiguration, InventoryError> {
        self.local_models
            .read()
            .map(|configuration| configuration.clone())
            .map_err(|_| {
                InventoryError::Internal("local-models configuration lock poisoned".to_owned())
            })
    }

    /// Applies a whole configuration and rescans when it differs from the current one. Applying
    /// the same configuration again does nothing.
    pub(crate) async fn configure_local_models(
        &self,
        configuration: LocalModelsConfiguration,
    ) -> Result<bool, InventoryError> {
        validate_local_models(&configuration)?;
        let (changed, roots) = {
            let mut current = self.local_models.write().map_err(|_| {
                InventoryError::Internal("local-models configuration lock poisoned".to_owned())
            })?;
            let changed = *current != configuration;
            let roots = configuration
                .roots
                .iter()
                .filter(|root| !current.roots.contains(root))
                .map(|root| canonical_or_self(&root.path))
                .collect::<BTreeSet<_>>();
            *current = configuration;
            (changed, roots)
        };
        if changed {
            self.rescan_model_roots(roots).await?;
        }
        Ok(changed)
    }

    async fn reconcile_model_inventory(
        &self,
        force: bool,
        scope: &ScanScope,
    ) -> Result<(), InventoryError> {
        let observed_generation = self.ensure_generation.load(Ordering::Acquire);
        let _guard = self.ensure_gate.lock().await;
        if !force && self.ensure_generation.load(Ordering::Acquire) != observed_generation {
            return Ok(());
        }

        let live_models = self
            .models
            .read()
            .map_err(|_| InventoryError::Internal("inventory lock poisoned".to_owned()))?
            .clone();
        let mut attempt = 0_u8;
        let (discovered, next_evidence, installed_packages) =
            loop {
                let config = self.config.clone();
                let local = self.local_models_configuration()?;
                let cache = self.cache.clone();
                let scan_live_models = live_models.clone();
                let previous = self
                    .root_scans
                    .lock()
                    .map_err(|_| InventoryError::Internal("root scan lock poisoned".to_owned()))?
                    .clone();
                let scan_scope = if attempt == 0 {
                    scope.clone()
                } else {
                    ScanScope::All
                };
                let scan_result = tokio::task::spawn_blocking(move || {
                    scan(
                        &config,
                        &local,
                        &cache,
                        &scan_live_models,
                        &ScannedRoots {
                            scope: scan_scope,
                            previous,
                        },
                    )
                })
                .await
                .map_err(|error| InventoryError::Internal(error.to_string()))??;
                *self.root_scans.lock().map_err(|_| {
                    InventoryError::Internal("root scan lock poisoned".to_owned())
                })? = scan_result.root_scans.clone();
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
                        metadata_key: scan_result
                            .metadata_keys
                            .get(&model.id)
                            .cloned()
                            .ok_or_else(|| {
                                InventoryError::Internal(format!(
                                    "ready model {} has no metadata evidence",
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
        let installed_packages_changed = {
            let mut current = self.installed_packages.write().map_err(|_| {
                InventoryError::Internal("installed package snapshot lock poisoned".to_owned())
            })?;
            let changed = *current != installed_packages;
            *current = installed_packages;
            changed
        };
        self.ensure_generation.fetch_add(1, Ordering::AcqRel);
        self.request_ollama_hub_resolution();
        let became_authoritative = !self.reconciliation_complete.swap(true, Ordering::AcqRel);
        if !(installed_packages_changed || became_authoritative) {
            return Ok(());
        }
        let revision = self.packages_revision.fetch_add(1, Ordering::AcqRel) + 1;
        if let Some(observer) = self
            .installed_packages_observer
            .read()
            .ok()
            .and_then(|observer| observer.as_ref().and_then(Weak::upgrade))
        {
            observer.installed_packages_changed(revision);
        }
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
    ) -> Result<InventoryEntryId, InventoryError> {
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
        let id = inventory_entry_id("active-file", &canonical, &content);
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
                    metadata_key: model_metadata_evidence_for_model(&model)?,
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
        self.packages_revision.fetch_add(1, Ordering::Release);
        self.notify_installed_packages_changed();
        Ok(model)
    }

    pub(crate) async fn remove_published_model(
        &self,
        id: &InventoryEntryId,
    ) -> Result<(), InventoryError> {
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
        installed.records.remove(id);
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
        self.packages_revision.fetch_add(1, Ordering::Release);
        self.notify_installed_packages_changed();
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
    models: &BTreeMap<InventoryEntryId, InventoryModel>,
    observations: &BTreeMap<InventoryEntryId, String>,
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
        let Ok(id) = InventoryEntryId::parse(raw_id) else {
            continue;
        };
        if model.id != id {
            continue;
        }
        models.insert(id, model);
    }
    let mut evidence = BTreeMap::new();
    for (raw_id, entry) in raw_evidence {
        let Ok(id) = InventoryEntryId::parse(raw_id) else {
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
    models: &BTreeMap<InventoryEntryId, InventoryModel>,
    evidence: &BTreeMap<InventoryEntryId, CacheEvidence>,
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
        (ModelLocation::Directory { root, .. } | ModelLocation::External { root, .. }, _) => {
            root.join(&component.path)
        }
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
    validate_local_models(&config.local_models)
}

fn validate_local_models(configuration: &LocalModelsConfiguration) -> Result<(), InventoryError> {
    for root in &configuration.roots {
        if !root.path.is_absolute() {
            return Err(InventoryError::InvalidRequest(format!(
                "model location must be absolute: {}",
                root.path.display()
            )));
        }
    }
    for declaration in &configuration.declarations {
        if let CustomModelDeclarationOrigin::Imported { path } = &declaration.origin
            && (!path.is_absolute()
                || declaration.target.is_empty()
                || declaration
                    .target
                    .iter()
                    .chain(declaration.projector.iter())
                    .any(|file| !Path::new(file).is_absolute()))
        {
            return Err(InventoryError::InvalidRequest(format!(
                "imported model {} must declare absolute file paths",
                declaration.model_id
            )));
        }
        if declaration.target.is_empty() {
            return Err(InventoryError::InvalidRequest(format!(
                "model {} declares no target file",
                declaration.model_id
            )));
        }
        // A missing imported file only contributes nothing; a folder is never a model file.
        if let CustomModelDeclarationOrigin::Imported { path } = &declaration.origin
            && let Some(folder) = std::iter::once(path.as_path())
                .chain(declaration.target.iter().map(Path::new))
                .find(|file| file.is_dir())
        {
            return Err(InventoryError::InvalidRequest(format!(
                "imported model {} names the folder {}; an imported model is one GGUF file",
                declaration.model_id,
                folder.display()
            )));
        }
    }
    Ok(())
}

struct InventoryScan {
    models: BTreeMap<InventoryEntryId, InventoryModel>,
    observations: BTreeMap<InventoryEntryId, String>,
    metadata_keys: BTreeMap<InventoryEntryId, String>,
    root_scans: RootScans,
}

/// Which locations a reconciliation scans. `Roots` names canonical location paths.
#[derive(Debug, Clone)]
pub(crate) enum ScanScope {
    All,
    Roots(BTreeSet<PathBuf>),
}

/// The last scan of each location, by kind and canonical path, reused for locations outside a
/// reconciliation's scope.
type RootScans = BTreeMap<(ModelRootKind, PathBuf), Vec<DiscoveryCandidate>>;

struct ScannedRoots {
    scope: ScanScope,
    previous: RootScans,
}

impl ScannedRoots {
    /// The previous scan of a location outside the scope, while all its files are still there.
    fn reusable(
        &self,
        key: &(ModelRootKind, PathBuf),
        inventory_root: &Path,
    ) -> Option<&Vec<DiscoveryCandidate>> {
        let in_scope = match &self.scope {
            ScanScope::All => true,
            ScanScope::Roots(roots) => roots.contains(&key.1),
        };
        if in_scope {
            return None;
        }
        self.previous.get(key).filter(|candidates| {
            candidates
                .iter()
                .all(|DiscoveryCandidate::Artifact(candidate)| {
                    artifact_observation_key(inventory_root, &candidate.source, &candidate.location)
                        .is_ok()
                })
        })
    }
}

fn scan(
    config: &InventoryConfig,
    local: &LocalModelsConfiguration,
    cache: &ModelCache,
    live_models: &BTreeMap<InventoryEntryId, InventoryModel>,
    scanned: &ScannedRoots,
) -> Result<InventoryScan, InventoryError> {
    let declared = DeclaredProjectors::new(local);
    let mut discovered = Vec::new();
    scan_managed(config, &declared, cache, &mut discovered)?;

    // Roots scan in precedence order, so the first occurrence of a path is the preferred one. A
    // root that cannot be read is skipped for this reconciliation rather than hiding every other
    // location's models.
    let managed_hub = canonical_or_self(&config.root.join("hub"));
    let mut distinct = BTreeSet::new();
    let mut roots = local
        .roots
        .iter()
        .filter(|root| root.enabled)
        .cloned()
        .collect::<Vec<_>>();
    roots.sort_by_key(|root| root.kind);
    roots.retain(|root| {
        let canonical = canonical_or_self(&root.path);
        canonical != managed_hub && distinct.insert(canonical)
    });
    let mut root_scans = RootScans::new();
    let mut stale_roots = Vec::new();
    for root in &roots {
        let key = (root.kind, canonical_or_self(&root.path));
        match scanned.reusable(&key, &config.root) {
            Some(candidates) => {
                root_scans.insert(key, candidates.clone());
            }
            None => stale_roots.push(root.clone()),
        }
    }
    let concurrency = std::thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(1)
        .clamp(1, 8);
    for roots in stale_roots.chunks(concurrency) {
        let root_results = std::thread::scope(|scope| {
            roots
                .iter()
                .map(|root| {
                    let declared = &declared;
                    scope.spawn(move || {
                        let mut models = Vec::new();
                        crate::local_roots::scan_root(root, declared, cache, &mut models)
                            .map(|()| models)
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .zip(roots)
                .map(|(handle, root)| {
                    let result = handle.join().map_err(|_| {
                        InventoryError::Internal("model discovery worker panicked".to_owned())
                    })?;
                    Ok::<_, InventoryError>(result.unwrap_or_else(|error| {
                        tracing::warn!(
                            root = %root.path.display(),
                            kind = ?root.kind,
                            %error,
                            "model location could not be scanned"
                        );
                        Vec::new()
                    }))
                })
                .collect::<Result<Vec<_>, InventoryError>>()
        })?;
        for (root, models) in roots.iter().zip(root_results) {
            root_scans.insert((root.kind, canonical_or_self(&root.path)), models);
        }
    }
    for root in &roots {
        if let Some(candidates) = root_scans.get(&(root.kind, canonical_or_self(&root.path))) {
            discovered.extend(candidates.iter().cloned());
        }
    }
    crate::local_roots::scan_declared_files(local, &mut discovered);
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

    // Earlier external sources have higher precedence for the same canonical path. Catalog-derived
    // packages retain distinct identities when they intentionally share a primary weights file.
    let mut seen_paths = BTreeSet::new();
    let mut models = BTreeMap::new();
    let mut observations = BTreeMap::new();
    let mut metadata_keys = BTreeMap::new();
    let mut stale = Vec::new();
    for candidate in discovered {
        let path = candidate.primary_path().to_path_buf();
        let canonical = path.canonicalize().unwrap_or(path);
        let path_is_distinct = match &candidate {
            DiscoveryCandidate::Artifact(candidate) => {
                if matches!(candidate.location, ModelLocation::MagnitudeCache { .. }) {
                    // Catalog packages can intentionally share a primary weights file. They remain
                    // distinct by complete package content while claiming the path against later
                    // external-cache discovery.
                    seen_paths.insert(canonical);
                    true
                } else {
                    seen_paths.insert(canonical)
                }
            }
        };
        if path_is_distinct {
            match candidate {
                DiscoveryCandidate::Artifact(candidate) => {
                    let observation_key = artifact_observation_key(
                        &config.root,
                        &candidate.source,
                        &candidate.location,
                    )?;
                    let metadata_key = model_metadata_evidence(
                        &candidate.content_id,
                        candidate
                            .primary
                            .file_name()
                            .and_then(|value| value.to_str())
                            .unwrap_or("local model"),
                    )?;
                    observations.insert(candidate.id.clone(), observation_key.clone());
                    metadata_keys.insert(candidate.id.clone(), metadata_key.clone());
                    if let Some(model) = reuse_metadata(
                        &candidate,
                        &observation_key,
                        &metadata_key,
                        &cached_models,
                        &cached_evidence,
                    ) {
                        models.insert(model.id.clone(), model);
                    } else {
                        stale.push(candidate);
                    }
                }
            }
        }
    }

    // Enrichment is bounded across candidates, rather than being serialized by directory.
    for candidates in stale.chunks(concurrency) {
        let enriched = std::thread::scope(|scope| {
            candidates
                .iter()
                .map(|candidate| scope.spawn(move || enrich_candidate(candidate, cache)))
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
        metadata_keys,
        root_scans,
    })
}

fn scan_managed(
    config: &InventoryConfig,
    declared: &DeclaredProjectors,
    cache: &ModelCache,
    output: &mut Vec<DiscoveryCandidate>,
) -> Result<(), InventoryError> {
    let managed = config.root.join("hub");
    if !managed.is_dir() {
        return Ok(());
    }
    let mut count = 0;
    for repo_entry in read_dir_sorted(&managed)? {
        let repo_name = repo_entry.file_name().to_string_lossy().into_owned();
        let Some(repository) = parse_hf_repo_dir(&repo_name) else {
            continue;
        };
        let repository_root = repo_entry.path();
        let snapshots = read_dir_sorted(&repository_root.join("snapshots"))?;
        for snapshot_entry in snapshots {
            let snapshot_kind = snapshot_entry.file_type().map_err(io_error)?;
            if !snapshot_kind.is_dir() || snapshot_kind.is_symlink() {
                continue;
            }
            count += 1;
            if count > MAX_SCAN_ENTRIES {
                return Err(InventoryError::Io(
                    "managed model scan exceeded entry bound".to_owned(),
                ));
            }
            let snapshot = snapshot_entry.path();
            let commit = snapshot_entry.file_name().to_string_lossy().into_owned();
            append_discovered_groups(
                &repository,
                &repository_root,
                &snapshot,
                &commit,
                &declared.for_snapshot(&repository, &commit, &snapshot),
                cache,
                output,
            )?;
            for package in config
                .catalog_models
                .iter()
                .flat_map(catalog_packages)
                .map(|(package, _)| package)
            {
                let ModelPackageSource::HuggingFace {
                    repository: package_repository,
                    revision,
                } = &package.source
                else {
                    continue;
                };
                if package_repository != &repository {
                    continue;
                }
                let components = components_for_catalog_package(package)?;
                if !catalog_components_present(&snapshot, &repository_root, &components) {
                    continue;
                }
                let primary = match primary_path(&snapshot, &components) {
                    Some(path) => path,
                    None => continue,
                };
                let content = content_id(&components);
                let id = inventory_entry_id("magnitude-cache", &snapshot, &content);
                let timestamp = modified_seconds(&snapshot).unwrap_or_else(now);
                output.push(DiscoveryCandidate::Artifact(Box::new(ArtifactCandidate {
                    id,
                    content_id: content,
                    created: timestamp,
                    ready_at: timestamp,
                    source: ModelSource::HuggingFace {
                        repository: package_repository.clone(),
                        requested_revision: revision.clone(),
                        commit: commit.clone(),
                        metadata: None,
                    },
                    location: ModelLocation::MagnitudeCache {
                        total_bytes: components.iter().map(|item| item.size_bytes).sum(),
                        components,
                        integrity: Integrity::Verified {
                            method: "catalog_content_identity".to_owned(),
                        },
                    },
                    primary,
                    deletable: true,
                })));
            }
        }
    }
    Ok(())
}

pub(crate) fn components_for_catalog_package(
    package: &ModelPackage,
) -> Result<Vec<ModelComponent>, InventoryError> {
    let paths = package
        .files
        .iter()
        .map(|file| (file.id.clone(), file.path.clone()))
        .collect::<BTreeMap<_, _>>();
    package
        .files
        .iter()
        .map(|file| {
            let shard_index = package.relationships.iter().find_map(|relationship| {
                if let ModelFileRelationship::Shard { file_id, index, .. } = relationship
                    && file_id == &file.id
                {
                    Some(*index)
                } else {
                    None
                }
            });
            let relationship =
                package
                    .relationships
                    .iter()
                    .find_map(|relationship| match relationship {
                        ModelFileRelationship::ProjectorFor {
                            projector_file_id,
                            weights_file_id,
                        } if projector_file_id == &file.id => Some(
                            magnitude_service_contracts::ComponentRelationship::ProjectorFor {
                                projector: paths.get(projector_file_id)?.clone(),
                                model: paths.get(weights_file_id)?.clone(),
                            },
                        ),
                        ModelFileRelationship::MtpFor {
                            mtp_file_id,
                            weights_file_id,
                        } if mtp_file_id == &file.id => {
                            Some(magnitude_service_contracts::ComponentRelationship::MtpFor {
                                mtp: paths.get(mtp_file_id)?.clone(),
                                model: paths.get(weights_file_id)?.clone(),
                            })
                        }
                        ModelFileRelationship::DraftFor {
                            draft_file_id,
                            weights_file_id,
                            method,
                        } if draft_file_id == &file.id => Some(
                            magnitude_service_contracts::ComponentRelationship::DraftFor {
                                draft: paths.get(draft_file_id)?.clone(),
                                model: paths.get(weights_file_id)?.clone(),
                                method: method.clone(),
                            },
                        ),
                        _ => None,
                    });
            Ok(ModelComponent {
                path: file.path.clone(),
                role: match file.role {
                    ModelFileRole::Weights if shard_index.is_some() => ComponentRole::Shard,
                    ModelFileRole::Weights => ComponentRole::Weights,
                    ModelFileRole::Projector => ComponentRole::Projector,
                    ModelFileRole::Draft => ComponentRole::Draft,
                    ModelFileRole::Mtp => ComponentRole::Mtp,
                    ModelFileRole::Auxiliary => ComponentRole::Auxiliary,
                },
                size_bytes: file.size_bytes,
                content: ContentIdentity::Sha256 {
                    value: file.sha256.clone(),
                },
                shard_index,
                relationship,
            })
        })
        .collect()
}

pub(crate) fn catalog_components_present(
    snapshot: &Path,
    repository_root: &Path,
    components: &[ModelComponent],
) -> bool {
    let canonical_blobs = match repository_root.join("blobs").canonicalize() {
        Ok(root) => root,
        Err(_) => return false,
    };
    components.iter().all(|component| {
        let blob = repository_root
            .join("blobs")
            .join(blob_key(&component.content));
        let Ok(canonical_blob) = blob.canonicalize() else {
            return false;
        };
        if !canonical_blob.starts_with(&canonical_blobs)
            || !blob
                .metadata()
                .is_ok_and(|metadata| metadata.is_file() && metadata.len() == component.size_bytes)
        {
            return false;
        }
        let destination = snapshot.join(&component.path);
        destination.metadata().is_ok_and(|metadata| {
            metadata.is_file()
                && metadata.len() == component.size_bytes
                && same_file::is_same_file(&destination, &canonical_blob).unwrap_or(false)
        })
    })
}

// Every complete group is published, drafters included: a drafter downloaded on its own is a
// package a custom model's configuration resolves, and it survives rescans like any other.
fn append_discovered_groups(
    repository: &str,
    repository_root: &Path,
    snapshot: &Path,
    commit: &str,
    declared: &BTreeMap<PathBuf, PathBuf>,
    cache: &ModelCache,
    output: &mut Vec<DiscoveryCandidate>,
) -> Result<(), InventoryError> {
    let blobs = RepositoryBlobs::open(repository_root).map_err(io_error)?;
    for group in discover_groups(snapshot, repository_root, cache, declared)? {
        let components = components_for_group(snapshot, &group, Some(&blobs))?;
        let Some(primary) = primary_path(snapshot, &components) else {
            continue;
        };
        let content = content_id(&components);
        let id = inventory_entry_id("magnitude-cache", snapshot, &content);
        let timestamp = modified_seconds(snapshot).unwrap_or_else(now);
        output.push(DiscoveryCandidate::Artifact(Box::new(ArtifactCandidate {
            id,
            content_id: content,
            created: timestamp,
            ready_at: timestamp,
            source: ModelSource::HuggingFace {
                repository: repository.to_owned(),
                requested_revision: commit.to_owned(),
                commit: commit.to_owned(),
                metadata: None,
            },
            location: ModelLocation::MagnitudeCache {
                total_bytes: components.iter().map(|item| item.size_bytes).sum(),
                components,
                integrity: Integrity::Unverified {
                    reason: "filesystem_discovery".to_owned(),
                },
            },
            primary,
            deletable: true,
        })));
    }
    Ok(())
}

/// Snapshot entries may link anywhere inside the hub: current `huggingface_hub` keeps the bytes in
/// a blob store every repository shares.
pub(crate) fn scan_hf_cache(
    owner: HubCacheOwner,
    cache: &Path,
    declared: &DeclaredProjectors,
    inspections: &ModelCache,
    output: &mut Vec<DiscoveryCandidate>,
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
        let current_commit = std::fs::read_to_string(repo_root.join("refs/main"))
            .ok()
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty());
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
            let groups = discover_groups(
                &snapshot,
                cache,
                inspections,
                &declared.for_snapshot(&repository, &commit, &snapshot),
            )?;
            let blobs = RepositoryBlobs::open(&repo_root).map_err(io_error)?;
            for group in groups {
                let components = components_for_group(&snapshot, &group, Some(&blobs))?;
                let primary = match primary_path(&snapshot, &components) {
                    Some(path) => path,
                    None => continue,
                };
                let content = content_id(&components);
                let kind = match owner {
                    HubCacheOwner::HuggingFace => "hugging-face-cache",
                    HubCacheOwner::LlamaCpp => "llama-cpp",
                };
                let id = inventory_entry_id(kind, &snapshot, &content);
                let created = modified_seconds(&snapshot).unwrap_or_else(now);
                output.push(DiscoveryCandidate::Artifact(Box::new(ArtifactCandidate {
                    id,
                    content_id: content,
                    created,
                    ready_at: created,
                    source: ModelSource::HuggingFace {
                        repository: repository.clone(),
                        requested_revision: if current_commit.as_deref() == Some(commit.as_str()) {
                            "main".to_owned()
                        } else {
                            commit.clone()
                        },
                        commit: commit.clone(),
                        metadata: None,
                    },
                    location: ModelLocation::HuggingFaceCache {
                        owner,
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

#[derive(Debug, Clone)]
pub(crate) struct ArtifactCandidate {
    pub(crate) id: InventoryEntryId,
    pub(crate) content_id: magnitude_service_contracts::ContentId,
    pub(crate) created: u64,
    pub(crate) ready_at: u64,
    pub(crate) source: ModelSource,
    pub(crate) location: ModelLocation,
    pub(crate) primary: PathBuf,
    pub(crate) deletable: bool,
}

#[derive(Debug, Clone)]
pub(crate) enum DiscoveryCandidate {
    Artifact(Box<ArtifactCandidate>),
}

impl DiscoveryCandidate {
    fn primary_path(&self) -> &Path {
        match self {
            Self::Artifact(candidate) => &candidate.primary,
        }
    }
}

// Discovery resolves stable identity and location before enrichment. An unchanged artifact can
// therefore reuse its persisted terminal inspection without reopening or reprobeing the GGUF.
fn reuse_metadata(
    candidate: &ArtifactCandidate,
    observation_key: &str,
    metadata_key: &str,
    cached_models: &BTreeMap<InventoryEntryId, InventoryModel>,
    cached_evidence: &BTreeMap<InventoryEntryId, CacheEvidence>,
) -> Option<InventoryModel> {
    let reusable = cached_evidence
        .get(&candidate.id)
        .filter(|evidence| evidence.content_id == candidate.content_id.0)
        .filter(|evidence| evidence.observation_key == observation_key)
        .filter(|evidence| evidence.metadata_key == metadata_key)
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
    )?;
    Ok(model)
}

// This construction boundary intentionally lists every independently acquired inventory field;
// grouping them would introduce an otherwise meaningless intermediate domain type.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_model(
    id: InventoryEntryId,
    content_id: magnitude_service_contracts::ContentId,
    created: u64,
    ready_at: u64,
    source: ModelSource,
    location: ModelLocation,
    primary: &Path,
    deletable: bool,
    cache: &ModelCache,
) -> Result<InventoryModel, InventoryError> {
    let primary_name = primary
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("local model");
    let metadata_evidence = model_metadata_evidence(&content_id, primary_name)?;
    let cached_metadata = cache
        .read_index::<CachedModelMetadata>(ModelIndexKind::InventoryMetadata, &metadata_evidence);
    let inspected = match cached_metadata {
        Some(inspection) => inspection,
        None => match gguf::inspect_complete(primary) {
            Ok(inspection) => {
                let evidence = fingerprint(&inspection.fingerprint_material);
                let name = inspection.name.clone().unwrap_or_else(|| {
                    primary
                        .file_stem()
                        .and_then(|value| value.to_str())
                        .unwrap_or("local model")
                        .to_owned()
                });
                let inspected = CachedModelMetadata {
                    name,
                    properties: InventoryProperties::Inspected {
                        architecture: inspection.architecture,
                        quantization: inspection.quantization,
                        quantization_name: inspection.quantization_name,
                        parameter_count: inspection.parameter_count,
                        active_parameter_count: inspection.active_parameter_count,
                        training_context_length: inspection.training_context_length,
                        nextn_predict_layers: inspection.nextn_predict_layers,
                        tokenizer: inspection.tokenizer,
                        modalities: inspection.modalities,
                        base_models: inspection
                            .base_models
                            .into_iter()
                            .filter_map(|base| base.name.or(base.repo_url))
                            .collect(),
                        evidence_fingerprint: evidence,
                    },
                    supported_parameters: Vec::new(),
                };
                cache.write_index(
                    ModelIndexKind::InventoryMetadata,
                    &metadata_evidence,
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

fn model_metadata_evidence(
    content_id: &magnitude_service_contracts::ContentId,
    primary_name: &str,
) -> Result<String, InventoryError> {
    serde_json::to_string(&(&content_id.0, primary_name))
        .map_err(|error| InventoryError::Internal(error.to_string()))
}

fn model_metadata_evidence_for_model(model: &InventoryModel) -> Result<String, InventoryError> {
    let primary_name = model
        .location
        .components()
        .iter()
        .find(|component| {
            matches!(
                component.role,
                ComponentRole::Weights | ComponentRole::Shard
            )
        })
        .and_then(|component| Path::new(&component.path).file_name())
        .and_then(|value| value.to_str())
        .unwrap_or("local model");
    model_metadata_evidence(&model.content_id, primary_name)
}

#[allow(clippy::too_many_arguments)]
fn unavailable_model(
    id: InventoryEntryId,
    content_id: magnitude_service_contracts::ContentId,
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
pub(crate) struct ModelGroup {
    pub(crate) paths: Vec<PathBuf>,
    pub(crate) projector: Option<PathBuf>,
}

/// Projectors that declarations pair with their targets, by absolute path. A declaration names
/// its projector, so it wins over guessing from the directory.
pub(crate) struct DeclaredProjectors {
    added: BTreeMap<(String, String), Vec<(String, String)>>,
}

impl DeclaredProjectors {
    pub(crate) fn new(configuration: &LocalModelsConfiguration) -> Self {
        let mut added = BTreeMap::<(String, String), Vec<(String, String)>>::new();
        for declaration in &configuration.declarations {
            if let (
                CustomModelDeclarationOrigin::Added { repository, commit },
                Some(target),
                Some(projector),
            ) = (
                &declaration.origin,
                declaration.target.first(),
                declaration.projector.as_ref(),
            ) {
                added
                    .entry((repository.as_str().to_owned(), commit.clone()))
                    .or_default()
                    .push((target.clone(), projector.clone()));
            }
        }
        Self { added }
    }

    pub(crate) fn for_snapshot(
        &self,
        repository: &str,
        commit: &str,
        snapshot: &Path,
    ) -> BTreeMap<PathBuf, PathBuf> {
        self.added
            .get(&(repository.to_owned(), commit.to_owned()))
            .into_iter()
            .flatten()
            .map(|(target, projector)| (snapshot.join(target), snapshot.join(projector)))
            .collect()
    }
}

/// Header inspection of a file found while scanning, cached by its path, size and modification
/// time so an unchanged location is rescanned without rereading headers.
pub(crate) fn inspect_cached(cache: &ModelCache, path: &Path) -> Option<gguf::GgufInspection> {
    let metadata = path.metadata().ok()?;
    let evidence = format!("gguf-file-v3\0{}", file_identity(path, &metadata));
    if let Some(inspection) = cache.read_index(ModelIndexKind::GgufInspection, &evidence) {
        return Some(inspection);
    }
    let inspection = gguf::inspect_complete(path).ok()?;
    cache.write_index(ModelIndexKind::GgufInspection, &evidence, &inspection);
    Some(inspection)
}

/// Every complete GGUF group under `root`. Projectors are recognized from their headers and pair
/// with a model by declaration, else as the only projector in its directory. Drafters and
/// unreadable files form their own groups; what they are is decided from their headers later,
/// never from their names.
pub(crate) fn discover_groups(
    root: &Path,
    containment_root: &Path,
    cache: &ModelCache,
    declared: &BTreeMap<PathBuf, PathBuf>,
) -> Result<Vec<ModelGroup>, InventoryError> {
    let mut files = Vec::new();
    collect_gguf(root, &canonical_or_self(containment_root), 0, &mut files)?;
    let mut groups: BTreeMap<PathBuf, Vec<(u32, u32, PathBuf)>> = BTreeMap::new();
    let mut standalone = Vec::new();
    let mut projectors = Vec::new();
    for path in files {
        if let Some((prefix, index, total)) = split_shard_name(&path) {
            groups.entry(prefix).or_default().push((index, total, path));
            continue;
        }
        let role = inspect_cached(cache, &path).map(|inspection| inspection.role);
        match role {
            Some(gguf::GgufRole::Projector) => projectors.push(path),
            role => standalone.push((path, role)),
        }
    }
    let pairing = |path: &Path, role: Option<&gguf::GgufRole>| match role {
        Some(gguf::GgufRole::Model { .. }) => declared
            .get(path)
            .filter(|projector| projectors.contains(projector))
            .cloned()
            .or_else(|| unique_projector_for(path, &projectors)),
        _ => None,
    };
    let mut output = standalone
        .iter()
        .map(|(path, role)| ModelGroup {
            projector: pairing(path, role.as_ref()),
            paths: vec![path.clone()],
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
        let role = inspect_cached(cache, &first).map(|inspection| inspection.role);
        if role == Some(gguf::GgufRole::Projector) {
            continue;
        }
        output.push(ModelGroup {
            projector: pairing(&first, role.as_ref()),
            paths: shards.into_iter().map(|(_, _, path)| path).collect(),
        });
    }
    Ok(output)
}

pub(crate) fn components_for_group(
    root: &Path,
    group: &ModelGroup,
    blobs: Option<&RepositoryBlobs>,
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
            content: content_identity_for_file(path, &metadata, blobs),
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
            content: content_identity_for_file(projector, &metadata, blobs),
            shard_index: None,
            relationship: components.first().map(|model| {
                magnitude_service_contracts::ComponentRelationship::ProjectorFor {
                    projector: relative.to_path_buf(),
                    model: model.path.clone(),
                }
            }),
        });
    }
    Ok(components)
}

pub(crate) fn collect_gguf(
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
        if entry.file_name().to_string_lossy().starts_with('.')
            || crate::local_roots::is_partial_download(&path)
        {
            continue;
        }
        let file_type = entry.file_type().map_err(io_error)?;
        if file_type.is_symlink() {
            let canonical = match path.canonicalize() {
                Ok(canonical) if canonical.starts_with(containment_root) => canonical,
                _ => continue,
            };
            if canonical.is_file()
                && path
                    .extension()
                    .and_then(|value| value.to_str())
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("gguf"))
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
    if !path.extension()?.to_str()?.eq_ignore_ascii_case("gguf") {
        return None;
    }
    let stem = path.file_stem()?.to_str()?;
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

pub(crate) fn primary_path(root: &Path, components: &[ModelComponent]) -> Option<PathBuf> {
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

pub(crate) fn read_dir_sorted(path: &Path) -> Result<Vec<fs::DirEntry>, InventoryError> {
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

pub(crate) fn repository_lock_path(root: &Path, repository: &str) -> PathBuf {
    root.join("locks")
        .join(format!("{}.lock", hf_repo_dir(repository)))
}

pub(crate) fn modified_seconds(path: &Path) -> Option<u64> {
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
        (ModelLocation::Directory { root, .. } | ModelLocation::External { root, .. }, _) => {
            root.clone()
        }
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

fn content_identity_for_file(
    path: &Path,
    metadata: &fs::Metadata,
    blobs: Option<&RepositoryBlobs>,
) -> ContentIdentity {
    let blob = blobs.and_then(|blobs| blobs.resolve(path).ok().flatten());
    let in_blob_store = blob.is_some();
    let name = blob.as_ref().and_then(RepositoryBlob::name);
    if in_blob_store
        && let Some(value) = name.and_then(|value| value.strip_prefix("lfs-sha256-"))
        && value.len() == 64
        && value.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return ContentIdentity::Sha256 {
            value: value.to_ascii_lowercase(),
        };
    }
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

pub(crate) fn canonical_or_self(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

pub(crate) fn io_error(error: impl std::fmt::Display) -> InventoryError {
    InventoryError::Io(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    use magnitude_service_contracts::models::{
        CatalogBaseId, CatalogVariantId, InstalledCatalogAttribution, InstalledModelPackages,
        ModelFile, ModelFileId, ModelFileRole, ModelPackageInstallationOrigin,
        ModelPackageProperties, ModelReleaseDate, ModelServingConfiguration, PackageValidation,
        RecommendableModel, ServableModelBundle, ServingProfile,
    };

    fn catalog_model(package: ModelPackage) -> RecommendableModel {
        RecommendableModel {
            model_id: CatalogBaseId::new("catalog").expect("catalog base ID"),
            variant_id: CatalogVariantId::new("gguf:q4").expect("catalog variant ID"),
            configuration: ModelServingConfiguration {
                bundle: ServableModelBundle::Standalone { package },
                profile: ServingProfile {
                    context_length: 4_096,
                },
            },
            display_name: "Catalog model".to_owned(),
            variant_label: "Q4".to_owned(),
            description: String::new(),
            release_date: ModelReleaseDate::new("2026-01-01").expect("valid test date"),
            license: "test".to_owned(),
            parameterization: magnitude_service_contracts::models::ModelParameterization::Dense {
                total_parameters: 8_000_000_000,
            },
            intelligence: 0,
            support: magnitude_service_contracts::models::CatalogSupport::Supported,
            fidelity_rank: 0,
            quantization_aware: false,
        }
    }
    use magnitude_service_contracts::ModelInventory;

    #[test]
    fn configured_store_does_not_adopt_host_hugging_face_caches() {
        let root = std::env::temp_dir().join("icn-owned-model-store");
        let config =
            InventoryConfig::with_roots(root, std::env::temp_dir().join("icn-owned-model-cache"))
                .expect("absolute model and cache roots");

        assert!(config.local_models.roots.is_empty());
    }

    #[test]
    fn managed_inventory_derives_catalog_packages_from_files() {
        let temporary = tempfile::tempdir().expect("temporary root");
        let store = temporary
            .path()
            .canonicalize()
            .expect("canonical temp")
            .join("store");
        let cache = ModelCache::new(&temporary.path().join("cache"));
        let repository = "owner/model";
        let commit = "commit";
        let repository_root = store.join("hub").join(hf_repo_dir(repository));
        let snapshot = repository_root.join("snapshots").join(commit);
        let blobs = repository_root.join("blobs");
        fs::create_dir_all(&snapshot).expect("snapshot");
        fs::create_dir_all(&blobs).expect("blobs");

        let source = blobs.join("source.gguf");
        write_minimal_gguf(&source);
        let bytes = fs::read(&source).expect("model bytes");
        let digest = format!("{:x}", Sha256::digest(&bytes));
        let artifact_name = blob_key(&ContentIdentity::Sha256 {
            value: digest.clone(),
        });
        fs::rename(&source, blobs.join(&artifact_name)).expect("publish model blob");
        #[cfg(unix)]
        std::os::unix::fs::symlink(
            PathBuf::from("../../blobs").join(&artifact_name),
            snapshot.join(&artifact_name),
        )
        .expect("snapshot link");
        #[cfg(not(unix))]
        fs::hard_link(blobs.join(&artifact_name), snapshot.join(&artifact_name))
            .expect("snapshot link");

        let model_file = ModelFile {
            id: ModelFileId(format!("file_{digest}")),
            path: PathBuf::from(&artifact_name),
            role: ModelFileRole::Weights,
            size_bytes: u64::try_from(bytes.len()).expect("fixture size"),
            tensor_storage_bytes: None,
            sha256: digest,
        };
        let mut config =
            InventoryConfig::with_roots(store, temporary.path().join("cache")).expect("config");
        config.catalog_models = vec![catalog_model(ModelPackage {
            id: ModelPackageId("package_catalog".to_owned()),
            source: ModelPackageSource::HuggingFace {
                repository: repository.to_owned(),
                revision: commit.to_owned(),
            },
            files: vec![model_file],
            relationships: Vec::new(),
            properties: ModelPackageProperties {
                format: "gguf".to_owned(),
                quantization: "unknown".to_owned(),
                quantization_name: "unknown".to_owned(),
                architecture: "unknown".to_owned(),
                maximum_context_length: Some(4_096),
                intrinsic_model_id: None,
                intrinsic_quality_id: None,
            },
        })];

        let result = scan(
            &config,
            &LocalModelsConfiguration::default(),
            &cache,
            &BTreeMap::new(),
            &ScannedRoots {
                scope: ScanScope::All,
                previous: RootScans::new(),
            },
        )
        .expect("filesystem-derived inventory");

        assert_eq!(result.models.len(), 1);
        let discovered = result.models.values().next().expect("catalog model");
        assert!(matches!(
            discovered.location,
            ModelLocation::MagnitudeCache { .. }
        ));
        assert!(matches!(
            discovered.availability,
            ModelAvailability::Available { .. }
        ));
        assert!(
            same_file::is_same_file(snapshot.join(&artifact_name), blobs.join(&artifact_name),)
                .expect("snapshot refers to the verified blob")
        );

        fs::remove_file(snapshot.join(&artifact_name)).expect("remove installed link");
        let after_delete = scan(
            &config,
            &LocalModelsConfiguration::default(),
            &cache,
            &BTreeMap::new(),
            &ScannedRoots {
                scope: ScanScope::All,
                previous: RootScans::new(),
            },
        )
        .expect("read-only filesystem-derived inventory");
        assert!(after_delete.models.is_empty());
        assert!(!snapshot.join(&artifact_name).exists());

        fs::copy(blobs.join(&artifact_name), snapshot.join(&artifact_name))
            .expect("unlinked copy with identical bytes");
        let after_copy = scan(
            &config,
            &LocalModelsConfiguration::default(),
            &cache,
            &BTreeMap::new(),
            &ScannedRoots {
                scope: ScanScope::All,
                previous: RootScans::new(),
            },
        )
        .expect("scan unlinked copy");
        assert!(after_copy.models.is_empty());
    }

    #[cfg(unix)]
    #[test]
    fn managed_inventory_scans_each_complete_snapshot_independently() {
        let temporary = tempfile::tempdir().expect("temporary root");
        let store = temporary
            .path()
            .canonicalize()
            .expect("canonical temp")
            .join("store");
        let cache = ModelCache::new(&temporary.path().join("cache"));
        let repository = "owner/model";
        let repository_root = store.join("hub").join(hf_repo_dir(repository));
        let snapshot = repository_root.join("snapshots/commit");
        let blobs = repository_root.join("blobs");
        fs::create_dir_all(&snapshot).expect("snapshot");
        fs::create_dir_all(&blobs).expect("blobs");

        let mut catalog_bytes = Vec::new();
        catalog_bytes.extend_from_slice(b"GGUF");
        catalog_bytes.extend_from_slice(&3_u32.to_le_bytes());
        catalog_bytes.extend_from_slice(&0_u64.to_le_bytes());
        catalog_bytes.extend_from_slice(&0_u64.to_le_bytes());
        catalog_bytes.resize(32, 0);
        let mut other_bytes = catalog_bytes.clone();
        other_bytes.extend_from_slice(&[1, 2, 3, 4]);
        let catalog_digest = format!("{:x}", Sha256::digest(&catalog_bytes));
        let other_digest = format!("{:x}", Sha256::digest(&other_bytes));
        let catalog_blob = blob_key(&ContentIdentity::Sha256 {
            value: catalog_digest.clone(),
        });
        let other_blob = blob_key(&ContentIdentity::Sha256 {
            value: other_digest,
        });
        fs::write(blobs.join(&catalog_blob), catalog_bytes).expect("catalog blob");
        fs::write(blobs.join(&other_blob), other_bytes).expect("other blob");
        std::os::unix::fs::symlink(
            PathBuf::from("../../blobs").join(&catalog_blob),
            snapshot.join("catalog.gguf"),
        )
        .expect("catalog snapshot link");
        std::os::unix::fs::symlink(
            PathBuf::from("../../blobs").join(&other_blob),
            snapshot.join("other.gguf"),
        )
        .expect("other snapshot link");

        let mut config =
            InventoryConfig::with_roots(store, temporary.path().join("cache")).expect("config");
        config.catalog_models = vec![catalog_model(ModelPackage {
            id: ModelPackageId("package_catalog".to_owned()),
            source: ModelPackageSource::HuggingFace {
                repository: repository.to_owned(),
                revision: "commit".to_owned(),
            },
            files: vec![ModelFile {
                id: ModelFileId(format!("file_{catalog_digest}")),
                path: PathBuf::from("catalog.gguf"),
                role: ModelFileRole::Weights,
                size_bytes: 32,
                tensor_storage_bytes: None,
                sha256: catalog_digest,
            }],
            relationships: Vec::new(),
            properties: ModelPackageProperties {
                format: "gguf".to_owned(),
                quantization: "unknown".to_owned(),
                quantization_name: "unknown".to_owned(),
                architecture: "unknown".to_owned(),
                maximum_context_length: Some(4_096),
                intrinsic_model_id: None,
                intrinsic_quality_id: None,
            },
        })];

        let discovered = scan(
            &config,
            &LocalModelsConfiguration::default(),
            &cache,
            &BTreeMap::new(),
            &ScannedRoots {
                scope: ScanScope::All,
                previous: RootScans::new(),
            },
        )
        .expect("single managed revision");
        assert_eq!(discovered.models.len(), 2);

        fs::create_dir_all(repository_root.join("snapshots/other-commit"))
            .expect("second snapshot");
        let with_incomplete_second_snapshot = scan(
            &config,
            &LocalModelsConfiguration::default(),
            &cache,
            &BTreeMap::new(),
            &ScannedRoots {
                scope: ScanScope::All,
                previous: RootScans::new(),
            },
        )
        .expect("managed snapshots");
        assert_eq!(with_incomplete_second_snapshot.models.len(), 2);
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

    /// Windows publishes snapshot entries as hard links, and the download assigns the entry id
    /// before the snapshot exists. Removal must find that id and reclaim the blob.
    #[tokio::test]
    async fn hard_linked_catalog_package_is_listed_once_and_removed_with_its_blob() {
        use magnitude_service_contracts::models::InstalledModelPackages as _;

        let temporary = tempfile::tempdir().unwrap();
        let store = temporary.path().join("store");
        let repository = "owner/model";
        let commit = "commit";
        let repository_root = store.join("hub").join(hf_repo_dir(repository));
        let snapshot = repository_root.join("snapshots").join(commit);
        let blobs = repository_root.join("blobs");
        fs::create_dir_all(&blobs).unwrap();
        let source = temporary.path().join("source.gguf");
        write_minimal_gguf(&source);
        let bytes = fs::read(&source).unwrap();
        let digest = format!("{:x}", Sha256::digest(&bytes));
        let blob = blobs.join(blob_key(&ContentIdentity::Sha256 {
            value: digest.clone(),
        }));
        let package = ModelPackage {
            id: ModelPackageId("package_catalog".to_owned()),
            source: ModelPackageSource::HuggingFace {
                repository: repository.to_owned(),
                revision: commit.to_owned(),
            },
            files: vec![ModelFile {
                id: ModelFileId(format!("file_{digest}")),
                path: PathBuf::from("model.gguf"),
                role: ModelFileRole::Weights,
                size_bytes: u64::try_from(bytes.len()).unwrap(),
                tensor_storage_bytes: None,
                sha256: digest,
            }],
            relationships: Vec::new(),
            properties: ModelPackageProperties {
                format: "gguf".to_owned(),
                quantization: "unknown".to_owned(),
                quantization_name: "unknown".to_owned(),
                architecture: "unknown".to_owned(),
                maximum_context_length: Some(4_096),
                intrinsic_model_id: None,
                intrinsic_quality_id: None,
            },
        };
        let download_time_id = inventory_entry_id(
            "magnitude-cache",
            &snapshot,
            &content_id(&components_for_catalog_package(&package).unwrap()),
        );
        fs::rename(&source, &blob).unwrap();
        fs::create_dir_all(&snapshot).unwrap();
        fs::hard_link(&blob, snapshot.join("model.gguf")).unwrap();

        let mut config =
            InventoryConfig::with_roots(store, temporary.path().join("cache")).unwrap();
        config.catalog_models = vec![catalog_model(package.clone())];
        let manager = ManagedModelStore::open(config).await.unwrap();
        manager.ensure_installed_model_inventory().await.unwrap();
        let models = manager.list().await.unwrap();
        assert_eq!(
            models.len(),
            1,
            "a hard-linked snapshot must not add a discovered duplicate"
        );
        assert_eq!(models[0].id, download_time_id);

        let installed = manager.list_installed().await.unwrap();
        assert_eq!(installed.packages.len(), 1);
        assert_eq!(
            installed.packages[0].origin,
            ModelPackageInstallationOrigin::Magnitude
        );
        let removed = manager
            .remove_installed(&installed.packages[0].package.id)
            .await
            .unwrap();
        assert!(removed.removed);
        assert_eq!(removed.freed_bytes, u64::try_from(bytes.len()).unwrap());
        assert!(!blob.exists());
        assert!(!snapshot.join("model.gguf").exists());
        assert!(
            !repository_root.exists(),
            "a repository folder left without a model is removed from Magnitude's store"
        );
        assert!(manager.list().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn installed_package_listing_reports_discovered_package() {
        let temporary = tempfile::tempdir().unwrap();
        let store = temporary.path().join("store");
        let (hf_cache, source) = create_hf_snapshot(temporary.path());
        write_minimal_gguf(&source.join("model.gguf"));

        let mut config =
            InventoryConfig::with_roots(store, temporary.path().join("cache")).unwrap();
        config.add_hugging_face_caches([hf_cache]);
        let manager = ManagedModelStore::open(config).await.unwrap();
        manager.ensure_installed_model_inventory().await.unwrap();
        let installed = manager.list_installed().await.unwrap();

        assert_eq!(installed.packages.len(), 1);
        assert_eq!(
            installed.packages[0].origin,
            magnitude_service_contracts::models::ModelPackageInstallationOrigin::HuggingFaceCache
        );
        let assessed = manager.list().await.unwrap();
        assert_eq!(assessed.len(), 1);
    }

    #[tokio::test]
    async fn external_exact_catalog_package_is_attributed_without_changing_ownership() {
        let temporary = tempfile::tempdir().unwrap();
        let store = temporary.path().join("store");
        let cache = temporary.path().join("cache");
        let (hf_cache, source) = create_hf_snapshot(temporary.path());
        write_minimal_gguf(&source.join("model.gguf"));

        let mut initial = InventoryConfig::with_roots(store.clone(), cache.clone()).unwrap();
        initial.add_hugging_face_caches([hf_cache.clone()]);
        let manager = ManagedModelStore::open(initial).await.unwrap();
        manager.ensure_installed_model_inventory().await.unwrap();
        let package = manager.list_installed().await.unwrap().packages[0]
            .package
            .clone();
        drop(manager);

        let mut configured = InventoryConfig::with_roots(store.clone(), cache.clone()).unwrap();
        configured.add_hugging_face_caches([hf_cache.clone()]);
        configured.catalog_models = vec![catalog_model(package.clone())];
        let manager = ManagedModelStore::open(configured).await.unwrap();
        manager.ensure_installed_model_inventory().await.unwrap();
        let installed = manager.list_installed().await.unwrap();
        assert_eq!(
            installed.packages[0].origin,
            ModelPackageInstallationOrigin::HuggingFaceCache
        );
        assert!(matches!(
            installed.packages[0].catalog_attribution,
            InstalledCatalogAttribution::Attributed { .. }
        ));
        drop(manager);

        let mut future_package = package;
        future_package.id = ModelPackageId("future-package".to_owned());
        let mut updated = InventoryConfig::with_roots(store, cache).unwrap();
        updated.add_hugging_face_caches([hf_cache]);
        updated.catalog_models = vec![catalog_model(future_package)];
        let manager = ManagedModelStore::open(updated).await.unwrap();
        manager.ensure_installed_model_inventory().await.unwrap();
        let installed = manager.list_installed().await.unwrap();
        assert!(matches!(
            installed.packages[0].catalog_attribution,
            InstalledCatalogAttribution::Attributed { .. }
        ));
    }

    /// The catalog's exact target found in a Hugging Face cache, the same repository, commit and
    /// weight file, is the catalog model even when the catalog's package also holds a projector
    /// the cache lacks. Another commit of the same file is not.
    #[tokio::test]
    async fn external_exact_catalog_target_weights_are_attributed_without_its_projector() {
        let temporary = tempfile::tempdir().unwrap();
        let store = temporary.path().join("store");
        let cache = temporary.path().join("cache");
        let (hf_cache, source) = create_hf_snapshot(temporary.path());
        write_minimal_gguf(&source.join("model.gguf"));

        let mut initial = InventoryConfig::with_roots(store.clone(), cache.clone()).unwrap();
        initial.add_hugging_face_caches([hf_cache.clone()]);
        let manager = ManagedModelStore::open(initial).await.unwrap();
        manager.ensure_installed_model_inventory().await.unwrap();
        let found = manager.list_installed().await.unwrap().packages[0]
            .package
            .clone();
        drop(manager);

        let weights = found.files[0].clone();
        let projector = ModelFile {
            id: ModelFileId(format!("file_{}", "b".repeat(64))),
            path: PathBuf::from("mmproj-BF16.gguf"),
            role: ModelFileRole::Projector,
            size_bytes: 1_000,
            tensor_storage_bytes: None,
            sha256: "b".repeat(64),
        };
        let mut target = found.clone();
        target.id = ModelPackageId("package_catalog_target".to_owned());
        target.files = vec![weights.clone(), projector.clone()];
        target.relationships = vec![
            magnitude_service_contracts::models::ModelFileRelationship::ProjectorFor {
                projector_file_id: projector.id.clone(),
                weights_file_id: weights.id.clone(),
            },
        ];
        let attribution = |target: ModelPackage, hidden: Vec<magnitude_service_contracts::models::ModelId>| {
            let (store, cache, hf_cache) = (store.clone(), cache.clone(), hf_cache.clone());
            async move {
                let mut configured = InventoryConfig::with_roots(store, cache).unwrap();
                configured.add_hugging_face_caches([hf_cache]);
                configured.local_models.hidden = hidden;
                configured.catalog_models = vec![catalog_model(target)];
                let manager = ManagedModelStore::open(configured).await.unwrap();
                manager.ensure_installed_model_inventory().await.unwrap();
                manager.list_installed().await.unwrap().packages[0]
                    .catalog_attribution
                    .clone()
            }
        };

        assert!(matches!(
            attribution(target.clone(), Vec::new()).await,
            InstalledCatalogAttribution::Attributed { .. }
        ));
        let found_id = crate::package_service::hub_model_id(&found).unwrap();
        assert_eq!(found_id.to_string(), "hf:test/model/model.gguf");

        // Requests and references made before the upgrade, naming the found model's `hf:` ID, are
        // the catalog model's.
        let mut configured = InventoryConfig::with_roots(store.clone(), cache.clone()).unwrap();
        configured.add_hugging_face_caches([hf_cache.clone()]);
        let definition = catalog_model(target.clone());
        configured.catalog_models = vec![definition.clone()];
        let manager = Arc::new(ManagedModelStore::open(configured).await.unwrap());
        manager.ensure_installed_model_inventory().await.unwrap();
        let resolver = crate::model_domains::ModelDomainResolver::new(
            manager,
            magnitude_service_contracts::models::RecommendableModelCatalog {
                models: vec![definition.clone()],
                diagnostics: Vec::new(),
            },
        );
        let catalog_id = magnitude_service_contracts::models::ModelId::catalog(
            &definition.model_id,
            &definition.variant_id,
        );
        assert_eq!(resolver.canonical_model_id(found_id.clone()).unwrap(), catalog_id);
        assert_eq!(
            resolver.canonical_model_id(catalog_id.clone()).unwrap(),
            catalog_id
        );
        assert_eq!(
            resolver.catalog_snapshot().unwrap().models[0].aliases,
            vec![found_id.clone()]
        );
        drop(resolver);

        // Hidden, the copy is a hidden found model, served under its own ID, and the catalog still
        // lists the alias so the catalog ID can unhide it.
        let mut hidden = InventoryConfig::with_roots(store.clone(), cache.clone()).unwrap();
        hidden.add_hugging_face_caches([hf_cache.clone()]);
        hidden.local_models.hidden = vec![found_id.clone()];
        hidden.catalog_models = vec![definition.clone()];
        let manager = Arc::new(ManagedModelStore::open(hidden).await.unwrap());
        manager.ensure_installed_model_inventory().await.unwrap();
        let resolver = crate::model_domains::ModelDomainResolver::new(
            manager,
            magnitude_service_contracts::models::RecommendableModelCatalog {
                models: vec![definition.clone()],
                diagnostics: Vec::new(),
            },
        );
        assert_eq!(resolver.canonical_model_id(found_id.clone()).unwrap(), found_id);
        assert_eq!(
            resolver.catalog_snapshot().unwrap().models[0].aliases,
            vec![found_id.clone()]
        );
        drop(resolver);
        assert_eq!(
            attribution(target.clone(), vec![found_id]).await,
            InstalledCatalogAttribution::NotCatalogTarget,
            "a hidden found copy of the catalog target stays a hidden found model"
        );
        let mut other_commit = target;
        other_commit.source = ModelPackageSource::HuggingFace {
            repository: "test/model".to_owned(),
            revision: "f".repeat(40),
        };
        assert_eq!(
            attribution(other_commit, Vec::new()).await,
            InstalledCatalogAttribution::NotCatalogTarget
        );
    }

    /// `huggingface_hub` without symlink support (Windows without Developer Mode, or
    /// `HF_HUB_DISABLE_SYMLINKS`) keeps a plain copy in the snapshot and no blob, so no blob name
    /// gives the file's Hub SHA-256. A copy at a catalog file's repository, commit and path, with its
    /// size and header digest, is that file: the catalog's exact target, not a second custom model.
    #[tokio::test]
    async fn plain_copy_layout_catalog_target_is_identified_by_its_hub_hash() {
        let temporary = tempfile::tempdir().unwrap();
        let store = temporary.path().join("store");
        let cache = temporary.path().join("cache");
        let hf_cache = temporary.path().join("hub");
        let commit = "3d55fac80935ae6456986ad2384b5cbcc4d6c948";
        let repository_root = hf_cache.join("models--Owner--Model-GGUF");
        let snapshot = repository_root.join("snapshots").join(commit);
        fs::create_dir_all(&snapshot).unwrap();
        fs::create_dir_all(repository_root.join("refs")).unwrap();
        fs::write(repository_root.join("refs/main"), commit).unwrap();
        let file = snapshot.join("Model-Q4_K_M.gguf");
        write_minimal_gguf_with_string_metadata(&file, &[("general.name", "Model")]);
        let bytes = fs::read(&file).unwrap();
        let sha256 = format!("{:x}", Sha256::digest(&bytes));
        let header_digest = crate::gguf::inspect_complete(&file).unwrap().header_digest;

        let weights = ModelFile {
            id: ModelFileId(format!("file_{sha256}")),
            path: PathBuf::from("Model-Q4_K_M.gguf"),
            role: ModelFileRole::Weights,
            size_bytes: bytes.len() as u64,
            tensor_storage_bytes: None,
            sha256: sha256.clone(),
        };
        let target = ModelPackage {
            id: crate::package_service::canonical_package_id(std::slice::from_ref(&weights), &[]),
            source: ModelPackageSource::HuggingFace {
                repository: "owner/Model-GGUF".to_owned(),
                revision: commit.to_owned(),
            },
            files: vec![weights],
            relationships: Vec::new(),
            properties: ModelPackageProperties {
                format: "gguf".to_owned(),
                quantization: "unknown".to_owned(),
                quantization_name: "unknown".to_owned(),
                architecture: "unknown".to_owned(),
                maximum_context_length: None,
                intrinsic_model_id: None,
                intrinsic_quality_id: None,
            },
        };
        let installed = |header_digests: BTreeMap<String, String>| {
            let (store, cache, hf_cache, target) =
                (store.clone(), cache.clone(), hf_cache.clone(), target.clone());
            async move {
                let mut configured = InventoryConfig::with_roots(store, cache).unwrap();
                configured.add_hugging_face_caches([hf_cache]);
                configured.catalog_models = vec![catalog_model(target)];
                configured.catalog_header_digests = header_digests;
                let manager = ManagedModelStore::open(configured).await.unwrap();
                manager.ensure_installed_model_inventory().await.unwrap();
                manager.list_installed().await.unwrap().packages
            }
        };

        let packages = installed(BTreeMap::from([(sha256.clone(), header_digest)])).await;
        assert_eq!(packages.len(), 1);
        assert_eq!(packages[0].package.files[0].sha256, sha256);
        assert_eq!(packages[0].package.id, target.id);
        assert!(matches!(
            packages[0].catalog_attribution,
            InstalledCatalogAttribution::Attributed { .. }
        ));

        // A copy whose header differs from the catalog file's is not that file.
        let packages = installed(BTreeMap::from([(sha256.clone(), "0".repeat(64))])).await;
        assert_eq!(packages.len(), 1);
        assert_ne!(packages[0].package.files[0].sha256, sha256);
        assert_eq!(
            packages[0].catalog_attribution,
            InstalledCatalogAttribution::NotCatalogTarget
        );
    }

    #[tokio::test]
    async fn completed_publication_updates_installed_snapshot_before_returning() {
        let temporary = tempfile::tempdir().unwrap();
        let model_path = temporary.path().join("active.gguf");
        write_minimal_gguf(&model_path);
        let manager = ManagedModelStore::open(
            InventoryConfig::with_roots(
                temporary.path().join("store"),
                temporary.path().join("cache"),
            )
            .unwrap(),
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
        config.add_hugging_face_caches([hf_cache]);
        let manager = ManagedModelStore::open(config).await.unwrap();
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
        let manager = ManagedModelStore::open(
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
    async fn structural_failure_isolated_to_the_affected_installed_model() {
        let temporary = tempfile::tempdir().unwrap();
        let store = temporary.path().join("store");
        let (hf_cache, source) = create_hf_snapshot(temporary.path());
        write_minimal_gguf(&source.join("working.gguf"));
        fs::write(source.join("broken.gguf"), b"not a gguf").unwrap();

        let mut config =
            InventoryConfig::with_roots(store, temporary.path().join("cache")).unwrap();
        config.add_hugging_face_caches([hf_cache]);
        let manager = ManagedModelStore::open(config).await.unwrap();

        manager.ensure_installed_model_inventory().await.unwrap();
        let installed = manager.list_installed().await.unwrap();

        assert_eq!(installed.packages.len(), 2);
        assert_eq!(
            installed
                .packages
                .iter()
                .filter(|package| matches!(package.validation, PackageValidation::Valid))
                .count(),
            1,
        );
        assert_eq!(
            installed
                .packages
                .iter()
                .filter(|package| {
                    matches!(package.validation, PackageValidation::Invalid { .. })
                })
                .count(),
            1,
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn managed_download_publication_only_requires_structural_inspection() {
        let temporary = tempfile::tempdir().unwrap();
        let store = temporary.path().join("store");
        let manager = ManagedModelStore::open(
            InventoryConfig::with_roots(store.clone(), temporary.path().join("cache")).unwrap(),
        )
        .await
        .unwrap();
        manager.ensure_installed_model_inventory().await.unwrap();

        let repository = "owner/repository";
        let commit = "a".repeat(40);
        let repository_root = store.join("hub").join(hf_repo_dir(repository));
        let snapshot = repository_root.join("snapshots").join(&commit);
        let blobs = repository_root.join("blobs");
        fs::create_dir_all(&snapshot).unwrap();
        fs::create_dir_all(&blobs).unwrap();
        let fixture = temporary.path().join("fixture.gguf");
        write_minimal_gguf(&fixture);
        let bytes = fs::read(&fixture).unwrap();
        let sha256 = format!("{:x}", Sha256::digest(&bytes));
        let content_identity = ContentIdentity::Sha256 {
            value: sha256.clone(),
        };
        let blob = blobs.join(blob_key(&content_identity));
        fs::write(&blob, &bytes).unwrap();
        let primary = snapshot.join("model.gguf");
        std::os::unix::fs::symlink(
            PathBuf::from("../../blobs").join(blob.file_name().unwrap()),
            &primary,
        )
        .unwrap();
        let components = vec![ModelComponent {
            path: PathBuf::from("model.gguf"),
            role: ComponentRole::Weights,
            size_bytes: u64::try_from(bytes.len()).unwrap(),
            content: ContentIdentity::Sha256 { value: sha256 },
            shard_index: None,
            relationship: None,
        }];
        let content = content_id(&components);
        let model_id = inventory_entry_id("magnitude-cache", &snapshot, &content);
        let ready_at = now();
        let model = build_model(
            model_id,
            content,
            ready_at,
            ready_at,
            ModelSource::HuggingFace {
                repository: repository.to_owned(),
                requested_revision: commit.clone(),
                commit,
                metadata: None,
            },
            ModelLocation::MagnitudeCache {
                total_bytes: components
                    .iter()
                    .map(|component| component.size_bytes)
                    .sum(),
                components,
                integrity: Integrity::Verified {
                    method: "content_identity".to_owned(),
                },
            },
            &primary,
            true,
            &manager.cache,
        )
        .unwrap();
        let published = manager.complete_and_publish_model(model).await.unwrap();

        assert!(matches!(
            published.availability,
            ModelAvailability::Available { .. }
        ));
        assert!(matches!(
            published.properties,
            InventoryProperties::Inspected { .. }
        ));
    }

    #[test]
    fn recognizes_only_complete_split_names() {
        let first = Path::new("model-00001-of-00002.gguf");
        assert_eq!(
            split_shard_name(first).map(|(_, index, total)| (index, total)),
            Some((1, 2))
        );
        assert_eq!(
            split_shard_name(Path::new("model-00002-of-00002.GGUF"))
                .map(|(_, index, total)| (index, total)),
            Some((2, 2))
        );
        assert!(split_shard_name(Path::new("model-1-of-2.gguf")).is_none());
    }

    #[cfg(unix)]
    #[test]
    fn discovers_case_insensitive_gguf_snapshot_links() {
        let temporary = tempfile::tempdir().unwrap();
        let blob = temporary.path().join("blob");
        write_minimal_gguf(&blob);
        let snapshot = temporary.path().join("snapshot");
        fs::create_dir(&snapshot).unwrap();
        std::os::unix::fs::symlink(&blob, snapshot.join("MODEL.GGUF")).unwrap();

        let groups = discover_groups(
            &snapshot,
            &temporary
                .path()
                .canonicalize()
                .expect("canonical temporary root"),
            &ModelCache::new(&temporary.path().join("cache")),
            &BTreeMap::new(),
        )
        .unwrap();

        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].paths[0], snapshot.join("MODEL.GGUF"));
    }

    /// A file a download tool is still writing (llama.cpp and browsers name it `*-partial.gguf`)
    /// is never a model: the watcher ignores its removal, so discovering it would leave it stale.
    #[test]
    fn partial_downloads_named_gguf_are_not_discovered() {
        let temporary = tempfile::tempdir().unwrap();
        let folder = temporary.path().join("models");
        fs::create_dir(&folder).unwrap();
        write_minimal_gguf(&folder.join("model.gguf"));
        write_minimal_gguf(&folder.join("other.gguf-partial.gguf"));
        write_minimal_gguf(&folder.join("third-partial.gguf"));

        let groups = discover_groups(
            &folder,
            &folder,
            &ModelCache::new(&temporary.path().join("cache")),
            &BTreeMap::new(),
        )
        .unwrap();

        assert_eq!(
            groups
                .iter()
                .map(|group| group.paths.clone())
                .collect::<Vec<_>>(),
            vec![vec![folder.join("model.gguf")]]
        );
    }

    #[test]
    fn groups_files_by_header_role_not_by_name() {
        use crate::test_gguf::{
            write_dflash_drafter, write_model, write_mtp_model, write_projector,
        };
        let temporary = tempfile::tempdir().unwrap();
        let cache = ModelCache::new(&temporary.path().join("cache"));
        let models = temporary.path().join("models");
        write_model(&models.join("laguna-s-2.1-Q4_K_M.gguf"), "laguna");
        write_dflash_drafter(&models.join("unlabelled-companion.gguf"), "drafter");
        write_mtp_model(&models.join("Qwen3.6-MTP-Q4_K_M.gguf"), "qwen mtp");
        write_projector(&models.join("vision.gguf"), "projector");
        fs::write(models.join("partial.gguf.part"), b"partial").unwrap();
        fs::write(models.join(".hidden.gguf"), b"resource fork").unwrap();

        let groups = discover_groups(&models, &models, &cache, &BTreeMap::new()).unwrap();

        let names = groups
            .iter()
            .map(|group| {
                (
                    group.paths[0]
                        .file_name()
                        .unwrap()
                        .to_str()
                        .unwrap()
                        .to_owned(),
                    group.projector.as_ref().map(|projector| {
                        projector.file_name().unwrap().to_str().unwrap().to_owned()
                    }),
                )
            })
            .collect::<BTreeMap<_, _>>();
        assert_eq!(
            names,
            BTreeMap::from([
                (
                    "Qwen3.6-MTP-Q4_K_M.gguf".to_owned(),
                    Some("vision.gguf".to_owned())
                ),
                (
                    "laguna-s-2.1-Q4_K_M.gguf".to_owned(),
                    Some("vision.gguf".to_owned())
                ),
                ("unlabelled-companion.gguf".to_owned(), None),
            ])
        );
    }

    #[test]
    fn declared_projector_wins_over_directory_guess() {
        use crate::test_gguf::{write_model, write_projector};
        let temporary = tempfile::tempdir().unwrap();
        let cache = ModelCache::new(&temporary.path().join("cache"));
        let models = temporary.path().join("models");
        write_model(&models.join("model.gguf"), "model");
        write_projector(&models.join("mmproj-f16.gguf"), "f16");
        write_projector(&models.join("mmproj-q8.gguf"), "q8");

        let undeclared = discover_groups(&models, &models, &cache, &BTreeMap::new()).unwrap();
        assert_eq!(
            undeclared[0].projector, None,
            "two projectors are ambiguous"
        );

        let declared = BTreeMap::from([(models.join("model.gguf"), models.join("mmproj-q8.gguf"))]);
        let groups = discover_groups(&models, &models, &cache, &declared).unwrap();
        assert_eq!(groups[0].projector, Some(models.join("mmproj-q8.gguf")));
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
        config.add_hugging_face_caches([hf_cache]);
        let manager = ManagedModelStore::open(config.clone()).await.unwrap();
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
        let persisted_bytes = fs::read(cache_root.join("indexes/inventory.json")).unwrap();
        let persisted: serde_json::Value = serde_json::from_slice(&persisted_bytes).unwrap();
        assert!(persisted.get("version").is_none());

        fs::remove_file(source.join("model.gguf")).unwrap();
        let reopened = ManagedModelStore::open(config.clone()).await.unwrap();
        let warm = reopened.list().await.unwrap();
        assert!(warm.is_empty());
        reopened.ensure_model_inventory().await.unwrap();
        assert!(reopened.list().await.unwrap().is_empty());

        // A changed identity is the only candidate enriched on the next reconciliation.
        write_minimal_gguf(&source.join("model.gguf"));
        fs::OpenOptions::new()
            .append(true)
            .open(source.join("model.gguf"))
            .unwrap()
            .write_all(&[0])
            .unwrap();
        reopened.ensure_model_inventory().await.unwrap();
        let changed = reopened.list().await.unwrap();
        assert_eq!(changed.len(), 1);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn startup_reconciliation_retries_after_a_scan_failure() {
        use std::os::unix::fs::PermissionsExt as _;

        let temporary = tempfile::tempdir().unwrap();
        let store = temporary.path().join("store");
        let snapshots = store.join("hub/models--owner--model/snapshots");
        fs::create_dir_all(&snapshots).unwrap();
        fs::set_permissions(&snapshots, fs::Permissions::from_mode(0o000)).unwrap();

        let config = InventoryConfig::with_roots(store, temporary.path().join("cache")).unwrap();
        let manager = ManagedModelStore::open(config).await.unwrap();

        // Leave the invalid tree in place beyond the first retry boundary. A one-shot startup
        // reconciliation would have exited permanently before the tree is repaired.
        tokio::time::sleep(BACKGROUND_RECONCILIATION_RETRY_DELAY.saturating_mul(2)).await;
        assert!(
            !manager
                .installed_packages_response()
                .unwrap()
                .reconciliation_complete
        );
        fs::set_permissions(&snapshots, fs::Permissions::from_mode(0o755)).unwrap();

        tokio::time::timeout(
            BACKGROUND_RECONCILIATION_RETRY_DELAY.saturating_mul(3),
            async {
                loop {
                    if manager
                        .installed_packages_response()
                        .unwrap()
                        .reconciliation_complete
                    {
                        break;
                    }
                    tokio::task::yield_now().await;
                }
            },
        )
        .await
        .expect("background reconciliation should recover");
    }

    #[tokio::test]
    async fn malformed_index_entry_is_isolated_from_other_candidates() {
        let temporary = tempfile::tempdir().unwrap();
        let store = temporary.path().join("store");
        let cache_root = temporary.path().join("cache");
        let (hf_cache, source) = create_hf_snapshot(temporary.path());
        write_minimal_gguf(&source.join("first.gguf"));
        write_minimal_gguf(&source.join("second.gguf"));

        let mut config = InventoryConfig::with_roots(store.clone(), cache_root.clone()).unwrap();
        config.add_hugging_face_caches([hf_cache]);
        let manager = ManagedModelStore::open(config.clone()).await.unwrap();
        manager.ensure_installed_model_inventory().await.unwrap();
        assert_eq!(manager.list().await.unwrap().len(), 2);

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

        let reopened = ManagedModelStore::open(config.clone()).await.unwrap();
        reopened.ensure_installed_model_inventory().await.unwrap();
        assert_eq!(reopened.list().await.unwrap().len(), 2);

        let metadata_dir = cache_root.join("indexes/inventory/model-metadata");
        let one_metadata = fs::read_dir(&metadata_dir)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        fs::write(one_metadata, b"not json").unwrap();
        fs::write(&index_path, b"not json").unwrap();
        let corrupted = ManagedModelStore::open(config.clone()).await.unwrap();
        corrupted.ensure_installed_model_inventory().await.unwrap();
        assert_eq!(corrupted.list().await.unwrap().len(), 2);

        fs::remove_file(&index_path).unwrap();
        fs::create_dir(&index_path).unwrap();
        let uncached = ManagedModelStore::open(config).await.unwrap();
        uncached.ensure_installed_model_inventory().await.unwrap();
        assert_eq!(uncached.list().await.unwrap().len(), 2);
    }
}
