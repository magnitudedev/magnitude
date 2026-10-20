//! The one host-side resolution path (integration spec §8.5): a model's installed material
//! resolved into the engine's device-free configuration, cached by exact material identity.
//! Loads, host-only operations (counting, template application, properties) and protocol
//! request preparation share it; there is no second tokenizer, template or properties path.

use std::collections::{BTreeMap, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, Weak};

use magnitude_engine::assessment::ModelPackagePaths;
use magnitude_engine::composition::EngineConfiguration;
use magnitude_engine::error::ResolveError;
use magnitude_engine::host::HostArtifacts;
use magnitude_engine::options::{
    ExecutionManifest, ModelPolicy, PackageOptions, ProjectorSelection, standard_service_limits,
};
use magnitude_executor::ExecutionPath;
use magnitude_executor::platform::MemoryReserves;
use magnitude_service_contracts::InventoryError;
use magnitude_service_contracts::models::{
    InstalledModelPackages as _, ModelBundleInput, ModelPackageId, ModelPackageOperand,
    ModelServingConfiguration, ResolvedServableModelBundle, ServableModelBundle,
    SpeculativeDraftSource, SpeculativeDraftSourceInput,
};
use magnitude_service_models::{
    ManagedModelStore, ServableModelBundleKey, servable_model_bundle_key_for_bundle,
};

use crate::assessment::assessor::engine_material;
use crate::assessment::environment::{SERVING_DEVICE, serving_policy};

/// Resolved configurations kept for reuse beyond those still held elsewhere. A resolution holds
/// the model's package metadata, tokenizers and templates, 100 to 250 MB for a 250K-token
/// vocabulary, so only the most recently used one is kept; one still held, such as the resident
/// model's, is found again without being kept here.
const RETAINED_CONFIGURATIONS: usize = 1;

/// One model's device-free engine configuration.
pub struct ResolvedConfiguration {
    /// Chat semantics: tokenizer, templates, properties.
    pub host: Arc<HostArtifacts>,
    /// What a worker loads. Its device request is automatic; a load replaces it with the
    /// previewed device.
    pub manifest: ExecutionManifest,
    /// The installed packages the configuration reads.
    pub package_ids: Vec<ModelPackageId>,
    /// Keeps the resolved material alive while the configuration is in use.
    _material: ResolvedServableModelBundle,
}

/// The shared resolved-configuration cache.
pub struct ResolvedConfigurations {
    models: Arc<ManagedModelStore>,
    kernel_directory: PathBuf,
    reserves: MemoryReserves,
    cache: ResolutionCache<ResolvedConfiguration>,
}

impl ResolvedConfigurations {
    pub fn new(
        models: Arc<ManagedModelStore>,
        kernel_directory: PathBuf,
        reserves: MemoryReserves,
    ) -> Self {
        Self {
            models,
            kernel_directory,
            reserves,
            cache: ResolutionCache::new(RETAINED_CONFIGURATIONS),
        }
    }

    /// The configuration's resolution, resolving it once for concurrent callers. A resolution is
    /// kept by its bundle and the files its packages are at now: a package is named by its
    /// content, which may be in several places or move, so a file moved, or the copy that served
    /// it gone, resolves afresh from where the package is, never from where it was.
    pub async fn resolve(
        &self,
        configuration: &ModelServingConfiguration,
    ) -> Result<Arc<ResolvedConfiguration>, InventoryError> {
        let (input, package_ids) = installed_bundle(&configuration.bundle);
        let material = self.models.resolve_bundle(input).await?;
        let paths = engine_material(&material)?;
        let key = ResolutionKey {
            bundle: servable_model_bundle_key_for_bundle(&configuration.bundle),
            files: std::iter::once(paths.target.clone())
                .chain(paths.projector.clone())
                .chain(paths.draft.clone())
                .collect(),
        };
        self.cache
            .resolve(key, || self.resolve_material(material, paths, package_ids))
            .await
    }

    /// Forget every configuration reading one of `package_ids`, before those packages are
    /// removed.
    pub fn evict_packages(&self, package_ids: &[ModelPackageId]) {
        self.cache.forget_where(|resolved| {
            resolved
                .package_ids
                .iter()
                .any(|package_id| package_ids.contains(package_id))
        });
    }

    async fn resolve_material(
        &self,
        material: ResolvedServableModelBundle,
        paths: ModelPackagePaths,
        package_ids: Vec<ModelPackageId>,
    ) -> Result<Arc<ResolvedConfiguration>, InventoryError> {
        let configuration =
            engine_configuration(paths, self.kernel_directory.clone(), self.reserves);
        let resolved = crate::spawn_blocking_traced(move || configuration.resolve())
            .await
            .map_err(|error| {
                InventoryError::Internal(format!("model resolution task failed: {error}"))
            })?
            .map_err(resolve_failure)?;
        Ok(Arc::new(ResolvedConfiguration {
            host: Arc::new(resolved.host),
            manifest: resolved.manifest,
            package_ids,
            _material: material,
        }))
    }
}

/// The engine configuration serving a bundle's material: its components, its declared method
/// (a separate draft's method is the bundle's, never `Auto`), the standard service limits.
fn engine_configuration(
    package: ModelPackagePaths,
    kernel_directory: PathBuf,
    reserves: MemoryReserves,
) -> EngineConfiguration {
    EngineConfiguration {
        package: PackageOptions {
            target: package.target,
            projector: match package.projector {
                Some(projector) => ProjectorSelection::Explicit(projector),
                None => ProjectorSelection::Disabled,
            },
            draft: package.draft,
        },
        model: ModelPolicy {
            method: package.method,
            ..serving_policy()
        },
        context_tokens: None,
        service: standard_service_limits(),
        path: ExecutionPath::Native,
        device: SERVING_DEVICE,
        kernel_cache: Some(kernel_directory),
        reserves,
    }
}

/// The bundle over its installed packages, and the packages it reads.
fn installed_bundle(bundle: &ServableModelBundle) -> (ModelBundleInput, Vec<ModelPackageId>) {
    let installed = |package_id: &ModelPackageId| ModelPackageOperand::Installed {
        package_id: package_id.clone(),
    };
    match bundle {
        ServableModelBundle::Standalone { package } => (
            ModelBundleInput::Standalone {
                package: installed(&package.id),
            },
            vec![package.id.clone()],
        ),
        ServableModelBundle::SpeculativeDecoding {
            target,
            draft_source,
            method,
        } => {
            let (draft_source, package_ids) = match draft_source {
                SpeculativeDraftSource::Embedded => {
                    (SpeculativeDraftSourceInput::Embedded, vec![target.id.clone()])
                }
                SpeculativeDraftSource::Separate { draft } => (
                    SpeculativeDraftSourceInput::Separate {
                        draft: installed(&draft.id),
                    },
                    vec![target.id.clone(), draft.id.clone()],
                ),
            };
            (
                ModelBundleInput::SpeculativeDecoding {
                    target: installed(&target.id),
                    draft_source,
                    method: method.clone(),
                },
                package_ids,
            )
        }
    }
}

/// A resolution failure as the service reports it.
pub fn resolve_failure(error: ResolveError) -> InventoryError {
    let code = match &error {
        ResolveError::Unsupported(_) => "unsupported_model",
        ResolveError::Artifact(_) => "invalid_model_package",
        ResolveError::InvalidConfiguration { .. } => "invalid_model_configuration",
    };
    InventoryError::ModelOperation {
        code: code.to_owned(),
        message: error.to_string(),
        retryable: false,
    }
}

/// A resolution's identity: the bundle, and the files its material is read from.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ResolutionKey {
    bundle: ServableModelBundleKey,
    files: Vec<PathBuf>,
}

type Entry<T> = Arc<tokio::sync::OnceCell<Arc<T>>>;

/// Resolutions by key: the `retained` most recently used, and every one something else still
/// holds, which is found again rather than resolved twice. Concurrent callers for one key share
/// one resolution; a failed resolution is not kept.
struct ResolutionCache<T, K = ResolutionKey> {
    retained: usize,
    recent: Mutex<VecDeque<(K, Entry<T>)>>,
    held: Mutex<BTreeMap<K, Weak<T>>>,
}

impl<T, K: Clone + Ord> ResolutionCache<T, K> {
    fn new(retained: usize) -> Self {
        Self {
            retained,
            recent: Mutex::new(VecDeque::new()),
            held: Mutex::new(BTreeMap::new()),
        }
    }

    async fn resolve<E, F, Fut>(&self, key: K, resolve: F) -> Result<Arc<T>, E>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<Arc<T>, E>>,
    {
        let entry = self.entry(&key);
        let resolved = entry
            .get_or_try_init(|| async {
                match self.still_held(&key) {
                    Some(resolved) => Ok(resolved),
                    None => resolve().await,
                }
            })
            .await
            .cloned();
        match &resolved {
            Ok(resolved) => self.hold(key, resolved),
            Err(_) => self.forget(&key, &entry),
        }
        resolved
    }

    fn forget_where(&self, matches: impl Fn(&T) -> bool) {
        self.recent
            .lock()
            .expect("resolution cache lock")
            .retain(|(_, entry)| entry.get().is_none_or(|resolved| !matches(resolved)));
        self.held
            .lock()
            .expect("resolution cache lock")
            .retain(|_, resolved| {
                resolved
                    .upgrade()
                    .is_some_and(|resolved| !matches(&resolved))
            });
    }

    fn entry(&self, key: &K) -> Entry<T> {
        let mut recent = self.recent.lock().expect("resolution cache lock");
        if let Some(position) = recent.iter().position(|(existing, _)| existing == key) {
            let used = recent.remove(position).expect("position is in range");
            let entry = Arc::clone(&used.1);
            recent.push_back(used);
            return entry;
        }
        let entry = Entry::default();
        recent.push_back((key.clone(), Arc::clone(&entry)));
        while recent.len() > self.retained {
            recent.pop_front();
        }
        entry
    }

    fn still_held(&self, key: &K) -> Option<Arc<T>> {
        self.held
            .lock()
            .expect("resolution cache lock")
            .get(key)
            .and_then(Weak::upgrade)
    }

    fn hold(&self, key: K, resolved: &Arc<T>) {
        let mut held = self.held.lock().expect("resolution cache lock");
        held.retain(|_, resolved| resolved.strong_count() > 0);
        held.insert(key, Arc::downgrade(resolved));
    }

    fn forget(&self, key: &K, entry: &Entry<T>) {
        self.recent
            .lock()
            .expect("resolution cache lock")
            .retain(|(existing, existing_entry)| {
                existing != key || !Arc::ptr_eq(existing_entry, entry)
            });
    }

    #[cfg(test)]
    fn sizes(&self) -> (usize, usize) {
        (
            self.recent.lock().expect("resolution cache lock").len(),
            self.held.lock().expect("resolution cache lock").len(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use magnitude_engine::options::{ModelMethod, ResolvedMethod};

    fn material(method: ModelMethod) -> ModelPackagePaths {
        let path = |variable: &str| {
            PathBuf::from(std::env::var_os(variable).unwrap_or_else(|| panic!("set {variable}")))
        };
        ModelPackagePaths {
            target: path("MAGNITUDE_TEST_TARGET_GGUF"),
            projector: None,
            draft: Some(path("MAGNITUDE_TEST_DRAFT_GGUF")),
            method,
        }
    }

    /// A speculative bundle serves its declared separate draft; a declared
    /// method the draft does not implement refuses the bundle instead of
    /// serving its target plain.
    #[test]
    #[ignore = "requires MAGNITUDE_TEST_TARGET_GGUF and its DSpark MAGNITUDE_TEST_DRAFT_GGUF"]
    fn a_draft_bundle_serves_its_declared_method_or_is_refused() {
        let kernels = std::env::temp_dir();
        let resolved = engine_configuration(
            material(ModelMethod::DSpark),
            kernels.clone(),
            MemoryReserves::standard(),
        )
        .resolve()
        .unwrap();
        assert!(matches!(
            resolved.manifest.model.method,
            ResolvedMethod::DFlash { proposals } if proposals > 0
        ));
        let draft = resolved.manifest.definition.draft.as_ref().unwrap();
        assert_eq!(draft.method.variant().to_string(), "DSpark");
        assert!(resolved.manifest.package.draft.is_some());

        for method in [ModelMethod::DFlash, ModelMethod::DFlash2] {
            let Err(error) =
                engine_configuration(material(method), kernels.clone(), MemoryReserves::standard())
                    .resolve()
            else {
                panic!("{method:?} resolved over a DSpark draft")
            };
            let InventoryError::ModelOperation { code, message, .. } = resolve_failure(error)
            else {
                panic!("a resolution failure is a model operation failure")
            };
            assert_eq!(code, "invalid_model_configuration");
            assert!(message.contains("the package's draft is DSpark"), "{message}");
        }
    }

    fn key(index: usize) -> ServableModelBundleKey {
        ServableModelBundleKey(format!("bundle-{index}"))
    }

    async fn resolve_counting(
        cache: &ResolutionCache<String, ServableModelBundleKey>,
        index: usize,
        resolutions: &std::sync::atomic::AtomicUsize,
    ) -> Arc<String> {
        cache
            .resolve(key(index), || async {
                resolutions.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Ok::<_, ()>(Arc::new(format!("model {index}")))
            })
            .await
            .unwrap()
    }

    /// However many models are previewed, loaded or counted against, only the most recent
    /// resolution and those still held stay in memory.
    #[tokio::test]
    async fn a_long_run_of_resolutions_keeps_only_the_recent_and_the_held() {
        let cache =
            ResolutionCache::<String, ServableModelBundleKey>::new(RETAINED_CONFIGURATIONS);
        let resolutions = std::sync::atomic::AtomicUsize::new(0);
        let resident = resolve_counting(&cache, 0, &resolutions).await;
        for round in 0..2_000 {
            resolve_counting(&cache, 1 + round % 200, &resolutions).await;
            let (recent, held) = cache.sizes();
            assert!(
                recent <= RETAINED_CONFIGURATIONS,
                "round {round}: {recent} recent"
            );
            assert!(held <= 2, "round {round}: {held} held");
        }
        let before = resolutions.load(std::sync::atomic::Ordering::Relaxed);
        let again = resolve_counting(&cache, 0, &resolutions).await;
        assert!(
            Arc::ptr_eq(&resident, &again),
            "a held resolution is found again"
        );
        assert_eq!(
            resolutions.load(std::sync::atomic::Ordering::Relaxed),
            before
        );

        drop((resident, again));
        resolve_counting(&cache, 7, &resolutions).await;
        resolve_counting(&cache, 8, &resolutions).await;
        assert_eq!(cache.sizes(), (1, 1));
        resolve_counting(&cache, 0, &resolutions).await;
        assert_eq!(
            resolutions.load(std::sync::atomic::Ordering::Relaxed),
            before + 3,
            "a released resolution is resolved again"
        );
    }

    #[tokio::test]
    async fn forgetting_drops_recent_and_held_resolutions_and_a_failure_is_not_kept() {
        let cache =
            ResolutionCache::<String, ServableModelBundleKey>::new(RETAINED_CONFIGURATIONS);
        let resolutions = std::sync::atomic::AtomicUsize::new(0);
        let held = resolve_counting(&cache, 1, &resolutions).await;
        resolve_counting(&cache, 2, &resolutions).await;
        cache.forget_where(|resolved| resolved.as_str() == "model 1");
        resolve_counting(&cache, 1, &resolutions).await;
        assert_eq!(resolutions.load(std::sync::atomic::Ordering::Relaxed), 3);
        drop(held);

        let failed = cache
            .resolve(key(9), || async { Err::<Arc<String>, _>("unreadable") })
            .await;
        assert_eq!(failed, Err("unreadable"));
        resolve_counting(&cache, 9, &resolutions).await;
        assert_eq!(resolutions.load(std::sync::atomic::Ordering::Relaxed), 4);
    }
}
