//! The one host-side resolution path (integration spec §8.5): a model's installed material
//! resolved into the engine's device-free configuration, cached by exact material identity.
//! Loads, host-only operations (counting, template application, properties) and protocol
//! request preparation share it; there is no second tokenizer, template or properties path.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use magnitude_engine::composition::EngineConfiguration;
use magnitude_engine::error::{ResolveError, UnsupportedModel};
use magnitude_engine::host::HostArtifacts;
use magnitude_engine::options::{
    ExecutionManifest, PackageOptions, ProjectorSelection, standard_service_limits,
};
use magnitude_executor::ExecutionPath;
use magnitude_executor::platform::{DeviceRequest, MemoryReserves};
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
use crate::assessment::environment::serving_policy;

/// Resolved configurations kept for reuse. A resolution holds the model's tokenizer and
/// templates, so only the few most recently used models are kept.
const CACHED_CONFIGURATIONS: usize = 4;

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

type Entry = Arc<tokio::sync::OnceCell<Arc<ResolvedConfiguration>>>;

/// The shared resolved-configuration cache.
pub struct ResolvedConfigurations {
    models: Arc<ManagedModelStore>,
    kernel_directory: PathBuf,
    reserves: MemoryReserves,
    entries: Mutex<VecDeque<(ServableModelBundleKey, Entry)>>,
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
            entries: Mutex::new(VecDeque::new()),
        }
    }

    /// The configuration's resolution, resolving it once for concurrent callers.
    pub async fn resolve(
        &self,
        configuration: &ModelServingConfiguration,
    ) -> Result<Arc<ResolvedConfiguration>, InventoryError> {
        let key = servable_model_bundle_key_for_bundle(&configuration.bundle);
        let entry = self.entry(&key);
        let resolved = entry
            .get_or_try_init(|| self.resolve_uncached(&configuration.bundle))
            .await
            .cloned();
        if resolved.is_err() {
            self.forget(&key, &entry);
        }
        resolved
    }

    /// Forget every configuration reading one of `package_ids`, before those packages are
    /// removed.
    pub fn evict_packages(&self, package_ids: &[ModelPackageId]) {
        self.entries
            .lock()
            .expect("resolved configuration lock")
            .retain(|(_, entry)| {
                entry.get().is_none_or(|resolved| {
                    !resolved
                        .package_ids
                        .iter()
                        .any(|package_id| package_ids.contains(package_id))
                })
            });
    }

    fn entry(&self, key: &ServableModelBundleKey) -> Entry {
        let mut entries = self.entries.lock().expect("resolved configuration lock");
        if let Some(position) = entries.iter().position(|(existing, _)| existing == key) {
            let used = entries.remove(position).expect("position is in range");
            let entry = Arc::clone(&used.1);
            entries.push_back(used);
            return entry;
        }
        let entry = Entry::default();
        entries.push_back((key.clone(), Arc::clone(&entry)));
        if entries.len() > CACHED_CONFIGURATIONS {
            entries.pop_front();
        }
        entry
    }

    fn forget(&self, key: &ServableModelBundleKey, entry: &Entry) {
        self.entries
            .lock()
            .expect("resolved configuration lock")
            .retain(|(existing, existing_entry)| {
                existing != key || !Arc::ptr_eq(existing_entry, entry)
            });
    }

    async fn resolve_uncached(
        &self,
        bundle: &ServableModelBundle,
    ) -> Result<Arc<ResolvedConfiguration>, InventoryError> {
        let (input, package_ids) = installed_bundle(bundle);
        let material = self.models.resolve_bundle(input).await?;
        let package = engine_material(&material)?;
        let configuration = EngineConfiguration {
            package: PackageOptions {
                target: package.target,
                projector: match package.projector {
                    Some(projector) => ProjectorSelection::Explicit(projector),
                    None => ProjectorSelection::Disabled,
                },
            },
            model: serving_policy(),
            context_tokens: None,
            service: standard_service_limits(),
            path: ExecutionPath::Native,
            device: DeviceRequest::Automatic,
            kernel_cache: Some(self.kernel_directory.clone()),
            reserves: self.reserves,
        };
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
        ResolveError::Unsupported(UnsupportedModel::Family { .. })
        | ResolveError::Unsupported(UnsupportedModel::Representation { .. })
        | ResolveError::Unsupported(UnsupportedModel::Backend { .. }) => "unsupported_model",
        ResolveError::Artifact(_) => "invalid_model_package",
        ResolveError::InvalidConfiguration { .. } => "invalid_model_configuration",
    };
    InventoryError::ModelOperation {
        code: code.to_owned(),
        message: error.to_string(),
        retryable: false,
    }
}
