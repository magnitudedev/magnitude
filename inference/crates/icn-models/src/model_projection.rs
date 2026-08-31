use std::collections::BTreeMap;

use icn_contracts::models::{
    EffectiveModel, InstalledModelPackage, ModelCapabilities, ModelFailure, ModelFileRole,
    ModelInstallationOwnership, ModelMetadata, ModelPackage, ModelPackageId,
    ModelPackageInspection, ModelPackageInstallationOrigin, ModelServingConfiguration, ReadyModel,
    ResolvedModelInstallation, ServableModelBundle, ServingProfile,
};

const MINIMUM_EXTERNAL_CONTEXT: u32 = 4_096;

pub(crate) fn effective_model(
    installed: &InstalledModelPackage,
    profile: ServingProfile,
) -> (EffectiveModel, ResolvedModelInstallation) {
    let configuration = ModelServingConfiguration {
        bundle: ServableModelBundle::Standalone {
            package: installed.package.clone(),
        },
        profile,
    };
    (
        effective_configuration_model(
            &configuration,
            &BTreeMap::from([(installed.package.id.clone(), installed)]),
        ),
        resolved_installation(installed),
    )
}

pub(crate) fn resolved_installation(
    installed: &InstalledModelPackage,
) -> ResolvedModelInstallation {
    let installed_bytes = installed
        .package
        .files
        .iter()
        .map(|file| file.size_bytes)
        .sum();
    ResolvedModelInstallation::Resolved {
        installed_bytes,
        primary_path: primary_model_path(installed),
        ownership: match installed.origin {
            ModelPackageInstallationOrigin::Magnitude => ModelInstallationOwnership::Magnitude,
            ModelPackageInstallationOrigin::HuggingFaceCache => {
                ModelInstallationOwnership::ExternalHuggingFace
            }
        },
    }
}

pub(crate) fn primary_model_path(installed: &InstalledModelPackage) -> std::path::PathBuf {
    if !installed.path.is_dir() {
        return installed.path.clone();
    }
    installed
        .package
        .files
        .iter()
        .filter(|file| file.role == ModelFileRole::Weights)
        .map(|file| &file.path)
        .min()
        .map_or_else(|| installed.path.clone(), |path| installed.path.join(path))
}

pub(crate) fn effective_configuration_model(
    configuration: &ModelServingConfiguration,
    present: &BTreeMap<ModelPackageId, &InstalledModelPackage>,
) -> EffectiveModel {
    let target = match &configuration.bundle {
        ServableModelBundle::Standalone { package }
        | ServableModelBundle::SpeculativeDecoding {
            target: package, ..
        } => package,
    };
    let Some(installed_target) = present.get(&target.id).copied() else {
        return EffectiveModel::Unavailable {
            failure: ModelFailure {
                code: "model_material_missing".to_owned(),
                message: "The selected model target is not installed".to_owned(),
                retryable: true,
            },
        };
    };
    for package in bundle_packages(&configuration.bundle) {
        let Some(installed) = present.get(&package.id).copied() else {
            return EffectiveModel::Unavailable {
                failure: ModelFailure {
                    code: "model_dependency_missing".to_owned(),
                    message: "Required model material is not installed".to_owned(),
                    retryable: true,
                },
            };
        };
        if let Some(failure) = inspection_failure(installed) {
            return EffectiveModel::Unavailable { failure };
        }
    }
    if configuration.profile.context_length < MINIMUM_EXTERNAL_CONTEXT {
        return EffectiveModel::Unavailable {
            failure: ModelFailure {
                code: "model_context_too_small".to_owned(),
                message: "The selected target has no supported serving context".to_owned(),
                retryable: false,
            },
        };
    }
    let ModelPackageInspection::Inspected { capabilities } = &installed_target.inspection else {
        unreachable!("inspection failures returned before effective model construction")
    };
    EffectiveModel::Ready {
        model: ready_model(
            &configuration.bundle,
            configuration.profile.clone(),
            capabilities.clone(),
        ),
    }
}

fn inspection_failure(installed: &InstalledModelPackage) -> Option<ModelFailure> {
    match &installed.inspection {
        ModelPackageInspection::Inspected { .. } => None,
        ModelPackageInspection::Invalid { failure }
        | ModelPackageInspection::Incompatible { failure } => Some(failure.clone()),
        ModelPackageInspection::Pending => Some(ModelFailure {
            code: "model_inspection_pending".to_owned(),
            message: "The selected target has not been inspected yet".to_owned(),
            retryable: true,
        }),
    }
}

pub(crate) fn ready_model(
    bundle: &ServableModelBundle,
    profile: ServingProfile,
    capabilities: ModelCapabilities,
) -> ReadyModel {
    let package = match bundle {
        ServableModelBundle::Standalone { package }
        | ServableModelBundle::SpeculativeDecoding {
            target: package, ..
        } => package,
    };
    ReadyModel {
        metadata: ModelMetadata {
            format: package.properties.format.clone(),
            architecture: package.properties.architecture.clone(),
            quantization: package.properties.quantization.clone(),
            quantization_name: package.properties.quantization_name.clone(),
            storage_bytes: bundle_packages(bundle)
                .flat_map(|package| &package.files)
                .map(|file| file.size_bytes)
                .sum(),
            maximum_context_length: package.properties.maximum_context_length,
        },
        profile,
        capabilities,
        speculative_method: match bundle {
            ServableModelBundle::Standalone { .. } => None,
            ServableModelBundle::SpeculativeDecoding { method, .. } => Some(method.clone()),
        },
    }
}

pub(crate) fn bundle_packages(
    bundle: &ServableModelBundle,
) -> Box<dyn Iterator<Item = &ModelPackage> + '_> {
    match bundle {
        ServableModelBundle::Standalone { package } => Box::new(std::iter::once(package)),
        ServableModelBundle::SpeculativeDecoding {
            target,
            draft_source,
            ..
        } => match draft_source {
            icn_contracts::models::SpeculativeDraftSource::Embedded => {
                Box::new(std::iter::once(target))
            }
            icn_contracts::models::SpeculativeDraftSource::Separate { draft } => {
                Box::new([target, draft].into_iter())
            }
        },
    }
}
