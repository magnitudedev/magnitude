//! Per-target model assessment: exact material resolution, the per-profile cache, deduplication
//! of concurrent equivalent work, and the engine's analytical assessment on the blocking pool.
//!
//! Assessment reads only GGUF headers (the release catalog's header bundle before download,
//! installed files after), opens no device and loads no model.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Weak};

use magnitude_engine::assessment::{
    AssessmentSetup, ModelAssessment as EngineAssessment, ModelAssessmentError,
    ModelCapabilities as EngineCapabilities, ModelPackagePaths, PreparedModelAssessment,
    finish_model_assessment, prepare_model_assessment,
};
use magnitude_engine::error::UnsupportedModel;
use magnitude_executor::assessment::{
    DomainFit, ExecutionAssessment, IncompatibleReason, PerformanceConfidence as EngineConfidence,
    PerformanceEstimate,
};
use magnitude_service_contracts::models::{
    AssessmentEnvironmentId, InstalledModelPackages as _, MemoryAssessment, ModelAssessment,
    ModelAssessmentId, ModelAssessmentProfile, ModelBundleInput, ModelCapabilities, ModelFailure,
    ModelPackageOperand, ModelReasoningCapabilities, ModelServingConfiguration,
    PerformanceConfidence, PerformanceEvidence, ResolvedServableModelBundle, ServableModelBundle,
    SpeculativeDraftSource, SpeculativeDraftSourceInput,
};
use magnitude_service_contracts::{ComponentRole, InventoryError, MemoryDomainId};
use magnitude_service_models::{
    CachedModelAssessment, ManagedModelStore, ModelDomainResolver, ReleaseCatalog,
    ServableModelBundleKey, servable_model_bundle_key_for_bundle,
};
use sha2::{Digest, Sha256};

use super::environment::{AssessmentEnvironment, EnvironmentError};
use super::measurement::MeasurementJob;

/// The service's performance depths: 25K, 50K and 75K where the context admits them, then the
/// full context. ACN ranking reads `min(50_000, context)`, which this set contains.
pub fn performance_depths(context_length: u32) -> Vec<u32> {
    let mut depths = [25_000, 50_000, 75_000]
        .into_iter()
        .filter(|depth| *depth < context_length)
        .collect::<Vec<_>>();
    depths.push(context_length);
    depths
}

/// One exact unit of assessment work: an environment, a resolved bundle and one profile.
#[derive(Clone)]
pub struct AssessmentWork {
    pub key: AssessmentWorkKey,
    pub profile: ModelAssessmentProfile,
    pub configuration: ModelServingConfiguration,
    pub preparation: Arc<tokio::sync::OnceCell<PreparationResult>>,
}

pub type PreparationResult = Result<Arc<PreparedModelAssessment>, ModelFailure>;

fn preparation_task_failure(error: impl std::fmt::Display) -> ModelFailure {
    inventory_model_failure(InventoryError::Internal(format!(
        "model preparation task failed: {error}"
    )))
}

fn preparation_failure(error: ModelAssessmentError) -> ModelFailure {
    inventory_model_failure(InventoryError::ModelOperation {
        code: "assessment_failed".to_owned(),
        message: error.to_string(),
        retryable: true,
    })
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct AssessmentWorkKey(pub String);

impl AssessmentWorkKey {
    pub fn new(
        environment: &AssessmentEnvironmentId,
        bundle: &ServableModelBundleKey,
        profile: &ModelAssessmentProfile,
    ) -> Self {
        Self(
            serde_json::to_string(&(&environment.0, &bundle.0, profile))
                .expect("assessment work identity is serializable"),
        )
    }
}

pub struct AssessmentOutcome {
    pub key: AssessmentWorkKey,
    pub result: Result<CachedModelAssessment, ModelFailure>,
}

/// Assesses resolved targets in one environment.
pub struct ModelAssessor {
    models: Arc<ManagedModelStore>,
    model_domains: Arc<ModelDomainResolver>,
    release_catalog: Arc<ReleaseCatalog>,
    catalog: Arc<seismic::DeviceCatalog>,
    measurement: MeasurementJob,
    work_gates: tokio::sync::Mutex<BTreeMap<String, Weak<tokio::sync::Mutex<()>>>>,
}

impl ModelAssessor {
    /// `catalog` is the service's device catalog, shared with residency and the hardware
    /// endpoint.
    pub fn new(
        models: Arc<ManagedModelStore>,
        model_domains: Arc<ModelDomainResolver>,
        release_catalog: Arc<ReleaseCatalog>,
        catalog: Arc<seismic::DeviceCatalog>,
        measurement: MeasurementJob,
    ) -> Self {
        Self {
            models,
            model_domains,
            release_catalog,
            catalog,
            measurement,
            work_gates: tokio::sync::Mutex::new(BTreeMap::new()),
        }
    }

    pub async fn select_setup(&self) -> Result<Arc<AssessmentSetup>, EnvironmentError> {
        AssessmentEnvironment::select(Arc::clone(&self.catalog)).await
    }

    /// The environment of `setup` with the device's measured basis.
    pub async fn establish_with_setup(
        &self,
        setup: Arc<AssessmentSetup>,
    ) -> Result<AssessmentEnvironment, EnvironmentError> {
        AssessmentEnvironment::establish(setup, &self.measurement).await
    }

    pub fn configuration_for(
        &self,
        subject: &magnitude_service_contracts::models::ModelAssessmentSubject,
    ) -> Result<ModelServingConfiguration, InventoryError> {
        self.model_domains.assessment_configuration(subject)
    }

    pub fn work_for(
        &self,
        environment: &AssessmentEnvironmentId,
        configuration: ModelServingConfiguration,
        profile: magnitude_service_contracts::models::ServingProfile,
        preparation: Arc<tokio::sync::OnceCell<PreparationResult>>,
    ) -> Result<AssessmentWork, InventoryError> {
        let profile = ModelAssessmentProfile {
            performance_context_tokens: performance_depths(profile.context_length),
            profile,
        };
        let bundle = servable_model_bundle_key_for_bundle(&configuration.bundle);
        Ok(AssessmentWork {
            key: AssessmentWorkKey::new(environment, &bundle, &profile),
            profile,
            configuration,
            preparation,
        })
    }

    /// Resolve and prepare one bundle from its headers, independent of the measurement basis.
    pub async fn prepare_bundle(
        &self,
        configuration: ModelServingConfiguration,
        setup: Arc<AssessmentSetup>,
    ) -> PreparationResult {
        let bundle_key = servable_model_bundle_key_for_bundle(&configuration.bundle);
        let resolved = self
            .resolve(&bundle_key, configuration.bundle)
            .await
            .map_err(assessment_drop_failure)?;
        let package = engine_material(&resolved).map_err(assessment_drop_failure)?;
        let prepared = crate::spawn_blocking_traced(move || {
            // The resolved material (a catalog bundle's materialized headers) lives until
            // preparation has read it.
            let _material = resolved;
            prepare_model_assessment(&package, &setup)
        })
        .await
        .map_err(preparation_task_failure)?
        .map_err(preparation_failure)?;
        Ok(Arc::new(prepared))
    }

    /// One attempt at `work`. `Ok` carries the target's terminal disposition; `Err` is an
    /// operational failure the pool records as `Dropped`.
    pub async fn assess(
        &self,
        work: AssessmentWork,
        environment: Arc<AssessmentEnvironment>,
    ) -> Result<AssessmentOutcome, InventoryError> {
        let AssessmentWork {
            key,
            profile,
            configuration,
            preparation,
        } = work;
        let bundle_key = servable_model_bundle_key_for_bundle(&configuration.bundle);
        // The exact work identity is the whole assessment identity, so it keys the cache.
        let evidence = key.0.clone();
        if let Some(cached) = self.models.read_model_assessment(&evidence) {
            return Ok(AssessmentOutcome {
                key,
                result: Ok(cached),
            });
        }
        // Serialize misses for one exact target in one environment; a waiter rechecks the cache
        // after admission so overlapping equivalent work reuses the owner's result.
        let gate_key = serde_json::to_string(&(&bundle_key.0, &environment.id.0))
            .expect("assessment gate identity is serializable");
        let _gate = self.work_gate(&gate_key).await.lock_owned().await;
        if let Some(cached) = self.models.read_model_assessment(&evidence) {
            return Ok(AssessmentOutcome {
                key,
                result: Ok(cached),
            });
        }
        let prepared = preparation
            .get_or_init(|| {
                self.prepare_bundle(configuration, Arc::clone(&environment.engine.setup))
            })
            .await;
        let prepared = match prepared {
            Ok(prepared) => Arc::clone(prepared),
            Err(failure) => {
                return Ok(AssessmentOutcome {
                    key,
                    result: Err(failure.clone()),
                });
            }
        };
        let assessed = assess_prepared(prepared, &bundle_key, profile, environment).await?;
        self.models.write_model_assessment(&evidence, &assessed);
        Ok(AssessmentOutcome {
            key,
            result: Ok(assessed),
        })
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

    /// The bundle's exact material: the release catalog's header bundle for catalog bundles,
    /// installed files for everything else.
    async fn resolve(
        &self,
        bundle_key: &ServableModelBundleKey,
        bundle: ServableModelBundle,
    ) -> Result<ResolvedServableModelBundle, InventoryError> {
        let release_catalog = Arc::clone(&self.release_catalog);
        let release_key = bundle_key.clone();
        let release =
            crate::spawn_blocking_traced(move || release_catalog.resolve_bundle(&release_key))
                .await
                .map_err(|error| {
                    InventoryError::Internal(format!(
                        "release model material task failed for {}: {error}",
                        bundle_key.0
                    ))
                })??;
        match release {
            Some(resolved) => Ok(resolved),
            None => self.models.resolve_bundle(bundle_input(bundle)).await,
        }
    }
}

/// The components the engine reads: the target GGUF (a split target by its first shard, beside
/// which the engine finds the others) and its projector. A draft package is not executed by the
/// engine.
pub(crate) fn engine_material(
    resolved: &ResolvedServableModelBundle,
) -> Result<ModelPackagePaths, InventoryError> {
    let components = &resolved.target_model.components;
    let target = components
        .iter()
        .filter(|component| {
            matches!(
                component.role,
                ComponentRole::Weights | ComponentRole::Shard
            )
        })
        .min_by_key(|component| component.shard_index.unwrap_or(0))
        .map(|component| component.path.clone())
        .ok_or_else(|| InventoryError::NotReady("model has no runnable weights".to_owned()))?;
    let projectors = components
        .iter()
        .filter(|component| component.role == ComponentRole::Projector)
        .map(|component| component.path.clone())
        .collect::<Vec<PathBuf>>();
    let projector = match projectors.as_slice() {
        [] => None,
        [projector] => Some(projector.clone()),
        _ => {
            return Err(InventoryError::InvalidRequest(
                "model resolves more than one projector".to_owned(),
            ));
        }
    };
    Ok(ModelPackagePaths { target, projector })
}

async fn assess_prepared(
    prepared: Arc<PreparedModelAssessment>,
    bundle_key: &ServableModelBundleKey,
    profile: ModelAssessmentProfile,
    environment: Arc<AssessmentEnvironment>,
) -> Result<CachedModelAssessment, InventoryError> {
    let depths = profile.performance_context_tokens.clone();
    let engine_environment = Arc::clone(&environment);
    let started = std::time::Instant::now();
    let assessed = crate::spawn_blocking_traced(move || {
        finish_model_assessment(&prepared, &engine_environment.engine, &depths)
    })
    .await
    .map_err(|error| InventoryError::Internal(format!("model assessment task failed: {error}")))?;
    tracing::info!(
        target.id = %bundle_key.0,
        assessment_microseconds = u64::try_from(started.elapsed().as_micros()).unwrap_or(u64::MAX),
        outcome = if assessed.is_ok() { "assessed" } else { "failed" },
        "engine model assessment completed"
    );
    if let Err(error) = &assessed {
        tracing::warn!(
            target.id = %bundle_key.0,
            %error,
            "engine model assessment failed"
        );
    }
    let assessed = assessed.map_err(|error| InventoryError::ModelOperation {
        code: "assessment_failed".to_owned(),
        message: error.to_string(),
        retryable: true,
    })?;
    model_assessment(assessed, bundle_key, profile, &environment)
}

/// Map the engine's complete result onto the service contract.
fn model_assessment(
    assessed: EngineAssessment,
    bundle_key: &ServableModelBundleKey,
    requested: ModelAssessmentProfile,
    environment: &AssessmentEnvironment,
) -> Result<CachedModelAssessment, InventoryError> {
    let (facts, execution) = match assessed {
        EngineAssessment::Unsupported(unsupported) => {
            let code = match &unsupported {
                UnsupportedModel::Family { .. } => "unsupported_family",
                UnsupportedModel::Representation { .. } => "unsupported_representation",
                UnsupportedModel::Backend { .. } => "unsupported_backend",
            };
            return Ok(incompatible(requested, code, unsupported.to_string()));
        }
        EngineAssessment::Assessed { facts, execution } => (facts, execution),
    };
    if facts.context_limit != requested.profile.context_length {
        return Err(InventoryError::ModelOperation {
            code: "assessment_context_mismatch".to_owned(),
            message: format!(
                "the engine serves {} context tokens but the profile names {}",
                facts.context_limit, requested.profile.context_length
            ),
            retryable: false,
        });
    }
    let assessment_id = assessment_id(bundle_key, &requested, &environment.id);
    let ModelAssessmentProfile {
        profile,
        performance_context_tokens,
    } = requested;
    let assessment = match execution {
        ExecutionAssessment::Fits {
            domains,
            performance,
            ..
        } => {
            let evidence = performance
                .iter()
                .map(performance_evidence)
                .collect::<Vec<_>>();
            if evidence
                .iter()
                .map(|sample| sample.context_tokens)
                .ne(performance_context_tokens.iter().copied())
            {
                return Err(InventoryError::ModelOperation {
                    code: "assessment_incomplete".to_owned(),
                    message: "the engine did not estimate every requested depth".to_owned(),
                    retryable: false,
                });
            }
            ModelAssessment::Fits {
                profile,
                assessment_id,
                memory: memory_assessments(&domains, environment),
                performance: evidence,
            }
        }
        ExecutionAssessment::DoesNotFit {
            domains,
            limiting,
            deficit_bytes,
            ..
        } => ModelAssessment::DoesNotFit {
            profile,
            assessment_id,
            memory: memory_assessments(&domains, environment),
            limiting_resource: domain_id(limiting, environment).as_str().to_owned(),
            deficit_bytes,
        },
        ExecutionAssessment::Incompatible { reason } => {
            let (code, message) = match reason {
                IncompatibleReason::OutsideBasis { classes } => (
                    "unsupported_operation",
                    format!(
                        "the device's measurement basis does not cover {}",
                        classes
                            .iter()
                            .map(|(key, reason)| match reason {
                                Some(reason) => format!("{key}: {reason}"),
                                None => key.to_string(),
                            })
                            .collect::<Vec<_>>()
                            .join("; ")
                    ),
                ),
                IncompatibleReason::Unsupported { reason } => {
                    ("unsupported_representation", reason)
                }
            };
            ModelAssessment::Incompatible {
                profile,
                failure: ModelFailure {
                    code: code.to_owned(),
                    message,
                    retryable: false,
                },
            }
        }
    };
    Ok(CachedModelAssessment {
        capabilities: capabilities(facts.capabilities),
        template_fingerprint: facts.template_fingerprint,
        profile: assessment,
    })
}

/// A package the engine cannot interpret has no engine-derived capabilities or template.
fn incompatible(
    requested: ModelAssessmentProfile,
    code: &str,
    message: String,
) -> CachedModelAssessment {
    CachedModelAssessment {
        capabilities: ModelCapabilities {
            vision: false,
            tools: false,
            structured_output: false,
            reasoning: ModelReasoningCapabilities {
                supported: false,
                efforts: Vec::new(),
                default_effort: None,
            },
        },
        template_fingerprint: String::new(),
        profile: ModelAssessment::Incompatible {
            profile: requested.profile,
            failure: ModelFailure {
                code: code.to_owned(),
                message,
                retryable: false,
            },
        },
    }
}

fn capabilities(capabilities: EngineCapabilities) -> ModelCapabilities {
    ModelCapabilities {
        vision: capabilities.vision,
        tools: capabilities.tools,
        structured_output: capabilities.structured_output,
        reasoning: ModelReasoningCapabilities {
            supported: capabilities.reasoning.supported(),
            efforts: capabilities.reasoning.efforts,
            default_effort: capabilities.reasoning.default_effort,
        },
    }
}

fn domain_id(domain: seismic::MemoryPoolId, environment: &AssessmentEnvironment) -> MemoryDomainId {
    crate::memory_domains::pool_domain_id(&environment.engine.setup.topology, domain)
}

fn memory_assessments(
    domains: &[DomainFit],
    environment: &AssessmentEnvironment,
) -> Vec<MemoryAssessment> {
    domains
        .iter()
        .map(|domain| MemoryAssessment {
            memory_domain_id: domain_id(domain.domain, environment),
            capacity_bytes: domain.capacity_bytes,
            required_bytes: domain.required_bytes,
            compatibility_reserve_bytes: domain.reserve_bytes,
            remaining_bytes: domain.remaining_bytes,
        })
        .collect()
}

fn performance_evidence(estimate: &PerformanceEstimate) -> PerformanceEvidence {
    PerformanceEvidence {
        context_tokens: estimate.context_tokens,
        lower_tokens_per_second: estimate.lower_tokens_per_second,
        estimated_tokens_per_second: estimate.estimated_tokens_per_second,
        upper_tokens_per_second: estimate.upper_tokens_per_second,
        confidence: match estimate.confidence {
            EngineConfidence::High => PerformanceConfidence::High,
            EngineConfidence::Moderate => PerformanceConfidence::Moderate,
            EngineConfidence::Low => PerformanceConfidence::Low,
        },
    }
}

fn assessment_id(
    bundle_key: &ServableModelBundleKey,
    profile: &ModelAssessmentProfile,
    environment: &AssessmentEnvironmentId,
) -> ModelAssessmentId {
    let mut digest = Sha256::new();
    digest.update(bundle_key.0.as_bytes());
    digest.update(profile.profile.context_length.to_le_bytes());
    digest.update(environment.0.as_bytes());
    for context_tokens in &profile.performance_context_tokens {
        digest.update(context_tokens.to_le_bytes());
    }
    ModelAssessmentId(format!("assessment_{:x}", digest.finalize()))
}

fn bundle_input(bundle: ServableModelBundle) -> ModelBundleInput {
    match bundle {
        ServableModelBundle::Standalone { package } => ModelBundleInput::Standalone {
            package: ModelPackageOperand::SourceBacked { package },
        },
        ServableModelBundle::SpeculativeDecoding {
            target,
            draft_source,
            method,
        } => ModelBundleInput::SpeculativeDecoding {
            target: ModelPackageOperand::SourceBacked { package: target },
            draft_source: match draft_source {
                SpeculativeDraftSource::Embedded => SpeculativeDraftSourceInput::Embedded,
                SpeculativeDraftSource::Separate { draft } => {
                    SpeculativeDraftSourceInput::Separate {
                        draft: ModelPackageOperand::SourceBacked { package: draft },
                    }
                }
            },
            method,
        },
    }
}

/// The dropped disposition of a target whose material could not be resolved.
pub fn assessment_drop_failure(error: InventoryError) -> ModelFailure {
    match error {
        error @ (InventoryError::InvalidId(_)
        | InventoryError::InvalidRequest(_)
        | InventoryError::NotFound(_)) => ModelFailure {
            code: "invalid_target".to_owned(),
            message: error.to_string(),
            retryable: false,
        },
        InventoryError::ModelOperation {
            code,
            message,
            retryable: false,
        } => ModelFailure {
            code,
            message,
            retryable: false,
        },
        error => inventory_model_failure(error),
    }
}

pub fn inventory_model_failure(error: InventoryError) -> ModelFailure {
    let (code, retryable) = match &error {
        InventoryError::InvalidId(_) => ("invalid_id".to_owned(), false),
        InventoryError::InvalidRequest(_) => ("invalid_request".to_owned(), false),
        InventoryError::NotFound(_) => ("not_found".to_owned(), false),
        InventoryError::NotReady(_) => ("not_ready".to_owned(), true),
        InventoryError::Busy(_) => ("busy".to_owned(), true),
        InventoryError::Loaded(_) => ("already_loaded".to_owned(), false),
        InventoryError::DeletionUnsafe(_) => ("deletion_unsafe".to_owned(), false),
        InventoryError::Unsupported(_) => ("unsupported".to_owned(), false),
        InventoryError::Io(_) => ("io_failed".to_owned(), true),
        InventoryError::Upstream(_) => ("upstream_failed".to_owned(), true),
        InventoryError::Integrity(_) => ("integrity_failed".to_owned(), false),
        InventoryError::ConcurrentMutation(_) => ("concurrent_mutation".to_owned(), true),
        InventoryError::ModelOperation {
            code, retryable, ..
        } => (code.clone(), *retryable),
        InventoryError::Internal(_) => ("internal".to_owned(), true),
    };
    ModelFailure {
        code,
        message: error.to_string(),
        retryable,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn performance_depths_filter_to_the_context_and_end_at_it() {
        assert_eq!(
            performance_depths(262_144),
            [25_000, 50_000, 75_000, 262_144]
        );
        assert_eq!(performance_depths(50_000), [25_000, 50_000]);
        assert_eq!(performance_depths(32_768), [25_000, 32_768]);
        assert_eq!(performance_depths(8_192), [8_192]);
    }

    #[test]
    fn desired_effective_and_discovered_targets_share_one_assessment_identity() {
        let environment = AssessmentEnvironmentId("environment".to_owned());
        let bundle = ServableModelBundleKey("bundle".to_owned());
        let profile = ModelAssessmentProfile {
            profile: magnitude_service_contracts::models::ServingProfile {
                context_length: 8_192,
            },
            performance_context_tokens: vec![8_192],
        };
        assert_eq!(
            AssessmentWorkKey::new(&environment, &bundle, &profile),
            AssessmentWorkKey::new(&environment, &bundle, &profile.clone()),
        );
        assert_ne!(
            AssessmentWorkKey::new(&environment, &bundle, &profile),
            AssessmentWorkKey::new(
                &AssessmentEnvironmentId("other".to_owned()),
                &bundle,
                &profile
            ),
        );
    }

    #[test]
    fn stable_artifact_rejection_is_scoped_to_one_assessment_target() {
        let failure = assessment_drop_failure(InventoryError::ModelOperation {
            code: "invalid_split_layout".to_owned(),
            message: "the shard layout is invalid".to_owned(),
            retryable: false,
        });
        assert_eq!(failure.code, "invalid_split_layout");
        assert!(!failure.retryable);

        let operational = assessment_drop_failure(InventoryError::ModelOperation {
            code: "assessment_failed".to_owned(),
            message: "the header could not be read".to_owned(),
            retryable: true,
        });
        assert_eq!(operational.code, "assessment_failed");
        assert!(operational.retryable);
    }

    #[test]
    fn engine_estimates_map_one_to_one() {
        let evidence = performance_evidence(&PerformanceEstimate {
            context_tokens: 50_000,
            lower_tokens_per_second: 9.0,
            estimated_tokens_per_second: 10.0,
            upper_tokens_per_second: 11.0,
            confidence: EngineConfidence::Moderate,
        });
        assert_eq!(evidence.context_tokens, 50_000);
        assert_eq!(evidence.lower_tokens_per_second, 9.0);
        assert_eq!(evidence.estimated_tokens_per_second, 10.0);
        assert_eq!(evidence.upper_tokens_per_second, 11.0);
        assert_eq!(evidence.confidence, PerformanceConfidence::Moderate);
    }
}
