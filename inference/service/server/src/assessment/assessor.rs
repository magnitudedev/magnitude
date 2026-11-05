//! Per-target model assessment: exact material resolution, the per-profile cache, deduplication
//! of concurrent equivalent work, and the engine's analytical assessment on the blocking pool.
//!
//! Assessment reads only GGUF headers (the release catalog's header bundle before download,
//! installed files after), opens no device and loads no model.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Weak};

use magnitude_engine::assessment::{
    DrafterVerdict as EngineDrafterVerdict, HeaderExtent as EngineHeaderExtent, Measurements,
    ModelAssessment as EngineAssessment, ModelAssessmentError,
    ModelCapabilities as EngineCapabilities, ModelExecution, ModelPackagePaths,
    ModelResidency as EngineResidency, Performance, PreparedModelAssessment,
    ProjectorVerdict as EngineProjectorVerdict, SpeculativeMethod as EngineSpeculativeMethod,
    StreamedPerformance, finish_model_assessment, header_refusal, prepare_model_assessment,
};
use magnitude_artifacts::gguf::Directory;
use magnitude_engine::error::{UnsupportedModel, UnsupportedReason as EngineUnsupportedReason};
use magnitude_engine::fingerprint::ModelFingerprint as EngineModelFingerprint;
use magnitude_engine::options::{ModelMethod, PROFILE_SEQUENCES};
use magnitude_executor::assessment::{DomainFit, ExecutionShortfall, PerformanceEstimate};
use magnitude_service_contracts::models::{
    AssessmentEnvironmentId, DrafterVerdict, InstalledModelPackages as _, KernelStatic,
    KnownUnsupportedReason, MemoryAssessment, ModelAssessment, ModelAssessmentId,
    ModelAssessmentProfile, ModelBundleInput, ModelCapabilities, ModelFailure, ModelFingerprint,
    ModelPackageOperand, ModelPerformance, ModelReasoningCapabilities, ModelResidency,
    ModelServingConfiguration, PerformanceEvidence, ProjectorVerdict, ResolvedServableModelBundle, ServableModelBundle,
    SpeculativeDraftSource, SpeculativeDraftSourceInput, SpeculativeMethod, StreamedModelPerformance,
    ColdExperts, StreamedBank, StreamedBankParts, UnsupportedReason, WeightTypeCount,
};
use magnitude_service_contracts::{ComponentRole, InventoryError, MemoryDomainId, ResolvedModel};
use magnitude_service_models::{
    CachedModelAssessment, HeaderExtent, HeaderVerdicts, ManagedModelStore, ModelDomainResolver,
    ReleaseCatalog, ServableModelBundleKey, servable_model_bundle_key_for_bundle,
};
use sha2::{Digest, Sha256};

use super::environment::AssessmentEnvironment;

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
    pub environment: Arc<AssessmentEnvironment>,
    pub profile: ModelAssessmentProfile,
    pub configuration: ModelServingConfiguration,
    pub preparation: Arc<tokio::sync::OnceCell<PreparationResult>>,
}

pub type PreparationResult = Result<Arc<PreparedModelAssessment>, ModelFailure>;

fn preparation_failure(error: ModelAssessmentError) -> ModelFailure {
    inventory_model_failure(preparation_error(error))
}

/// An engine failure that produced no verdict. Material that is truncated or not a GGUF is never
/// worth retrying; other failures may be.
pub(crate) fn preparation_error(error: ModelAssessmentError) -> InventoryError {
    let (code, retryable) = match &error {
        ModelAssessmentError::Artifact(artifact) if artifact.unreadable() => {
            ("unreadable_artifact", false)
        }
        _ => ("assessment_failed", true),
    };
    InventoryError::ModelOperation {
        code: code.to_owned(),
        message: error.to_string(),
        retryable,
    }
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

/// Assesses resolved targets in the current environment: one structure, under whatever the
/// measurement job has stored of the machine.
pub struct ModelAssessor {
    models: Arc<ManagedModelStore>,
    model_domains: Arc<ModelDomainResolver>,
    release_catalog: Arc<ReleaseCatalog>,
    environment: std::sync::RwLock<Arc<AssessmentEnvironment>>,
    work_gates: tokio::sync::Mutex<BTreeMap<String, Weak<tokio::sync::Mutex<()>>>>,
}

impl ModelAssessor {
    /// The assessor of targets in `environment`.
    pub fn new(
        models: Arc<ManagedModelStore>,
        model_domains: Arc<ModelDomainResolver>,
        release_catalog: Arc<ReleaseCatalog>,
        environment: Arc<AssessmentEnvironment>,
    ) -> Self {
        Self {
            models,
            model_domains,
            release_catalog,
            environment: std::sync::RwLock::new(environment),
            work_gates: tokio::sync::Mutex::new(BTreeMap::new()),
        }
    }

    /// The environment new work is assessed in.
    pub fn environment(&self) -> Arc<AssessmentEnvironment> {
        Arc::clone(
            &self
                .environment
                .read()
                .expect("assessment environment lock poisoned"),
        )
    }

    /// Assess new work under `measurements` of the same structure. The new environment when its
    /// identity differs from the current one's; `None` when the measurements change nothing of
    /// it, which then still read the models' miss records as they are stored now.
    pub fn adopt(&self, measurements: Measurements) -> Option<Arc<AssessmentEnvironment>> {
        let mut current = self
            .environment
            .write()
            .expect("assessment environment lock poisoned");
        let next = current.measured(measurements);
        let moved = next.id != current.id;
        // Of the same identity, the new measurements still read the miss records as they are
        // stored now (`Measurements::misses`).
        *current = Arc::new(next);
        moved.then(|| Arc::clone(&current))
    }

    pub fn models(&self) -> &Arc<ManagedModelStore> {
        &self.models
    }

    pub fn release_catalog(&self) -> &Arc<ReleaseCatalog> {
        &self.release_catalog
    }

    /// Resolve a bundle's exact material: release headers for a catalog bundle, installed files
    /// or source-backed previews for everything else.
    pub async fn resolve_material(
        &self,
        bundle: ServableModelBundle,
    ) -> Result<ResolvedServableModelBundle, InventoryError> {
        let bundle_key = servable_model_bundle_key_for_bundle(&bundle);
        self.resolve(&bundle_key, bundle).await
    }

    pub fn configuration_for(
        &self,
        subject: &magnitude_service_contracts::models::ModelAssessmentSubject,
    ) -> Result<ModelServingConfiguration, InventoryError> {
        self.model_domains.assessment_configuration(subject)
    }

    /// The work of assessing `configuration` at `profile` in `environment`.
    pub fn work_for(
        environment: &Arc<AssessmentEnvironment>,
        configuration: ModelServingConfiguration,
        profile: magnitude_service_contracts::models::ServingProfile,
        preparation: Arc<tokio::sync::OnceCell<PreparationResult>>,
    ) -> AssessmentWork {
        let profile = ModelAssessmentProfile {
            performance_context_tokens: performance_depths(profile.context_length),
            profile,
        };
        let bundle = servable_model_bundle_key_for_bundle(&configuration.bundle);
        AssessmentWork {
            key: AssessmentWorkKey::new(&environment.id, &bundle, &profile),
            environment: Arc::clone(environment),
            profile,
            configuration,
            preparation,
        }
    }

    /// Resolve and prepare one bundle from its headers. A preparation reads only the
    /// environment's structure, which every environment of this assessor shares.
    pub async fn prepare_bundle(
        &self,
        configuration: ModelServingConfiguration,
    ) -> PreparationResult {
        let structure = Arc::clone(&self.environment().setup.structure);
        let bundle_key = servable_model_bundle_key_for_bundle(&configuration.bundle);
        let resolved = self
            .resolve(&bundle_key, configuration.bundle)
            .await
            .map_err(assessment_drop_failure)?;
        let package = engine_material(&resolved).map_err(assessment_drop_failure)?;
        let prepared = magnitude_service_utils::blocking::run(move || {
            // The resolved material (a catalog bundle's materialized headers) lives until
            // preparation has read it.
            let _material = resolved;
            prepare_model_assessment(&package, &structure)
        })
        .await
        .map_err(preparation_failure)?;
        Ok(Arc::new(prepared))
    }

    /// One attempt at `work`. `Ok` carries the target's terminal disposition; `Err` is an
    /// operational failure the pool records as `Dropped`.
    pub async fn assess(&self, work: AssessmentWork) -> Result<AssessmentOutcome, InventoryError> {
        let AssessmentWork {
            key,
            environment,
            profile,
            configuration,
            preparation,
        } = work;
        let bundle_key = servable_model_bundle_key_for_bundle(&configuration.bundle);
        // The exact work identity is the assessment's identity but for a streamed fit's own miss
        // record, so it keys the cache.
        let evidence = key.0.clone();
        if let Some(cached) = self.cached(&environment, &evidence) {
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
        if let Some(cached) = self.cached(&environment, &evidence) {
            return Ok(AssessmentOutcome {
                key,
                result: Ok(cached),
            });
        }
        let prepared = match preparation
            .get_or_init(|| self.prepare_bundle(configuration))
            .await
        {
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

    /// The assessment cached under `evidence`, a key of `environment`, while it is the
    /// environment's assessment now ([`current`]).
    pub(crate) fn cached(
        &self,
        environment: &AssessmentEnvironment,
        evidence: &str,
    ) -> Option<CachedModelAssessment> {
        self.models
            .read_model_assessment(evidence)
            .filter(|cached| current(environment, cached))
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
            magnitude_service_utils::blocking::run(move || release_catalog.resolve_bundle(&release_key))
                .await?;
        match release {
            Some(resolved) => Ok(resolved),
            None => self.models.resolve_bundle(bundle_input(bundle)).await,
        }
    }
}

/// The components the engine reads: the target GGUF (a split target by its first shard, beside
/// which the engine finds the others), its projector and a separate draft, with the method the
/// bundle declares. A standalone package is served plain, even when its file carries MTP heads:
/// heads are used only through a bundle that declares them. A speculative bundle declares its
/// method, and the engine rejects a draft that does not implement it rather than serving the
/// target plain.
pub(crate) fn engine_material(
    resolved: &ResolvedServableModelBundle,
) -> Result<ModelPackagePaths, InventoryError> {
    let (method, draft) = match &resolved.bundle {
        ServableModelBundle::Standalone { .. } => (ModelMethod::Plain, None),
        ServableModelBundle::SpeculativeDecoding {
            draft_source,
            method,
            ..
        } => match (draft_source, method, &resolved.draft_model) {
            (SpeculativeDraftSource::Embedded, SpeculativeMethod::Mtp, None) => {
                (ModelMethod::Mtp, None)
            }
            (SpeculativeDraftSource::Separate { .. }, method, Some(draft)) => {
                let method = match method {
                    SpeculativeMethod::DFlash => ModelMethod::DFlash,
                    SpeculativeMethod::DFlash2 => ModelMethod::DFlash2,
                    SpeculativeMethod::DSpark => ModelMethod::DSpark,
                    SpeculativeMethod::LiLiCorr => ModelMethod::LiLiCorr,
                    SpeculativeMethod::Mtp => ModelMethod::Mtp,
                };
                (method, Some(draft_weights(draft)?))
            }
            (SpeculativeDraftSource::Embedded, _, _) => {
                return Err(InventoryError::InvalidRequest(
                    "an embedded draft is an MTP head; DFlash, DFlash2, DSpark and LiLiCorr need a separate draft"
                        .to_owned(),
                ));
            }
            (SpeculativeDraftSource::Separate { .. }, _, None) => {
                return Err(InventoryError::Integrity(
                    "a separate-draft bundle resolved without its draft".to_owned(),
                ));
            }
        },
    };
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
    Ok(ModelPackagePaths {
        target,
        projector,
        draft,
        method,
    })
}

/// A separate draft package's one GGUF.
fn draft_weights(draft: &ResolvedModel) -> Result<PathBuf, InventoryError> {
    let weights = draft
        .components
        .iter()
        .filter(|component| {
            matches!(
                component.role,
                ComponentRole::Weights | ComponentRole::Draft
            )
        })
        .map(|component| component.path.clone())
        .collect::<Vec<_>>();
    match weights.as_slice() {
        [path] => Ok(path.clone()),
        [] => Err(InventoryError::NotReady(
            "the draft has no runnable weights".to_owned(),
        )),
        _ => Err(InventoryError::InvalidRequest(
            "the draft resolves more than one weights file".to_owned(),
        )),
    }
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
    let assessed = magnitude_service_utils::blocking::run(move || {
        let setup = &engine_environment.setup;
        finish_model_assessment(&prepared, setup, &depths, PROFILE_SEQUENCES)
    })
    .await;
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
pub(crate) fn model_assessment(
    assessed: EngineAssessment,
    bundle_key: &ServableModelBundleKey,
    requested: ModelAssessmentProfile,
    environment: &AssessmentEnvironment,
) -> Result<CachedModelAssessment, InventoryError> {
    let (facts, execution) = match assessed {
        EngineAssessment::Unsupported {
            unsupported,
            fingerprint,
        } => {
            return Ok(unsupported_model(
                requested,
                unsupported,
                fingerprint.as_ref().map(model_fingerprint),
            ));
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
    let projector = projector_verdict(facts.projector);
    let drafter = drafter_verdict(facts.drafter);
    let misses = match &execution {
        ModelExecution::Fits {
            residency: EngineResidency::Streamed { misses, .. },
            ..
        } => Some(misses.0.clone()),
        _ => None,
    };
    let assessment_id = assessment_id(bundle_key, &requested, &environment.id, misses.as_deref());
    let ModelAssessmentProfile {
        profile,
        performance_context_tokens,
    } = requested;
    let assessment = match execution {
        ModelExecution::Fits { domains, residency } => ModelAssessment::Fits {
            profile,
            assessment_id,
            memory: memory_assessments(&domains, environment),
            residency: model_residency(residency, &performance_context_tokens)?,
            projector,
            drafter,
        },
        ModelExecution::DoesNotFit(ExecutionShortfall {
            domains,
            limiting,
            deficit_bytes,
            ..
        }) => ModelAssessment::DoesNotFit {
            profile,
            assessment_id,
            memory: memory_assessments(&domains, environment),
            limiting_resource: domain_id(limiting, environment).as_str().to_owned(),
            deficit_bytes: deficit_bytes.get(),
            projector,
            drafter,
        },
    };
    Ok(CachedModelAssessment {
        capabilities: capabilities(facts.capabilities),
        template_fingerprint: facts.template_fingerprint,
        fingerprint: facts.fingerprint.as_ref().map(model_fingerprint),
        profile: assessment,
        misses,
    })
}

/// A package the engine cannot execute has no engine-derived capabilities or template.
fn unsupported_reason(reason: EngineUnsupportedReason) -> UnsupportedReason {
    UnsupportedReason::Known(match reason {
        EngineUnsupportedReason::Architecture { architecture } => {
            KnownUnsupportedReason::Architecture { architecture }
        }
        EngineUnsupportedReason::WeightTypes { types } => KnownUnsupportedReason::WeightTypes {
            types: types
                .into_iter()
                .map(|count| WeightTypeCount {
                    ggml_type: count.ggml_type,
                    tensor_count: count.tensor_count,
                })
                .collect(),
        },
        EngineUnsupportedReason::RopeScaling { kind } => {
            KnownUnsupportedReason::RopeScaling { kind }
        }
        EngineUnsupportedReason::UnrecognizedTensor { name } => {
            KnownUnsupportedReason::UnrecognizedTensor { name }
        }
        EngineUnsupportedReason::UnrecognizedMetadata { key } => {
            KnownUnsupportedReason::UnrecognizedMetadata { key }
        }
        EngineUnsupportedReason::Tokenizer { model, pre } => {
            KnownUnsupportedReason::Tokenizer { model, pre }
        }
        EngineUnsupportedReason::MissingMetadata { key } => {
            KnownUnsupportedReason::MissingMetadata { key }
        }
        EngineUnsupportedReason::ChatTemplate => KnownUnsupportedReason::ChatTemplate,
        EngineUnsupportedReason::KernelDomain {
            backend,
            entry,
            statics,
        } => KnownUnsupportedReason::KernelDomain {
            backend,
            entry,
            statics: statics
                .into_iter()
                .map(|(name, value)| KernelStatic { name, value })
                .collect(),
        },
        EngineUnsupportedReason::Backend { backend } => KnownUnsupportedReason::Backend { backend },
        EngineUnsupportedReason::Unreadable => KnownUnsupportedReason::Unreadable,
        EngineUnsupportedReason::TargetMismatch { detail } => {
            KnownUnsupportedReason::TargetMismatch { detail }
        }
        EngineUnsupportedReason::TiedOutputHead => KnownUnsupportedReason::TiedOutputHead,
        EngineUnsupportedReason::DraftArchitecture { architecture } => {
            KnownUnsupportedReason::DraftArchitecture { architecture }
        }
    })
}

/// The engine's refusal of a model file's header alone, which search shows on a result: the
/// header interpretation assessment admits a target by, without weight types or this machine.
pub struct EngineHeaderVerdicts;

impl HeaderVerdicts for EngineHeaderVerdicts {
    fn identity(&self) -> String {
        crate::build_identity::native_build()
    }

    fn refusal(&self, header: &Directory, extent: HeaderExtent) -> Vec<UnsupportedReason> {
        let extent = match extent {
            HeaderExtent::Whole => EngineHeaderExtent::Whole,
            HeaderExtent::Leading => EngineHeaderExtent::Leading,
        };
        header_refusal(header, extent)
            .map(|unsupported| unsupported_reasons(unsupported).0)
            .unwrap_or_default()
    }
}

fn unsupported_reasons(unsupported: UnsupportedModel) -> (Vec<UnsupportedReason>, String) {
    (
        unsupported
            .reasons
            .into_iter()
            .map(unsupported_reason)
            .collect(),
        unsupported.detail,
    )
}

fn projector_verdict(verdict: EngineProjectorVerdict) -> ProjectorVerdict {
    match verdict {
        EngineProjectorVerdict::None => ProjectorVerdict::None,
        EngineProjectorVerdict::Bound => ProjectorVerdict::Bound,
        EngineProjectorVerdict::Rejected(unsupported) => {
            let (reasons, detail) = unsupported_reasons(unsupported);
            ProjectorVerdict::Rejected { reasons, detail }
        }
    }
}

fn drafter_verdict(verdict: EngineDrafterVerdict) -> DrafterVerdict {
    match verdict {
        EngineDrafterVerdict::None => DrafterVerdict::None,
        EngineDrafterVerdict::Bound {
            method,
            memory_bytes,
        } => DrafterVerdict::Bound {
            method: match method {
                EngineSpeculativeMethod::Mtp => SpeculativeMethod::Mtp,
                EngineSpeculativeMethod::DFlash => SpeculativeMethod::DFlash,
                EngineSpeculativeMethod::DFlash2 => SpeculativeMethod::DFlash2,
                EngineSpeculativeMethod::DSpark => SpeculativeMethod::DSpark,
                EngineSpeculativeMethod::LiLiCorr => SpeculativeMethod::LiLiCorr,
            },
            memory_bytes,
        },
        EngineDrafterVerdict::Rejected(unsupported) => {
            let (reasons, detail) = unsupported_reasons(unsupported);
            DrafterVerdict::Rejected { reasons, detail }
        }
    }
}

/// The two forms of a fingerprint, which share every field: each direction's conversion, made
/// from the one list of their fields. Neither form's crate sees the other, so the conversions are
/// here; each destructures its source and builds its target whole, so a field either form gains
/// or loses fails to compile until the list follows.
macro_rules! fingerprint_conversions {
    ($($field:ident),* $(,)?) => {
        /// The contract's form of the engine's fingerprint.
        pub(crate) fn model_fingerprint(fingerprint: &EngineModelFingerprint) -> ModelFingerprint {
            let EngineModelFingerprint { $($field),* } = fingerprint;
            ModelFingerprint { $($field: $field.clone()),* }
        }

        /// The engine's form of a fingerprint on the contract: what the engine finds the
        /// model's miss record by.
        fn engine_fingerprint(fingerprint: &ModelFingerprint) -> EngineModelFingerprint {
            let ModelFingerprint { $($field),* } = fingerprint;
            EngineModelFingerprint { $($field: $field.clone()),* }
        }
    };
}

fingerprint_conversions!(
    architecture,
    block_count,
    embedding_length,
    head_count,
    head_count_kv,
    key_length,
    vocab_size,
    expert_count,
    tokenizer_digest,
);

/// Whether `cached` is `environment`'s assessment now, read under a key of it: one that reads
/// no miss record, or a streamed fit whose estimates read its own model's record as it is stored
/// now. No other model's record enters it.
fn current(environment: &AssessmentEnvironment, cached: &CachedModelAssessment) -> bool {
    cached.misses.as_ref().is_none_or(|misses| {
        let fingerprint = cached.fingerprint.as_ref().map(engine_fingerprint);
        environment.setup.measurements.misses(fingerprint.as_ref()).1.0 == *misses
    })
}

fn unsupported_model(
    requested: ModelAssessmentProfile,
    unsupported: UnsupportedModel,
    fingerprint: Option<ModelFingerprint>,
) -> CachedModelAssessment {
    let (reasons, detail) = unsupported_reasons(unsupported);
    // A file the engine refused before establishing a context (and with no training context in its
    // header) has no serving profile, never a zero-token one.
    let profile = Some(requested.profile).filter(|profile| profile.context_length > 0);
    unsupported_assessment(profile, reasons, detail, fingerprint)
}

/// A file whose header or bytes cannot be read: refused before the engine sees it, without a
/// serving profile.
pub(crate) fn unreadable_assessment(detail: String) -> CachedModelAssessment {
    unsupported_assessment(
        None,
        vec![UnsupportedReason::Known(KnownUnsupportedReason::Unreadable)],
        detail,
        None,
    )
}

fn unsupported_assessment(
    profile: Option<magnitude_service_contracts::models::ServingProfile>,
    reasons: Vec<UnsupportedReason>,
    detail: String,
    fingerprint: Option<ModelFingerprint>,
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
        fingerprint,
        profile: ModelAssessment::Unsupported {
            profile,
            reasons,
            detail,
        },
        misses: None,
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

fn domain_id(domain: seismic::StoreId, environment: &AssessmentEnvironment) -> MemoryDomainId {
    crate::memory_domains::store_domain_id(&environment.setup.structure.topology, domain)
}

/// How a fitting model holds its weights on the contract, with its speed held so.
fn model_residency(
    residency: EngineResidency,
    depths: &[u32],
) -> Result<ModelResidency, InventoryError> {
    Ok(match residency {
        EngineResidency::Resident { performance } => ModelResidency::Resident {
            performance: model_performance(performance, depths)?,
        },
        EngineResidency::Streamed {
            shares,
            performance,
            floor_performance,
            ..
        } => ModelResidency::Streamed {
            bank: streamed_bank(shares),
            performance: streamed_performance(performance, depths)?,
            floor_performance: streamed_performance(floor_performance, depths)?,
        },
    })
}

/// A streamed fit's bank on the contract, whose bounds its shares uphold.
fn streamed_bank(shares: magnitude_executor::assessment::ExpertShares) -> StreamedBank {
    StreamedBankParts {
        experts: shares.experts(),
        slots: shares.slots(),
        floor_slots: shares.floor(),
        cold_experts: match shares.cold() {
            magnitude_executor::assessment::ColdSource::HostTier { .. } => ColdExperts::HostRam,
            magnitude_executor::assessment::ColdSource::File(_) => ColdExperts::Disk,
        },
    }
    .try_into()
    .expect("expert shares uphold a streamed bank's bounds")
}

/// A streamed fit's speed on the contract: unmeasured, or as [`model_performance`] gives it.
fn streamed_performance(
    performance: StreamedPerformance,
    depths: &[u32],
) -> Result<StreamedModelPerformance, InventoryError> {
    Ok(match performance {
        StreamedPerformance::Unmeasured => StreamedModelPerformance::Unmeasured,
        StreamedPerformance::Assessed(performance) => model_performance(performance, depths)?.into(),
    })
}

/// A fitting model's speed on the contract: estimated at exactly the requested depths, pending,
/// or unavailable with why.
fn model_performance(
    performance: Performance,
    depths: &[u32],
) -> Result<ModelPerformance, InventoryError> {
    Ok(match performance {
        Performance::Pending => ModelPerformance::Pending,
        Performance::Unavailable { reason } => ModelPerformance::Unavailable {
            failure: ModelFailure {
                code: "performance_unavailable".to_owned(),
                message: reason,
                retryable: false,
            },
        },
        Performance::Estimated(estimates) => {
            let samples = estimates
                .iter()
                .map(performance_evidence)
                .collect::<Vec<_>>();
            if samples
                .iter()
                .map(|sample| sample.context_tokens)
                .ne(depths.iter().copied())
            {
                return Err(InventoryError::ModelOperation {
                    code: "assessment_incomplete".to_owned(),
                    message: "the engine did not estimate every requested depth".to_owned(),
                    retryable: false,
                });
            }
            ModelPerformance::Estimated { samples }
        }
    })
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
        estimated_tokens_per_second: estimate.tokens_per_second,
    }
}

fn assessment_id(
    bundle_key: &ServableModelBundleKey,
    profile: &ModelAssessmentProfile,
    environment: &AssessmentEnvironmentId,
    misses: Option<&str>,
) -> ModelAssessmentId {
    let mut digest = Sha256::new();
    digest.update(bundle_key.0.as_bytes());
    digest.update(profile.profile.context_length.to_le_bytes());
    digest.update(environment.0.as_bytes());
    for context_tokens in &profile.performance_context_tokens {
        digest.update(context_tokens.to_le_bytes());
    }
    // A streamed fit's own miss record, which its estimates are from.
    if let Some(misses) = misses {
        digest.update(misses.as_bytes());
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
        InventoryError::DoesNotFit(_) => ("model_does_not_fit".to_owned(), false),
        InventoryError::ModelUnsupported { .. } => ("model_unsupported".to_owned(), false),
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
    use magnitude_engine::assessment::MissesDigest;

    /// An unsupported file whose context is unknown has no serving profile, never a zero-token one;
    /// one with a context keeps it.
    #[test]
    fn an_unsupported_model_never_has_a_zero_context_profile() {
        let profile = |context_length| ModelAssessmentProfile {
            profile: magnitude_service_contracts::models::ServingProfile { context_length },
            performance_context_tokens: Vec::new(),
        };
        let unsupported = || {
            UnsupportedModel::reason(
                EngineUnsupportedReason::Architecture {
                    architecture: "deepseek4".to_owned(),
                },
                "no registered model family",
            )
        };
        let profile_of = |assessment: CachedModelAssessment| match assessment.profile {
            ModelAssessment::Unsupported { profile, .. } => profile,
            other => panic!("not unsupported: {other:?}"),
        };
        assert_eq!(
            profile_of(unsupported_model(profile(0), unsupported(), None)),
            None
        );
        assert_eq!(
            profile_of(unsupported_model(profile(4096), unsupported(), None)),
            Some(magnitude_service_contracts::models::ServingProfile {
                context_length: 4096
            })
        );
        assert_eq!(
            profile_of(unreadable_assessment("truncated".to_owned())),
            None
        );
    }

    /// A readable header that omits a required key names it rather than reading as unreadable.
    #[test]
    fn a_missing_key_keeps_its_name_in_the_contract() {
        let key = "nemotron_h_moe.attention.layer_norm_epsilon";
        let (reasons, detail) = unsupported_reasons(UnsupportedModel::reason(
            EngineUnsupportedReason::MissingMetadata {
                key: key.to_owned(),
            },
            format!("missing metadata {key}"),
        ));
        assert_eq!(
            reasons,
            [UnsupportedReason::Known(
                KnownUnsupportedReason::MissingMetadata {
                    key: key.to_owned()
                }
            )]
        );
        assert_eq!(detail, format!("missing metadata {key}"));
    }

    fn resolved_model(path: &str) -> ResolvedModel {
        use magnitude_service_contracts::{
            ContentId, InventoryEntryId, InventoryModel, InventoryProperties, ModelAvailability,
            ModelLocation, ModelSource, ResolvedComponent,
        };
        ResolvedModel {
            model: InventoryModel {
                id: InventoryEntryId(path.to_owned()),
                content_id: ContentId(path.to_owned()),
                created: 0,
                name: path.to_owned(),
                supported_parameters: Vec::new(),
                availability: ModelAvailability::Available { ready_at: 0 },
                source: ModelSource::Local {
                    declared_by: magnitude_service_contracts::LocalDeclaration::Discovery,
                },
                location: ModelLocation::Directory {
                    source_id: "source".to_owned(),
                    root: PathBuf::from("/models"),
                    components: Vec::new(),
                    total_bytes: 1,
                    integrity: magnitude_service_contracts::Integrity::Unverified {
                        reason: "test".to_owned(),
                    },
                },
                properties: InventoryProperties::Inspected {
                    architecture: None,
                    quantization: None,
                    quantization_name: None,
                    parameter_count: None,
                    active_parameter_count: None,
                    training_context_length: None,
                    nextn_predict_layers: None,
                    tokenizer: None,
                    base_models: Vec::new(),
                    evidence_fingerprint: "test".to_owned(),
                },
                operations: Vec::new(),
                updated_at: 0,
            },
            components: vec![ResolvedComponent {
                path: PathBuf::from("/models").join(path),
                role: ComponentRole::Weights,
                shard_index: None,
                relationship: None,
            }],
        }
    }

    fn package(id: &str) -> magnitude_service_contracts::models::ModelPackage {
        use magnitude_service_contracts::models::{
            ModelFile, ModelFileId, ModelFileRole, ModelPackage, ModelPackageId,
            ModelPackageProperties, ModelPackageSource,
        };
        ModelPackage {
            id: ModelPackageId(id.to_owned()),
            source: ModelPackageSource::Local {
                path: PathBuf::from("/models"),
            },
            files: vec![ModelFile {
                id: ModelFileId(id.to_owned()),
                path: PathBuf::from(format!("{id}.gguf")),
                role: ModelFileRole::Weights,
                size_bytes: 1,
                tensor_storage_bytes: None,
                sha256: "a".repeat(64),
            }],
            relationships: Vec::new(),
            properties: ModelPackageProperties {
                format: "gguf".to_owned(),
                quantization: "unknown".to_owned(),
                quantization_name: "unknown".to_owned(),
                architecture: "test".to_owned(),
                maximum_context_length: None,
                intrinsic_model_id: None,
                intrinsic_quality_id: None,
            },
        }
    }

    fn speculative(draft: bool, method: SpeculativeMethod) -> ResolvedServableModelBundle {
        ResolvedServableModelBundle::new(
            ServableModelBundle::SpeculativeDecoding {
                target: package("target"),
                draft_source: if draft {
                    SpeculativeDraftSource::Separate {
                        draft: package("draft"),
                    }
                } else {
                    SpeculativeDraftSource::Embedded
                },
                method,
            },
            resolved_model("target.gguf"),
            draft.then(|| resolved_model("draft.gguf")),
        )
    }

    /// The engine reads a separate-draft bundle's draft with the bundle's
    /// declared method; a method its draft source cannot carry is refused.
    #[test]
    fn engine_material_carries_the_declared_draft_and_method() {
        for (method, expected) in [
            (SpeculativeMethod::DFlash, ModelMethod::DFlash),
            (SpeculativeMethod::DFlash2, ModelMethod::DFlash2),
            (SpeculativeMethod::DSpark, ModelMethod::DSpark),
            (SpeculativeMethod::LiLiCorr, ModelMethod::LiLiCorr),
            (SpeculativeMethod::Mtp, ModelMethod::Mtp),
        ] {
            let material = engine_material(&speculative(true, method)).unwrap();
            assert_eq!(material.target, PathBuf::from("/models/target.gguf"));
            assert_eq!(material.draft, Some(PathBuf::from("/models/draft.gguf")));
            assert_eq!(material.method, expected);
        }
        let standalone = engine_material(&ResolvedServableModelBundle::new(
            ServableModelBundle::Standalone {
                package: package("target"),
            },
            resolved_model("target.gguf"),
            None,
        ))
        .unwrap();
        assert_eq!(
            (standalone.draft, standalone.method),
            (None, ModelMethod::Plain),
            "a standalone package never drafts with heads its file carries"
        );
        let embedded = engine_material(&speculative(false, SpeculativeMethod::Mtp)).unwrap();
        assert_eq!((embedded.draft, embedded.method), (None, ModelMethod::Mtp));
        for (draft, method) in [
            (false, SpeculativeMethod::DFlash),
            (false, SpeculativeMethod::DFlash2),
            (false, SpeculativeMethod::DSpark),
            (false, SpeculativeMethod::LiLiCorr),
        ] {
            assert!(matches!(
                engine_material(&speculative(draft, method)),
                Err(InventoryError::InvalidRequest(_))
            ));
        }
    }

    /// A streamed fit reaches the contract with its bank's shares of each routed layer's experts
    /// and its speed unmeasured; a resident fit keeps its speed.
    #[test]
    fn a_streamed_fit_reaches_the_contract_with_its_shares() {
        let shares = magnitude_executor::assessment::ExpertShares::new(
            64,
            192,
            std::num::NonZeroU64::new(256).unwrap(),
            magnitude_executor::assessment::ColdSource::HostTier { slots: 256 },
        );
        assert_eq!(
            model_residency(
                EngineResidency::Streamed {
                    shares,
                    performance: StreamedPerformance::Unmeasured,
                    floor_performance: StreamedPerformance::Assessed(Performance::Pending),
                    misses: MissesDigest::of(&Ok(None)),
                },
                &[25_000],
            )
            .unwrap(),
            ModelResidency::Streamed {
                bank: StreamedBankParts {
                    experts: std::num::NonZeroU64::new(256).unwrap(),
                    slots: 192,
                    floor_slots: 64,
                    cold_experts: ColdExperts::HostRam,
                }
                .try_into()
                .unwrap(),
                performance: StreamedModelPerformance::Unmeasured,
                floor_performance: StreamedModelPerformance::Pending,
            }
        );
        assert_eq!(
            model_residency(
                EngineResidency::Resident {
                    performance: Performance::Pending,
                },
                &[25_000],
            )
            .unwrap(),
            ModelResidency::Resident {
                performance: ModelPerformance::Pending,
            }
        );
    }

    /// A streamed load's new miss record renews only its own model's assessment: the
    /// environment's identity holds no record, and another model's cached streamed fit stays
    /// current.
    #[test]
    fn one_models_new_miss_record_leaves_every_other_assessment_cached() {
        use magnitude_executor::KernelCache;
        use magnitude_executor::assessment::{
            ExecutorCosts, MissRecord, RequestReads, Residuals, StoredCosts, StoredMisses,
        };
        let kernels = tempfile::tempdir().unwrap();
        let structure = Arc::new(
            super::super::environment::assessment_structure(
                &seismic::Devices::discover(&[]).expect("device catalog"),
                super::super::environment::ServingDevice::of(seismic::ExecutorSelector::HostCpu),
                None,
            )
            .expect("the host CPU is usable"),
        );
        // A record of each cost point the walk reads until the costs derive: the probe's first.
        let records = KernelCache::measurement_records(kernels.path());
        while let StoredCosts::Missing(key) =
            ExecutorCosts::stored(&structure.topology, &structure.selected.info, &records).unwrap()
        {
            let timing = seismic::measure::Timing {
                median: 1e-3,
                interval: None,
                samples: 8,
            };
            seismic::measure::RecordStore::store(&records, &key, &seismic::measure::Record::Measured(timing))
                .unwrap();
        }
        let environment = AssessmentEnvironment::new(
            Arc::clone(&structure),
            structure.stored_measurements(kernels.path()),
        )
        .unwrap();
        assert!(matches!(
            environment.setup.measurements,
            Measurements::Measured { .. }
        ));
        let fingerprint = |architecture: &str| EngineModelFingerprint {
            architecture: architecture.into(),
            block_count: 2,
            embedding_length: 64,
            head_count: 4,
            head_count_kv: 2,
            key_length: 16,
            vocab_size: 100,
            expert_count: Some(4),
            tokenizer_digest: "digest".into(),
        };
        let (own, other) = (fingerprint("own"), fingerprint("other"));
        // Whether an assessment is current reads only its fingerprint and its record's digest.
        let streamed = |fingerprint: &EngineModelFingerprint| CachedModelAssessment {
            misses: Some(
                environment.setup.measurements.misses(Some(fingerprint)).1.0,
            ),
            ..unsupported_assessment(
                None,
                Vec::new(),
                String::new(),
                Some(model_fingerprint(fingerprint)),
            )
        };
        let (own_cached, other_cached) = (streamed(&own), streamed(&other));
        assert!(current(&environment, &own_cached));
        assert!(current(&environment, &other_cached));

        // A streamed load of `own` stores its record.
        let executor = structure.selected.identity.clone();
        let model = serde_json::to_value(&own).unwrap();
        let stored = StoredMisses {
            executor: executor.clone(),
            model: model.clone(),
            record: MissRecord {
                steps: 1,
                rows: 1,
                layers: Vec::new(),
                reads: RequestReads::default(),
                residuals: Residuals::default(),
                hints: Default::default(),
            },
        };
        KernelCache::miss_records(kernels.path())
            .store(&StoredMisses::key(&executor, &model), &stored, u64::MAX)
            .unwrap();
        let measured = environment.measured(structure.stored_measurements(kernels.path()));
        assert_eq!(measured.id, environment.id, "no record enters the environment's identity");
        assert!(!current(&measured, &own_cached), "its own record renews the model's assessment");
        assert!(current(&measured, &other_cached), "another model's assessment stays cached");
    }

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
    fn unreadable_material_is_not_retried() {
        let unreadable = preparation_error(ModelAssessmentError::Artifact(
            magnitude_artifacts::Error::Invalid("truncated header".to_owned()),
        ));
        assert!(matches!(
            unreadable,
            InventoryError::ModelOperation { retryable: false, ref code, .. } if code == "unreadable_artifact"
        ));
        let operational = preparation_error(ModelAssessmentError::Planning("busy".to_owned()));
        assert!(matches!(
            operational,
            InventoryError::ModelOperation {
                retryable: true,
                ..
            }
        ));
    }

    #[test]
    fn engine_estimates_map_one_to_one() {
        let evidence = performance_evidence(&PerformanceEstimate {
            context_tokens: 50_000,
            tokens_per_second: 10.0,
        });
        assert_eq!(evidence.context_tokens, 50_000);
        assert_eq!(evidence.estimated_tokens_per_second, 10.0);
    }
}
