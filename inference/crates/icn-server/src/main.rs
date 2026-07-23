use std::net::SocketAddr;
use std::num::NonZeroU32;
use std::path::PathBuf;
#[cfg(not(test))]
use std::process::{Command as ProcessCommand, Stdio};
use std::sync::{Arc, RwLock};

use anyhow::Context;
use clap::{Parser, Subcommand};
use futures_util::{StreamExt, future::BoxFuture, stream::BoxStream};
use icn_api::{
    AppState, BackendMutationGuard, BackendRegistry, FakeBackend, RuntimeController,
    ServerIdentity, app,
};
use icn_contracts::models::{
    AssessModelResult, AssessModelsRequest, AssessModelsResponse, AssessmentEnvironmentId,
    FitModelResult, FitModelsRequest, FitModelsResponse, InstalledModelPackages as _,
    LoadModelReady, LoadModelRequest, MemoryAssessment, ModelEvaluator,
    ModelFailure as DomainModelFailure, ModelLoadEvent, ModelLoadStage,
    ModelOfferingTarget as DomainModelOfferingTarget, ModelPackageId, ModelPackageOperand,
    ModelServingConfiguration, ModelServingConfigurationId, ModelTargetInput, OfferingAssessment,
    OfferingAssessmentId, PerformanceConfidence, PerformanceEvidence,
    RemoveInstalledModelPackageResponse, RuntimeResidencyId,
    ServingProfile as DomainServingProfile,
};
use icn_contracts::{
    CacheType, CompletionBackend, ComponentRole, ExecutionConfig, ExecutionIntent, FlashAttention,
    GenerationPerformanceAssessment, GpuLayers, HardwareAssessment, HardwareProvider,
    HardwareSnapshot, InventoryError, InventoryHardwareAssessor, ModelExecutionAssessment,
    ModelHardwareAssessor, ModelPreviewProfile, ProjectorConfig, ResolvedModel, SplitMode,
    TemplateAssessment, TemplateAssessor,
};
use icn_engine::{LlamaCompletionBackend, ModelLoadError, MtpCandidateSelection, NativeBackend};
use icn_hardware::CapacityPolicy;
use icn_models::{
    InventoryConfig, ManagedModelDownloads, ModelManager, ModelPreviewService,
    NativeRecommendableCatalog,
};
use sha2::{Digest, Sha256};
use tokio_stream::wrappers::UnboundedReceiverStream;
use tower_http::trace::{DefaultOnResponse, TraceLayer};

mod build_identity;
mod telemetry;

#[derive(Debug, Parser)]
#[command(
    name = "magnitude-icn",
    version,
    about = "Magnitude inference control node"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
// Clap's flat `serve` command intentionally keeps its complete execution profile visible in
// `--help`; boxing individual flags would only optimize the one-time CLI parse allocation.
#[allow(clippy::large_enum_variant)]
enum Command {
    Serve {
        #[arg(long, default_value = "127.0.0.1:8080")]
        bind: SocketAddr,
        /// Opaque owner-provided identity echoed by the startup and health protocols.
        #[arg(long, default_value = "standalone")]
        instance_id: String,
        /// Owning process. ICN exits if this process disappears.
        #[arg(long)]
        parent_pid: Option<u32>,
        /// Private owner capability. Prefer the environment-backed form used by managed launch.
        #[arg(long, env = "MAGNITUDE_ICN_AUTH_TOKEN", hide_env_values = true)]
        auth_token: Option<String>,
        /// Deterministic in-memory backend used only by protocol tests.
        #[arg(long)]
        fake: bool,
        /// Magnitude-owned model inventory and Hugging Face cache root.
        #[arg(long, visible_alias = "models-dir")]
        model_store: Option<PathBuf>,
        /// Additional read-only directories containing GGUF models.
        #[arg(long = "model-source")]
        model_sources: Vec<PathBuf>,
        /// Additional read-only Hugging Face hub cache roots.
        #[arg(long = "hf-cache", visible_alias = "hf-cache-dir")]
        hf_caches: Vec<PathBuf>,
    },
    Doctor,
    Version {
        #[arg(long)]
        json: bool,
    },
    #[command(hide = true)]
    PlanWorker,
    #[command(hide = true)]
    TemplateWorker,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
struct RuntimePlanDefaults {
    context_size: u32,
    batch_size: u32,
    ubatch_size: u32,
    max_sequences: u32,
    prefill_quantum: u32,
    execution: ExecutionConfig,
    projector_use_gpu: bool,
    projector_warmup: bool,
    image_min_tokens: Option<NonZeroU32>,
    image_max_tokens: Option<NonZeroU32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RuntimeExecutionProfile {
    context_length: u32,
    parallel_sequences: u32,
}

#[derive(Debug, Clone)]
struct ResidentTarget {
    model_id: String,
    residency_id: String,
    profile: RuntimeExecutionProfile,
    package_ids: Vec<ModelPackageId>,
}

#[derive(Debug, Clone)]
struct RuntimeState {
    resident: Option<ResidentTarget>,
}

fn runtime_plan_defaults() -> RuntimePlanDefaults {
    RuntimePlanDefaults {
        // Managed product models always overwrite context and parallelism from their persisted
        // serving configuration. This conservative value is only the discovery/migration fallback
        // for unmanaged local artifacts that predate serving configurations.
        context_size: 4096,
        batch_size: 512,
        ubatch_size: 512,
        max_sequences: 1,
        prefill_quantum: 512,
        execution: ExecutionConfig {
            gpu_layers: GpuLayers::Auto,
            use_mmap: true,
            use_mlock: false,
            split_mode: SplitMode::Layer,
            tensor_split: None,
            cache_type_k: CacheType::F16,
            cache_type_v: CacheType::F16,
            offload_kqv: true,
            operation_offload: true,
            swa_full: false,
            kv_unified: true,
            threads: None,
            threads_batch: None,
            flash_attention: FlashAttention::Auto,
        },
        projector_use_gpu: true,
        projector_warmup: true,
        image_min_tokens: None,
        image_max_tokens: None,
    }
}

fn execution_intent(
    model_path: PathBuf,
    projector_path: Option<PathBuf>,
    defaults: &RuntimePlanDefaults,
) -> anyhow::Result<ExecutionIntent> {
    Ok(ExecutionIntent {
        model_path,
        context_size: defaults.context_size,
        batch_size: defaults.batch_size,
        ubatch_size: defaults.ubatch_size,
        max_sequences: defaults.max_sequences,
        prefill_quantum: defaults.prefill_quantum,
        execution: defaults.execution.clone(),
        projector: projector_path.map(|path| {
            let mut projector = ProjectorConfig::new(path);
            projector.use_gpu = defaults.projector_use_gpu;
            projector.warmup = defaults.projector_warmup;
            projector.image_min_tokens = defaults.image_min_tokens;
            projector.image_max_tokens = defaults.image_max_tokens;
            projector
        }),
        mtp: icn_contracts::MtpConfig::default(),
    })
}

#[derive(Clone)]
struct ResidentNativeExecutor {
    generation: u64,
    model_id: String,
    backend: Arc<LlamaCompletionBackend>,
}

struct NativeHardwareAssessor {
    defaults: RuntimePlanDefaults,
    native_backend: NativeBackend,
    native_executor: Arc<RwLock<Option<ResidentNativeExecutor>>>,
    gate: tokio::sync::Mutex<()>,
    planning_slots: Arc<tokio::sync::Semaphore>,
    calibration: tokio::sync::Mutex<CalibrationCache>,
}

#[derive(Default)]
struct CalibrationCache {
    topology_fingerprint: Option<String>,
    result: Option<Result<llama_cpp_2::model::params::fit::FitCalibration, String>>,
}

const ASSESSMENT_RESOLVER_REVISION: &str = "icn-backend-plan-v1";
const CAPACITY_POLICY_REVISION: &str = "stable-total-reserve-v1";
const MTP_SELECTOR_REVISION: &str = "icn-mtp-selector-v1";
const MODEL_ASSESSMENT_CONCURRENCY: usize = 12;

#[derive(Debug, serde::Deserialize, serde::Serialize)]
struct PlanningWorkerRequest {
    primary: PathBuf,
    projector: Option<PathBuf>,
    mtp: Vec<PathBuf>,
    defaults: Vec<RuntimePlanDefaults>,
    estimate_performance: bool,
    capacity_policy: CapacityPolicy,
    calibration: Option<llama_cpp_2::model::params::fit::FitCalibration>,
    calibration_unavailable: Option<String>,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
struct PlanningWorkerResponse {
    assessments: Vec<ModelExecutionAssessment>,
    calibration: Option<Result<llama_cpp_2::model::params::fit::FitCalibration, String>>,
}

#[derive(Debug, serde::Deserialize, serde::Serialize)]
struct TemplateWorkerRequest {
    model_path: PathBuf,
}

#[derive(Debug, Default)]
struct NativeTemplateAssessor;

impl TemplateAssessor for NativeTemplateAssessor {
    fn cache_identity(&self) -> &str {
        concat!("icn-native-model-template:", env!("CARGO_PKG_VERSION"))
    }

    fn assess(
        &self,
        inputs: &icn_contracts::EffectiveTemplateInputs,
    ) -> Result<TemplateAssessment, String> {
        run_isolated_template_inspection(TemplateWorkerRequest {
            model_path: inputs.model_path.clone(),
        })
        .map_err(|error| format!("{error:#}"))
    }
}

#[cfg(not(test))]
const PLANNING_WORKER_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);
#[cfg(not(test))]
const MAX_PLANNING_WORKER_OUTPUT_BYTES: usize = 1024 * 1024;
#[cfg(not(test))]
const TEMPLATE_WORKER_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

impl NativeHardwareAssessor {
    fn effective_defaults(&self, profile: Option<&ModelPreviewProfile>) -> RuntimePlanDefaults {
        let mut defaults = self.defaults.clone();
        if let Some(profile) = profile {
            defaults.context_size = profile.context_length;
            defaults.max_sequences = profile.parallel_sequences;
        }
        defaults
    }

    async fn assess_resolved(
        &self,
        resolved: ResolvedModel,
        profile: Option<&icn_contracts::ModelPreviewProfile>,
    ) -> Result<HardwareAssessment, InventoryError> {
        let profiles = profile.cloned().into_iter().collect();
        let mut assessments = self.assess_resolved_profiles(resolved, profiles).await?;
        assessments.pop().ok_or_else(|| {
            InventoryError::Internal("native planner returned no assessment".to_owned())
        })
    }

    async fn assess_resolved_profiles(
        &self,
        resolved: ResolvedModel,
        profiles: Vec<ModelPreviewProfile>,
    ) -> Result<Vec<HardwareAssessment>, InventoryError> {
        Ok(self
            .assess_resolved_plans(resolved, profiles, false)
            .await?
            .into_iter()
            .map(|assessment| assessment.hardware)
            .collect())
    }

    async fn assess_resolved_execution_profiles(
        &self,
        resolved: ResolvedModel,
        profiles: Vec<ModelPreviewProfile>,
    ) -> Result<Vec<ModelExecutionAssessment>, InventoryError> {
        self.assess_resolved_plans(resolved, profiles, true).await
    }

    async fn assess_resolved_plans(
        &self,
        resolved: ResolvedModel,
        profiles: Vec<ModelPreviewProfile>,
        estimate_performance: bool,
    ) -> Result<Vec<ModelExecutionAssessment>, InventoryError> {
        self.assess_resolved_plans_with_policy(
            resolved,
            profiles,
            estimate_performance,
            CapacityPolicy::default(),
        )
        .await
    }

    async fn assess_resolved_plans_with_policy(
        &self,
        resolved: ResolvedModel,
        profiles: Vec<ModelPreviewProfile>,
        estimate_performance: bool,
        capacity_policy: CapacityPolicy,
    ) -> Result<Vec<ModelExecutionAssessment>, InventoryError> {
        let id = resolved.model.id.clone();
        let primary = resolved
            .components
            .iter()
            .filter(|component| {
                matches!(
                    component.role,
                    ComponentRole::Weights | ComponentRole::Shard
                )
            })
            .min_by_key(|component| component.shard_index.unwrap_or(0))
            .map(|component| component.path.clone())
            .ok_or_else(|| InventoryError::NotReady("model has no runnable weights".into()))?;
        let projector = resolved
            .components
            .iter()
            .find(|component| component.role == ComponentRole::Projector)
            .map(|component| component.path.clone());
        let mtp: Vec<PathBuf> = resolved
            .components
            .iter()
            .filter(|component| matches!(component.role, ComponentRole::Mtp | ComponentRole::Draft))
            .map(|component| component.path.clone())
            .collect();
        let defaults = if profiles.is_empty() {
            vec![self.effective_defaults(None)]
        } else {
            profiles
                .iter()
                .map(|profile| self.effective_defaults(Some(profile)))
                .collect()
        };
        // Hardware-only planning never calibrates. For execution assessment, only the first
        // request holds the calibration lock across native planning. Once a model-free result is
        // cached, concurrent model inspections proceed independently through the bounded pool.
        let mut calibration_guard = if estimate_performance {
            Some(self.calibration.lock().await)
        } else {
            None
        };
        let calibration_result = calibration_guard
            .as_ref()
            .and_then(|guard| guard.result.as_ref().cloned());
        if calibration_result.is_some() {
            calibration_guard.take();
        }
        let (calibration, calibration_unavailable) = match calibration_result {
            Some(Ok(calibration)) => (Some(calibration), None),
            Some(Err(error)) => (None, Some(error)),
            None => (None, None),
        };
        let request = PlanningWorkerRequest {
            primary,
            projector,
            mtp,
            defaults,
            estimate_performance,
            capacity_policy,
            calibration,
            calibration_unavailable,
        };
        let permit = Arc::clone(&self.planning_slots)
            .acquire_owned()
            .await
            .map_err(|_| InventoryError::Internal("native planner pool closed".to_owned()))?;
        let response = match spawn_blocking_traced(move || {
            let _permit = permit;
            run_isolated_planning(request)
        })
        .await
        {
            Ok(Ok(response)) => response,
            Ok(Err(error)) => {
                return Err(InventoryError::Internal(format!(
                    "hardware assessment failed for {}: {error:#}",
                    id.0
                )));
            }
            Err(error) => {
                return Err(InventoryError::Internal(format!(
                    "hardware assessment task failed for {}: {error}",
                    id.0
                )));
            }
        };
        if let Some(mut guard) = calibration_guard
            && guard.result.is_none()
            && let Some(calibration) = response.calibration.clone()
        {
            guard.result = Some(calibration);
        }
        Ok(response.assessments)
    }

    fn assessment_cache_key(
        &self,
        profile: Option<&ModelPreviewProfile>,
        snapshot: &HardwareSnapshot,
    ) -> Result<String, InventoryError> {
        serde_json::to_string(&(
            ASSESSMENT_RESOLVER_REVISION,
            CAPACITY_POLICY_REVISION,
            CapacityPolicy::default().reserve_bytes_per_domain,
            MTP_SELECTOR_REVISION,
            icn_hardware::GENERATION_PERFORMANCE_METHOD,
            llama_cpp_2::model::params::fit::FIT_DECODE_WORKLOAD_METHOD,
            llama_cpp_2::model::params::fit::FIT_CALIBRATION_METHOD,
            &snapshot.native_build,
            &snapshot.enabled_backends,
            &snapshot.topology_fingerprint,
            self.effective_defaults(profile),
        ))
        .map_err(|error| InventoryError::Internal(error.to_string()))
    }
}

struct NativeModelEvaluator {
    models: Arc<ModelManager>,
    assessor: Arc<NativeHardwareAssessor>,
}

impl NativeModelEvaluator {
    fn new(models: Arc<ModelManager>, assessor: Arc<NativeHardwareAssessor>) -> Self {
        Self { models, assessor }
    }

    async fn environment_id(&self) -> Result<AssessmentEnvironmentId, InventoryError> {
        let snapshot = HardwareProvider::snapshot(self.assessor.as_ref()).await?;
        let mut digest = Sha256::new();
        digest.update(b"magnitude-assessment-environment-v1\0");
        digest.update(snapshot.native_build.as_bytes());
        digest.update(b"\0");
        digest.update(snapshot.topology_fingerprint.as_bytes());
        Ok(AssessmentEnvironmentId(format!(
            "environment_{:x}",
            digest.finalize()
        )))
    }

    fn resolved_for_planning(
        resolved: &icn_contracts::models::ResolvedModelTarget,
    ) -> ResolvedModel {
        let mut target = resolved.target_model.clone();
        if let Some(draft) = &resolved.draft_model {
            target
                .components
                .extend(draft.components.iter().cloned().map(|mut component| {
                    component.role = ComponentRole::Draft;
                    component
                }));
        }
        target
    }

    async fn assess_profiles(
        &self,
        resolved: &icn_contracts::models::ResolvedModelTarget,
        profiles: &[DomainServingProfile],
        reserve_bytes: u64,
        include_performance: bool,
    ) -> Result<Vec<OfferingAssessment>, InventoryError> {
        let environment_id = self.environment_id().await?;
        let evidence = profiles
            .iter()
            .map(|profile| {
                serde_json::to_string(&(
                    ASSESSMENT_RESOLVER_REVISION,
                    CAPACITY_POLICY_REVISION,
                    MTP_SELECTOR_REVISION,
                    &environment_id.0,
                    &resolved.target_id.0,
                    profile.context_length,
                    profile.parallel_sequences,
                    reserve_bytes,
                    include_performance,
                ))
                .map_err(|error| InventoryError::Internal(error.to_string()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut results = evidence
            .iter()
            .map(|key| self.models.read_offering_assessment(key))
            .collect::<Vec<_>>();
        let missing = results
            .iter()
            .enumerate()
            .filter_map(|(index, assessment)| assessment.is_none().then_some(index))
            .collect::<Vec<_>>();
        if missing.is_empty() {
            return Ok(results
                .into_iter()
                .map(|assessment| assessment.expect("cache hit was checked"))
                .collect());
        }
        let native_profiles = missing
            .iter()
            .map(|index| ModelPreviewProfile {
                id: format!("assessment-{index}"),
                context_length: profiles[*index].context_length,
                parallel_sequences: profiles[*index].parallel_sequences,
            })
            .collect::<Vec<_>>();
        let assessed = self
            .assessor
            .assess_resolved_plans_with_policy(
                Self::resolved_for_planning(resolved),
                native_profiles,
                include_performance,
                CapacityPolicy {
                    reserve_bytes_per_domain: reserve_bytes,
                },
            )
            .await?;
        for (index, assessment) in missing.into_iter().zip(assessed) {
            let assessment = offering_assessment(
                &resolved.target_id,
                profiles[index].clone(),
                reserve_bytes,
                assessment,
            );
            self.models
                .write_offering_assessment(&evidence[index], &assessment);
            results[index] = Some(assessment);
        }
        Ok(results
            .into_iter()
            .map(|assessment| assessment.expect("missing assessment was populated"))
            .collect())
    }

    async fn fit_one(
        &self,
        resolved: &icn_contracts::models::ResolvedModelTarget,
        request: &FitModelsRequest,
    ) -> Result<FitModelResult, InventoryError> {
        let target_limit = match &resolved.target {
            icn_contracts::models::ModelOfferingTarget::Package { package } => {
                package.properties.maximum_context_length
            }
            icn_contracts::models::ModelOfferingTarget::SpeculativeDecodingPair {
                target, ..
            } => target.properties.maximum_context_length,
        };
        let maximum_context = request
            .maximum_context_length
            .min(target_limit)
            .min(200_000);
        if maximum_context < request.minimum_context_length {
            return Ok(FitModelResult::InvalidTarget {
                request_id: icn_contracts::models::ModelAssessmentRequestId(String::new()),
                failure: DomainModelFailure {
                    code: "model_context_limit".to_owned(),
                    message: format!(
                        "model context limit is {maximum_context} tokens, below the requested minimum of {}",
                        request.minimum_context_length
                    ),
                    retryable: false,
                },
            });
        }

        let reserve = request
            .capacity_policy
            .required_reserve_bytes_per_memory_domain;
        let mut lower = request.minimum_context_length;
        let mut upper = maximum_context;
        let upper_assessment = self
            .assess_profiles(
                resolved,
                &[DomainServingProfile {
                    context_length: upper,
                    parallel_sequences: 1,
                }],
                reserve,
                false,
            )
            .await?
            .pop()
            .expect("one requested profile produces one assessment");
        let mut best_context =
            matches!(upper_assessment, OfferingAssessment::Fits { .. }).then_some(upper);
        if best_context.is_none() {
            let minimum = self
                .assess_profiles(
                    resolved,
                    &[DomainServingProfile {
                        context_length: lower,
                        parallel_sequences: 1,
                    }],
                    reserve,
                    false,
                )
                .await?
                .pop()
                .expect("one requested profile produces one assessment");
            if !matches!(minimum, OfferingAssessment::Fits { .. }) {
                return Ok(match minimum {
                    OfferingAssessment::DoesNotFit {
                        limiting_resource,
                        deficit_bytes,
                        ..
                    } => FitModelResult::DoesNotFit {
                        request_id: icn_contracts::models::ModelAssessmentRequestId(String::new()),
                        target_id: resolved.target_id.clone(),
                        limiting_resource,
                        deficit_bytes,
                    },
                    OfferingAssessment::Incompatible { failure, .. } => {
                        FitModelResult::InvalidTarget {
                            request_id: icn_contracts::models::ModelAssessmentRequestId(
                                String::new(),
                            ),
                            failure,
                        }
                    }
                    OfferingAssessment::Fits { .. } => unreachable!(),
                });
            }
            best_context = Some(lower);
            while lower + 1 < upper {
                let middle = lower + (upper - lower) / 2;
                let assessment = self
                    .assess_profiles(
                        resolved,
                        &[DomainServingProfile {
                            context_length: middle,
                            parallel_sequences: 1,
                        }],
                        reserve,
                        false,
                    )
                    .await?
                    .pop()
                    .expect("one requested profile produces one assessment");
                if matches!(assessment, OfferingAssessment::Fits { .. }) {
                    lower = middle;
                    best_context = Some(middle);
                } else {
                    upper = middle;
                }
            }
        }
        let context_length = best_context.expect("minimum fitting context was recorded");
        let parallel_profiles = (1..=request.maximum_parallel_sequences)
            .map(|parallel_sequences| DomainServingProfile {
                context_length,
                parallel_sequences,
            })
            .collect::<Vec<_>>();
        let assessments = self
            .assess_profiles(resolved, &parallel_profiles, reserve, true)
            .await?;
        let Some((profile, assessment)) = parallel_profiles
            .into_iter()
            .zip(assessments)
            .rev()
            .find(|(_, assessment)| matches!(assessment, OfferingAssessment::Fits { .. }))
        else {
            return Err(InventoryError::Internal(
                "a previously fitting context no longer fits".to_owned(),
            ));
        };
        let configuration_id = serving_configuration_id(&resolved.target_id, &profile);
        Ok(FitModelResult::Fitted {
            request_id: icn_contracts::models::ModelAssessmentRequestId(String::new()),
            target_id: resolved.target_id.clone(),
            configuration: ModelServingConfiguration {
                id: configuration_id,
                target: resolved.target.clone(),
                profile,
            },
            assessment,
        })
    }
}

fn serving_configuration_id(
    target_id: &icn_contracts::models::ModelOfferingTargetId,
    profile: &DomainServingProfile,
) -> ModelServingConfigurationId {
    let mut digest = Sha256::new();
    digest.update(b"magnitude-serving-configuration-v1\0");
    digest.update(target_id.0.as_bytes());
    digest.update(b"\0");
    digest.update(profile.context_length.to_le_bytes());
    digest.update(profile.parallel_sequences.to_le_bytes());
    ModelServingConfigurationId(format!("configuration_{:x}", digest.finalize()))
}

fn offering_assessment(
    target_id: &icn_contracts::models::ModelOfferingTargetId,
    profile: DomainServingProfile,
    reserve_bytes: u64,
    assessment: ModelExecutionAssessment,
) -> OfferingAssessment {
    let context_tokens = profile.context_length;
    let configuration_id = serving_configuration_id(target_id, &profile);
    let mut digest = Sha256::new();
    digest.update(b"magnitude-offering-assessment-v1\0");
    digest.update(target_id.0.as_bytes());
    digest.update(b"\0");
    digest.update(profile.context_length.to_le_bytes());
    digest.update(profile.parallel_sequences.to_le_bytes());
    digest.update(reserve_bytes.to_le_bytes());
    let assessment_id = OfferingAssessmentId(format!("assessment_{:x}", digest.finalize()));
    match assessment.hardware {
        HardwareAssessment::Fits { memory, .. } => OfferingAssessment::Fits {
            profile,
            configuration_id,
            assessment_id,
            memory: memory
                .domains
                .into_iter()
                .map(|domain| MemoryAssessment {
                    memory_domain_id: domain.memory_domain,
                    capacity_bytes: domain.available_bytes.saturating_add(reserve_bytes),
                    required_bytes: domain.required_bytes,
                    required_reserve_bytes: reserve_bytes,
                    remaining_bytes: domain.margin_bytes,
                })
                .collect(),
            performance: performance_evidence(assessment.performance, context_tokens),
        },
        HardwareAssessment::DoesNotFit {
            memory,
            limiting_resource,
            ..
        } => OfferingAssessment::DoesNotFit {
            profile,
            configuration_id,
            assessment_id,
            memory: memory
                .domains
                .into_iter()
                .map(|domain| MemoryAssessment {
                    memory_domain_id: domain.memory_domain,
                    capacity_bytes: domain.available_bytes.saturating_add(reserve_bytes),
                    required_bytes: domain.required_bytes,
                    required_reserve_bytes: reserve_bytes,
                    remaining_bytes: domain.margin_bytes,
                })
                .collect(),
            limiting_resource,
            deficit_bytes: memory.deficit_bytes.max(1),
        },
        HardwareAssessment::InvalidArtifact { code, message } => OfferingAssessment::Incompatible {
            profile,
            configuration_id,
            failure: DomainModelFailure {
                code,
                message,
                retryable: false,
            },
        },
        HardwareAssessment::IncompatibleArtifact { code, message } => {
            OfferingAssessment::Incompatible {
                profile,
                configuration_id,
                failure: DomainModelFailure {
                    code,
                    message,
                    retryable: false,
                },
            }
        }
        HardwareAssessment::NotAssessed { reason } => OfferingAssessment::Incompatible {
            profile,
            configuration_id,
            failure: DomainModelFailure {
                code: "not_assessed".to_owned(),
                message: reason,
                retryable: true,
            },
        },
    }
}

fn performance_evidence(
    assessment: GenerationPerformanceAssessment,
    context_tokens: u32,
) -> Option<PerformanceEvidence> {
    match assessment {
        GenerationPerformanceAssessment::Estimated {
            method,
            confidence,
            points,
            ..
        } => points
            .into_iter()
            .find(|point| point.context_tokens == context_tokens)
            .map(|point| PerformanceEvidence {
                context_tokens: point.context_tokens,
                lower_tokens_per_second: point.lower_tokens_per_second,
                estimated_tokens_per_second: point.expected_tokens_per_second,
                upper_tokens_per_second: point.upper_tokens_per_second,
                confidence: match confidence {
                    icn_contracts::GenerationPerformanceConfidence::High => {
                        PerformanceConfidence::High
                    }
                    icn_contracts::GenerationPerformanceConfidence::Moderate => {
                        PerformanceConfidence::Moderate
                    }
                    icn_contracts::GenerationPerformanceConfidence::Low => {
                        PerformanceConfidence::Low
                    }
                },
                method,
            }),
        GenerationPerformanceAssessment::Unavailable { .. } => None,
    }
}

impl ModelEvaluator for NativeModelEvaluator {
    fn assess(
        &self,
        request: AssessModelsRequest,
    ) -> BoxFuture<'_, Result<AssessModelsResponse, InventoryError>> {
        Box::pin(async move {
            let environment_id = self.environment_id().await?;
            let reserve_bytes = request
                .capacity_policy
                .required_reserve_bytes_per_memory_domain;
            let include_performance = request.include_performance;
            let evaluated = futures_util::stream::iter(request.requests.into_iter().enumerate())
                .map(|(index, item)| async move {
                    let request_id = item.request_id;
                    let result = match self.models.resolve_target(item.target).await {
                        Ok(resolved) => AssessModelResult::Assessed {
                            request_id,
                            target_id: resolved.target_id.clone(),
                            profiles: self
                                .assess_profiles(
                                    &resolved,
                                    &item.profiles,
                                    reserve_bytes,
                                    include_performance,
                                )
                                .await?,
                        },
                        Err(error) => AssessModelResult::InvalidTarget {
                            request_id,
                            failure: DomainModelFailure {
                                code: "invalid_target".to_owned(),
                                message: error.to_string(),
                                retryable: false,
                            },
                        },
                    };
                    Ok::<_, InventoryError>((index, result))
                })
                .buffer_unordered(MODEL_ASSESSMENT_CONCURRENCY)
                .collect::<Vec<_>>()
                .await;
            let mut results = evaluated.into_iter().collect::<Result<Vec<_>, _>>()?;
            results.sort_unstable_by_key(|(index, _)| *index);
            Ok(AssessModelsResponse {
                environment_id,
                results: results.into_iter().map(|(_, result)| result).collect(),
            })
        })
    }

    fn fit(
        &self,
        request: FitModelsRequest,
    ) -> BoxFuture<'_, Result<FitModelsResponse, InventoryError>> {
        Box::pin(async move {
            if request.minimum_context_length == 0
                || request.minimum_context_length > request.maximum_context_length
                || request.maximum_parallel_sequences == 0
            {
                return Err(InventoryError::InvalidRequest(
                    "fit bounds must be positive and ordered".to_owned(),
                ));
            }
            let environment_id = self.environment_id().await?;
            let mut results = Vec::with_capacity(request.targets.len());
            for item in &request.targets {
                let request_id = item.request_id.clone();
                match self.models.resolve_target(item.target.clone()).await {
                    Ok(resolved) => {
                        let mut result = self.fit_one(&resolved, &request).await?;
                        match &mut result {
                            FitModelResult::Fitted {
                                request_id: result_request_id,
                                ..
                            }
                            | FitModelResult::DoesNotFit {
                                request_id: result_request_id,
                                ..
                            }
                            | FitModelResult::InvalidTarget {
                                request_id: result_request_id,
                                ..
                            } => *result_request_id = request_id,
                        }
                        results.push(result);
                    }
                    Err(error) => results.push(FitModelResult::InvalidTarget {
                        request_id,
                        failure: DomainModelFailure {
                            code: "invalid_target".to_owned(),
                            message: error.to_string(),
                            retryable: false,
                        },
                    }),
                }
            }
            Ok(FitModelsResponse {
                environment_id,
                results,
            })
        })
    }
}

fn planner_concurrency() -> usize {
    std::thread::available_parallelism().map_or(1, |cores| cores.get().clamp(1, 16))
}

fn unavailable_performance(
    code: &str,
    message: impl Into<String>,
) -> GenerationPerformanceAssessment {
    GenerationPerformanceAssessment::Unavailable {
        method: icn_hardware::GENERATION_PERFORMANCE_METHOD.to_owned(),
        code: code.to_owned(),
        message: message.into(),
    }
}

fn assess_planning_request(
    request: PlanningWorkerRequest,
) -> anyhow::Result<PlanningWorkerResponse> {
    let native_backend = NativeBackend::initialize()?;
    assess_planning_request_with_backend(request, &native_backend)
}

fn assess_planning_request_with_backend(
    request: PlanningWorkerRequest,
    native_backend: &NativeBackend,
) -> anyhow::Result<PlanningWorkerResponse> {
    let backend = native_backend.as_llama_backend();
    let mut plans = request
        .defaults
        .into_iter()
        .map(|defaults| {
            execution_intent(
                request.primary.clone(),
                request.projector.clone(),
                &defaults,
            )
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let calibration = if request.estimate_performance {
        Some(
            match (request.calibration, request.calibration_unavailable) {
                (Some(calibration), _) => Ok(calibration),
                (None, Some(error)) => Err(error),
                (None, None) => llama_cpp_2::model::params::fit::FitCalibration::measure(backend)
                    .map_err(|error| error.to_string()),
            },
        )
    } else {
        None
    };
    let capacity_policy = request.capacity_policy;
    let assess_without_performance = |code: &str, message: String| {
        icn_hardware::assess_profiles_with_backend(backend, &plans, capacity_policy).map(
            |assessments| {
                assessments
                    .into_iter()
                    .map(|hardware| ModelExecutionAssessment {
                        hardware,
                        performance: unavailable_performance(code, message.clone()),
                    })
                    .collect()
            },
        )
    };
    let base = match calibration.as_ref() {
        Some(Ok(calibration)) => icn_hardware::assess_execution_profiles_with_backend(
            backend,
            &plans,
            capacity_policy,
            calibration,
        )
        .or_else(|error| {
            assess_without_performance("performance_estimation_failed", error.to_string())
        }),
        Some(Err(calibration_error)) => {
            assess_without_performance("calibration_failed", calibration_error.clone())
        }
        None => icn_hardware::assess_profiles_with_backend(backend, &plans, capacity_policy).map(
            |assessments| {
                assessments
                    .into_iter()
                    .map(|hardware| ModelExecutionAssessment {
                        hardware,
                        performance: GenerationPerformanceAssessment::not_requested(),
                    })
                    .collect()
            },
        ),
    }?;
    let assessments = plans
        .iter_mut()
        .zip(base)
        .map(|(plan, base)| {
            if !matches!(base.hardware, HardwareAssessment::Fits { .. }) {
                return Ok(base);
            }
            plan.mtp = icn_mtp::select_mtp_with_backend(
                backend,
                plan,
                icn_mtp::CandidatePolicy::Automatic(&request.mtp),
            )
            .context("failed to select a native MTP configuration")?;
            if matches!(plan.mtp, icn_contracts::MtpConfig::Disabled { .. }) {
                return Ok(base);
            }
            let hardware =
                icn_hardware::assess_with_backend(backend, plan, capacity_policy)?.assessment;
            let performance = if matches!(hardware, HardwareAssessment::Fits { .. }) {
                // Phase 1 intentionally estimates baseline target-model decode. MTP changes fit
                // memory but is not credited with an unmeasured speculative-decoding speedup.
                base.performance
            } else {
                unavailable_performance(
                    "configuration_does_not_fit",
                    "generation performance is unavailable for a configuration that does not fit",
                )
            };
            Ok(ModelExecutionAssessment {
                hardware,
                performance,
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(PlanningWorkerResponse {
        assessments,
        calibration,
    })
}

#[cfg(test)]
fn test_native_backend() -> NativeBackend {
    static BACKEND: std::sync::OnceLock<NativeBackend> = std::sync::OnceLock::new();
    BACKEND
        .get_or_init(|| NativeBackend::initialize().expect("initialize test native backend"))
        .clone()
}

#[cfg(test)]
fn run_isolated_planning(request: PlanningWorkerRequest) -> anyhow::Result<PlanningWorkerResponse> {
    icn_engine::disable_native_diagnostics();
    let native_backend = test_native_backend();
    assess_planning_request_with_backend(request, &native_backend)
}

#[cfg(not(test))]
fn run_isolated_planning(request: PlanningWorkerRequest) -> anyhow::Result<PlanningWorkerResponse> {
    use std::io::Write as _;

    let executable = std::env::current_exe().context("failed to locate ICN planner executable")?;
    let mut child = ProcessCommand::new(executable)
        .arg("plan-worker")
        .env("MAGNITUDE_OTEL", "0")
        .env("RUST_LOG", "error")
        .env_remove("MAGNITUDE_OTEL_ENDPOINT")
        .env_remove("OTEL_EXPORTER_OTLP_ENDPOINT")
        .env_remove("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
        .env_remove("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("failed to start isolated native planner")?;
    serde_json::to_writer(
        child
            .stdin
            .as_mut()
            .context("isolated native planner stdin was unavailable")?,
        &request,
    )
    .context("failed to encode isolated native planner request")?;
    child
        .stdin
        .take()
        .context("isolated native planner stdin was unavailable")?
        .flush()
        .context("failed to flush isolated native planner request")?;
    let deadline = std::time::Instant::now() + PLANNING_WORKER_TIMEOUT;
    loop {
        if child
            .try_wait()
            .context("failed to observe isolated native planner")?
            .is_some()
        {
            break;
        }
        if std::time::Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            anyhow::bail!("isolated native planner exceeded its time bound");
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let output = child
        .wait_with_output()
        .context("failed to await isolated native planner")?;
    if output.stdout.len() > MAX_PLANNING_WORKER_OUTPUT_BYTES
        || output.stderr.len() > MAX_PLANNING_WORKER_OUTPUT_BYTES
    {
        anyhow::bail!("isolated native planner exceeded its output bound");
    }
    if !output.status.success() {
        let diagnostic = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!(
            "isolated native planner exited with {}: {}",
            output.status,
            diagnostic.trim().chars().take(4_096).collect::<String>()
        );
    }
    serde_json::from_slice(&output.stdout)
        .context("isolated native planner returned an invalid assessment")
}

fn run_planning_worker() -> anyhow::Result<()> {
    let request = serde_json::from_reader(std::io::stdin().lock())
        .context("failed to decode native planner request")?;
    let assessment = assess_planning_request(request)?;
    serde_json::to_writer(std::io::stdout().lock(), &assessment)
        .context("failed to encode native planner result")?;
    Ok(())
}

fn inspect_template_request(request: TemplateWorkerRequest) -> anyhow::Result<TemplateAssessment> {
    icn_engine::disable_native_diagnostics();
    let native_backend = NativeBackend::initialize()?;
    inspect_template_request_with_backend(request, &native_backend)
}

fn inspect_template_request_with_backend(
    request: TemplateWorkerRequest,
    native_backend: &NativeBackend,
) -> anyhow::Result<TemplateAssessment> {
    let inspection = icn_reasoning::inspect_template_inputs_with_backend(
        native_backend.as_llama_backend(),
        &icn_contracts::EffectiveTemplateInputs {
            model_path: request.model_path,
        },
    )?;
    Ok(TemplateAssessment {
        capabilities: inspection.capabilities,
        reasoning: inspection.reasoning,
        fingerprint: inspection.template_fingerprint,
    })
}

#[cfg(test)]
fn run_isolated_template_inspection(
    request: TemplateWorkerRequest,
) -> anyhow::Result<TemplateAssessment> {
    let native_backend = test_native_backend();
    inspect_template_request_with_backend(request, &native_backend)
}

#[cfg(not(test))]
fn run_isolated_template_inspection(
    request: TemplateWorkerRequest,
) -> anyhow::Result<TemplateAssessment> {
    use std::io::Write as _;

    let executable = std::env::current_exe().context("failed to locate ICN template worker")?;
    let mut child = ProcessCommand::new(executable)
        .arg("template-worker")
        .env("MAGNITUDE_OTEL", "0")
        .env("RUST_LOG", "error")
        .env_remove("MAGNITUDE_OTEL_ENDPOINT")
        .env_remove("OTEL_EXPORTER_OTLP_ENDPOINT")
        .env_remove("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
        .env_remove("OTEL_EXPORTER_OTLP_LOGS_ENDPOINT")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("failed to start isolated native template worker")?;
    serde_json::to_writer(
        child
            .stdin
            .as_mut()
            .context("template worker stdin was unavailable")?,
        &request,
    )
    .context("failed to encode template worker request")?;
    child
        .stdin
        .take()
        .context("template worker stdin was unavailable")?
        .flush()
        .context("failed to flush template worker request")?;
    let deadline = std::time::Instant::now() + TEMPLATE_WORKER_TIMEOUT;
    loop {
        if child
            .try_wait()
            .context("failed to observe isolated native template worker")?
            .is_some()
        {
            break;
        }
        if std::time::Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            anyhow::bail!("isolated native template worker exceeded its time bound");
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let output = child
        .wait_with_output()
        .context("failed to await isolated native template worker")?;
    if output.stdout.len() > MAX_PLANNING_WORKER_OUTPUT_BYTES
        || output.stderr.len() > MAX_PLANNING_WORKER_OUTPUT_BYTES
    {
        anyhow::bail!("isolated native template worker exceeded its output bound");
    }
    if !output.status.success() {
        anyhow::bail!(
            "template worker exited with {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
                .trim()
                .chars()
                .take(4_096)
                .collect::<String>()
        );
    }
    serde_json::from_slice(&output.stdout).context("template worker returned an invalid assessment")
}

fn run_template_worker() -> anyhow::Result<()> {
    let request = serde_json::from_reader(std::io::stdin().lock())
        .context("failed to decode native template request")?;
    let assessment = inspect_template_request(request)?;
    serde_json::to_writer(std::io::stdout().lock(), &assessment)
        .context("failed to encode native template assessment")?;
    Ok(())
}

impl InventoryHardwareAssessor for NativeHardwareAssessor {
    fn cache_key(&self) -> BoxFuture<'_, Result<String, InventoryError>> {
        Box::pin(async move {
            let snapshot = HardwareProvider::snapshot(self).await?;
            ModelHardwareAssessor::cache_key(self, None, &snapshot)
        })
    }

    fn assess(
        &self,
        resolved: ResolvedModel,
    ) -> BoxFuture<'_, Result<HardwareAssessment, InventoryError>> {
        Box::pin(self.assess_resolved(resolved, None))
    }

    fn assess_serving(
        &self,
        resolved: ResolvedModel,
        profile: icn_contracts::ServingProfile,
    ) -> BoxFuture<'_, Result<HardwareAssessment, InventoryError>> {
        Box::pin(async move {
            self.assess_resolved(
                resolved,
                Some(&ModelPreviewProfile {
                    id: "serving".to_owned(),
                    context_length: profile.context_length,
                    parallel_sequences: profile.parallel_sequences,
                }),
            )
            .await
        })
    }
}

impl ModelHardwareAssessor for NativeHardwareAssessor {
    fn cache_key(
        &self,
        profile: Option<&ModelPreviewProfile>,
        snapshot: &HardwareSnapshot,
    ) -> Result<String, InventoryError> {
        self.assessment_cache_key(profile, snapshot)
    }

    fn assess_profile(
        &self,
        model: ResolvedModel,
        profile: Option<ModelPreviewProfile>,
    ) -> BoxFuture<'_, Result<HardwareAssessment, InventoryError>> {
        Box::pin(async move { self.assess_resolved(model, profile.as_ref()).await })
    }

    fn assess_profiles(
        &self,
        model: ResolvedModel,
        profiles: Vec<ModelPreviewProfile>,
    ) -> BoxFuture<'_, Result<Vec<HardwareAssessment>, InventoryError>> {
        Box::pin(async move { self.assess_resolved_profiles(model, profiles).await })
    }

    fn assess_execution_profiles(
        &self,
        model: ResolvedModel,
        profiles: Vec<ModelPreviewProfile>,
    ) -> BoxFuture<'_, Result<Vec<ModelExecutionAssessment>, InventoryError>> {
        Box::pin(async move {
            self.assess_resolved_execution_profiles(model, profiles)
                .await
        })
    }
}

impl HardwareProvider for NativeHardwareAssessor {
    fn snapshot(&self) -> BoxFuture<'_, Result<HardwareSnapshot, InventoryError>> {
        Box::pin(async move {
            let _guard = self.gate.lock().await;
            let native_executor = self
                .native_executor
                .read()
                .map_err(|_| InventoryError::Internal("native executor lock poisoned".to_owned()))?
                .clone();
            let native_build = build_identity::native_build();
            let observed_runtime = native_executor
                .as_ref()
                .map(|resident| (resident.generation, resident.model_id.clone()));
            let enabled_backends = build_identity::enabled_backends()
                .into_iter()
                .map(str::to_owned)
                .collect();
            let native_backend = self.native_backend.clone();
            let mut snapshot = spawn_blocking_traced(move || match native_executor {
                Some(resident) => {
                    let snapshot = resident
                        .backend
                        .observe_hardware(
                            resident.generation,
                            CapacityPolicy::default(),
                            native_build,
                            enabled_backends,
                        )
                        .map_err(|error| InventoryError::Internal(error.to_string()))?;
                    Ok(snapshot)
                }
                None => Ok(native_backend.discover_hardware(
                    CapacityPolicy::default(),
                    native_build,
                    enabled_backends,
                )),
            })
            .await
            .map_err(|error| InventoryError::Internal(error.to_string()))??;
            if let Some((generation, model_id)) = observed_runtime {
                let unchanged = self
                    .native_executor
                    .read()
                    .map_err(|_| {
                        InventoryError::Internal("native executor lock poisoned".to_owned())
                    })?
                    .as_ref()
                    .is_some_and(|current| {
                        current.generation == generation && current.model_id == model_id
                    });
                if !unchanged {
                    snapshot.resident_memory = None;
                }
            }
            let mut calibration = self.calibration.lock().await;
            if calibration.topology_fingerprint.as_deref()
                != Some(snapshot.topology_fingerprint.as_str())
            {
                calibration.topology_fingerprint = Some(snapshot.topology_fingerprint.clone());
                calibration.result = None;
            }
            Ok(snapshot)
        })
    }
}

#[derive(Clone)]
struct NativeRuntimeController {
    backends: BackendRegistry,
    inventory: Arc<ModelManager>,
    native_backend: NativeBackend,
    native_executor: Arc<RwLock<Option<ResidentNativeExecutor>>>,
    defaults: RuntimePlanDefaults,
    state: Arc<tokio::sync::RwLock<RuntimeState>>,
    mutation: Arc<tokio::sync::Mutex<()>>,
}

#[derive(Clone)]
struct RuntimeFailure {
    code: &'static str,
    message: String,
    retryable: bool,
}

impl RuntimeFailure {
    fn new(code: &'static str, message: impl Into<String>, retryable: bool) -> Self {
        Self {
            code,
            message: message.into(),
            retryable,
        }
    }
}

struct RuntimeTransitionFailure {
    event: RuntimeFailure,
}

impl RuntimeTransitionFailure {
    fn new(event: RuntimeFailure) -> Self {
        Self { event }
    }
}

impl From<InventoryError> for RuntimeTransitionFailure {
    fn from(error: InventoryError) -> Self {
        let message = error.to_string();
        Self::new(RuntimeFailure::new(
            "runtime_transition_failed",
            message,
            true,
        ))
    }
}

impl NativeRuntimeController {
    fn load_failure(error: InventoryError) -> DomainModelFailure {
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
            InventoryError::Runtime {
                code, retryable, ..
            } => (code.clone(), *retryable),
            InventoryError::Internal(_) => ("internal".to_owned(), true),
        };
        DomainModelFailure {
            code,
            message: error.to_string(),
            retryable,
        }
    }

    fn new(
        backends: BackendRegistry,
        inventory: Arc<ModelManager>,
        native_backend: NativeBackend,
        native_executor: Arc<RwLock<Option<ResidentNativeExecutor>>>,
        defaults: RuntimePlanDefaults,
        initial: RuntimeState,
    ) -> Self {
        Self {
            backends,
            inventory,
            native_backend,
            native_executor,
            defaults,
            state: Arc::new(tokio::sync::RwLock::new(initial)),
            mutation: Arc::new(tokio::sync::Mutex::new(())),
        }
    }

    fn profile_defaults(
        &self,
        profile: &RuntimeExecutionProfile,
    ) -> Result<RuntimePlanDefaults, InventoryError> {
        let mut defaults = self.defaults.clone();
        defaults.context_size = profile.context_length;
        defaults.max_sequences = profile.parallel_sequences;
        Ok(defaults)
    }

    async fn resolved_configuration_load(
        &self,
        configuration: &ModelServingConfiguration,
    ) -> Result<
        (
            ResolvedModel,
            ExecutionIntent,
            MtpCandidateSelection,
            Vec<ModelPackageId>,
        ),
        InventoryError,
    > {
        let (target, package_ids) = match &configuration.target {
            DomainModelOfferingTarget::Package { package } => (
                ModelTargetInput::Package {
                    package: ModelPackageOperand::Installed {
                        package_id: package.id.clone(),
                    },
                },
                vec![package.id.clone()],
            ),
            DomainModelOfferingTarget::SpeculativeDecodingPair { target, draft, .. } => (
                ModelTargetInput::SpeculativeDecodingPair {
                    target: ModelPackageOperand::Installed {
                        package_id: target.id.clone(),
                    },
                    draft: ModelPackageOperand::Installed {
                        package_id: draft.id.clone(),
                    },
                },
                vec![target.id.clone(), draft.id.clone()],
            ),
        };
        let resolved = self.inventory.resolve_target(target).await?;
        let mut model = resolved.target_model;
        if let Some(draft) = resolved.draft_model {
            model
                .components
                .extend(draft.components.into_iter().map(|mut component| {
                    component.role = ComponentRole::Draft;
                    component
                }));
        }
        let primary = model
            .components
            .iter()
            .filter(|component| {
                matches!(
                    component.role,
                    ComponentRole::Weights | ComponentRole::Shard
                )
            })
            .min_by_key(|component| component.shard_index.unwrap_or(0))
            .map(|component| component.path.clone())
            .ok_or_else(|| InventoryError::NotReady("model has no runnable weights".into()))?;
        let projector = model
            .components
            .iter()
            .find(|component| component.role == ComponentRole::Projector)
            .map(|component| component.path.clone());
        let mtp = model
            .components
            .iter()
            .filter(|component| matches!(component.role, ComponentRole::Mtp | ComponentRole::Draft))
            .map(|component| component.path.clone())
            .collect();
        let defaults = self.profile_defaults(&RuntimeExecutionProfile {
            context_length: configuration.profile.context_length,
            parallel_sequences: configuration.profile.parallel_sequences,
        })?;
        let plan = execution_intent(primary, projector, &defaults).map_err(|error| {
            InventoryError::Internal(format!("failed to resolve runtime intent: {error:#}"))
        })?;
        Ok((
            model,
            plan,
            MtpCandidateSelection::Automatic(mtp),
            package_ids,
        ))
    }

    #[tracing::instrument(
        name = "icn.runtime.load.operation",
        skip_all,
        fields(model.configuration.id = %configuration_id)
    )]
    async fn perform_prepared_transition(
        self,
        configuration_id: String,
        profile: RuntimeExecutionProfile,
        resolved: ResolvedModel,
        plan: ExecutionIntent,
        mtp_selection: MtpCandidateSelection,
        package_ids: Vec<ModelPackageId>,
        events: tokio::sync::mpsc::UnboundedSender<ModelLoadEvent>,
    ) -> Result<RuntimeResidencyId, RuntimeTransitionFailure> {
        let model_id = configuration_id;
        let existing = self.state.read().await.clone();

        if existing
            .resident
            .as_ref()
            .is_some_and(|resident| resident.model_id == model_id && resident.profile == profile)
        {
            if let Some(lease) = self.backends.lease() {
                drop(lease);
                let residency_id = existing
                    .resident
                    .expect("matching resident was checked")
                    .residency_id;
                return Ok(RuntimeResidencyId(residency_id));
            }
        };
        let _backend_mutation = self.backends.begin_mutation().await;

        if existing.resident.is_some() {
            let _ = events.send(ModelLoadEvent::Progress {
                stage: ModelLoadStage::Unloading,
                fraction: None,
            });
        }
        *self.state.write().await = RuntimeState { resident: None };

        if let Ok(mut slot) = self.native_executor.write() {
            *slot = None;
        }
        self.backends.clear();

        let load_model_id = model_id.clone();
        let native_backend = self.native_backend.clone();
        let progress_events = events.clone();
        let _ = events.send(ModelLoadEvent::Progress {
            stage: ModelLoadStage::Loading,
            fraction: Some(0.0),
        });
        let load_task = spawn_blocking_traced(move || {
            native_backend.load_with_progress(load_model_id, plan, mtp_selection, move |fraction| {
                let _ = progress_events.send(ModelLoadEvent::Progress {
                    stage: ModelLoadStage::Loading,
                    fraction: Some(fraction),
                });
            })
        });
        let load_result = load_task.await;
        let backend = match load_result {
            Ok(Ok(backend)) => Arc::new(backend),
            Ok(Err(error)) => {
                let (code, retryable) = match &error {
                    ModelLoadError::InvalidConfiguration(_) => ("invalid_configuration", false),
                    ModelLoadError::MtpSelection(_) => ("incompatible_auxiliary", false),
                    ModelLoadError::Planning(_) => ("planner_failed", true),
                    ModelLoadError::AssessmentRejected(assessment) => match assessment.as_ref() {
                        HardwareAssessment::DoesNotFit { .. } => ("does_not_fit", false),
                        HardwareAssessment::InvalidArtifact { .. } => ("invalid_artifact", false),
                        HardwareAssessment::IncompatibleArtifact { .. } => {
                            ("incompatible_artifact", false)
                        }
                        HardwareAssessment::NotAssessed { .. } => ("planner_failed", true),
                        HardwareAssessment::Fits { .. } => ("planner_invariant", true),
                    },
                    ModelLoadError::MemoryAttribution(_) => ("memory_attribution_failed", true),
                    ModelLoadError::Backend(_) => ("backend_load_failed", true),
                };
                return Err(RuntimeTransitionFailure::new(RuntimeFailure::new(
                    code,
                    error.to_string(),
                    retryable,
                )));
            }
            Err(error) => {
                return Err(RuntimeTransitionFailure::new(RuntimeFailure::new(
                    "load_task_failed",
                    error.to_string(),
                    true,
                )));
            }
        };
        let _ = events.send(ModelLoadEvent::Progress {
            stage: ModelLoadStage::Verifying,
            fraction: None,
        });
        match backend.properties() {
            Ok(_) => {}
            Err(error) => {
                return Err(RuntimeTransitionFailure::new(RuntimeFailure::new(
                    "verification_failed",
                    error.to_string(),
                    false,
                )));
            }
        }
        let mut aliases = std::collections::BTreeSet::new();
        aliases.insert(resolved.model.name.clone());
        let generation = self
            .backends
            .replace(Arc::clone(&backend) as Arc<dyn CompletionBackend>, aliases);
        if let Ok(mut slot) = self.native_executor.write() {
            *slot = Some(ResidentNativeExecutor {
                generation,
                model_id: model_id.clone(),
                backend: Arc::clone(&backend),
            });
        }
        let residency_id = format!("residency-{model_id}-{generation}");
        let state = RuntimeState {
            resident: Some(ResidentTarget {
                model_id: model_id.clone(),
                residency_id: residency_id.clone(),
                profile,
                package_ids,
            }),
        };
        *self.state.write().await = state.clone();
        tracing::info!("runtime model ready");
        Ok(RuntimeResidencyId(residency_id))
    }

    async fn unload_resident_with_admission_closed(
        &self,
        model_id: &str,
        _backend_mutation: BackendMutationGuard,
    ) -> Result<bool, InventoryError> {
        let previous = self.state.read().await.clone();
        if !previous
            .resident
            .as_ref()
            .is_some_and(|resident| resident.model_id == model_id)
        {
            return Ok(false);
        }

        if let Ok(mut slot) = self.native_executor.write() {
            *slot = None;
        }
        self.backends.clear();
        *self.state.write().await = RuntimeState { resident: None };
        Ok(true)
    }

    async fn unload_resident_locked(&self, model_id: &str) -> Result<bool, InventoryError> {
        let backend_mutation = self.backends.begin_mutation().await;
        self.unload_resident_with_admission_closed(model_id, backend_mutation)
            .await
    }
}

impl RuntimeController for NativeRuntimeController {
    fn load_configuration(&self, request: LoadModelRequest) -> BoxStream<'static, ModelLoadEvent> {
        let controller = self.clone();
        let (events, receiver) = tokio::sync::mpsc::unbounded_channel();
        tokio::spawn(async move {
            let _ = events.send(ModelLoadEvent::Progress {
                stage: ModelLoadStage::Queued,
                fraction: None,
            });
            let _guard = controller.mutation.lock().await;
            let configuration = request.configuration;
            let _ = events.send(ModelLoadEvent::Progress {
                stage: ModelLoadStage::Resolving,
                fraction: None,
            });
            let (resolved, plan, mtp_selection, package_ids) =
                match controller.resolved_configuration_load(&configuration).await {
                    Ok(resolved) => resolved,
                    Err(error) => {
                        let _ = events.send(ModelLoadEvent::Failed {
                            failure: Self::load_failure(error),
                        });
                        return;
                    }
                };
            let profile = RuntimeExecutionProfile {
                context_length: configuration.profile.context_length,
                parallel_sequences: configuration.profile.parallel_sequences,
            };
            let residency_id = match controller
                .clone()
                .perform_prepared_transition(
                    configuration.id.0.clone(),
                    profile,
                    resolved,
                    plan,
                    mtp_selection,
                    package_ids,
                    events.clone(),
                )
                .await
            {
                Ok(residency_id) => residency_id,
                Err(failure) => {
                    let _ = events.send(ModelLoadEvent::Failed {
                        failure: DomainModelFailure {
                            code: failure.event.code.to_owned(),
                            message: failure.event.message,
                            retryable: failure.event.retryable,
                        },
                    });
                    return;
                }
            };
            let mut digest = Sha256::new();
            digest.update(b"magnitude-runtime-execution-evidence-v1\0");
            digest.update(configuration.id.0.as_bytes());
            digest.update(b"\0");
            digest.update(residency_id.0.as_bytes());
            let _ = events.send(ModelLoadEvent::Ready {
                ready: LoadModelReady {
                    residency_id,
                    configuration_id: configuration.id,
                    execution_evidence_id: format!("execution_{:x}", digest.finalize()),
                },
            });
        });
        UnboundedReceiverStream::new(receiver).boxed()
    }

    fn unload_residency(&self, residency_id: String) -> BoxFuture<'_, Result<(), InventoryError>> {
        Box::pin(async move {
            let _guard = self.mutation.lock().await;
            let resident = self
                .state
                .read()
                .await
                .resident
                .clone()
                .filter(|resident| resident.residency_id == residency_id)
                .ok_or_else(|| InventoryError::NotFound(residency_id.clone()))?;
            self.unload_resident_locked(&resident.model_id).await?;
            Ok(())
        })
    }

    fn remove_installed(
        &self,
        package_id: ModelPackageId,
    ) -> BoxFuture<'_, Result<RemoveInstalledModelPackageResponse, InventoryError>> {
        Box::pin(async move {
            let _guard = self.mutation.lock().await;
            if self
                .state
                .read()
                .await
                .resident
                .as_ref()
                .is_some_and(|resident| resident.package_ids.contains(&package_id))
            {
                return Err(InventoryError::Loaded(package_id.0));
            }
            self.inventory.remove_installed(&package_id).await
        })
    }

    fn lease(
        &self,
        configuration_id: String,
    ) -> BoxFuture<'_, Result<icn_api::BackendLease, InventoryError>> {
        Box::pin(async move {
            let _guard = self.mutation.lock().await;
            let resident = self.state.read().await.clone();
            if resident
                .resident
                .as_ref()
                .is_none_or(|resident| resident.model_id != configuration_id)
            {
                return Err(InventoryError::NotReady(format!(
                    "configuration {configuration_id} is not loaded"
                )));
            }
            self.backends.lease().ok_or_else(|| {
                InventoryError::NotReady(format!(
                    "configuration {configuration_id} is not available for inference"
                ))
            })
        })
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let _telemetry = telemetry::init()?;
    // Native planner diagnostics are extremely verbose and can dominate metadata-only fitting.
    // ICN emits bounded, structured operation telemetry at the service boundary instead.
    icn_engine::disable_native_diagnostics();
    match Cli::parse().command {
        Command::Serve {
            bind,
            instance_id,
            parent_pid,
            auth_token,
            fake,
            model_store,
            model_sources,
            hf_caches,
        } => {
            let inventory_root = match model_store {
                Some(root) => root,
                None => InventoryConfig::default_root()
                    .context("failed to determine default model store")?,
            };
            let mut inventory_config = InventoryConfig::with_root(inventory_root)
                .context("invalid model inventory configuration")?;
            inventory_config.model_sources.extend(model_sources);
            inventory_config.hf_cache_dirs.extend(hf_caches);
            let plan_defaults = runtime_plan_defaults();
            let native_backend = NativeBackend::initialize()
                .context("failed to initialize the process native backend")?;
            let inventory = Arc::new(
                ModelManager::open_with_template_assessor(
                    inventory_config,
                    Some(Arc::new(NativeTemplateAssessor)),
                )
                .await
                .context("failed to initialize model inventory")?,
            );
            let native_executor_slot = Arc::new(RwLock::new(None));
            let inventory_hardware_assessor = Arc::new(NativeHardwareAssessor {
                defaults: plan_defaults.clone(),
                native_backend: native_backend.clone(),
                native_executor: Arc::clone(&native_executor_slot),
                gate: tokio::sync::Mutex::new(()),
                planning_slots: Arc::new(tokio::sync::Semaphore::new(planner_concurrency())),
                calibration: tokio::sync::Mutex::new(CalibrationCache::default()),
            });
            inventory
                .set_hardware_assessor(inventory_hardware_assessor.clone())
                .context("failed to configure inventory hardware assessment")?;
            let previewer = Arc::new(ModelPreviewService::new(
                inventory.clone(),
                inventory_hardware_assessor.clone(),
            ));
            let recommendable_catalog = Arc::new(NativeRecommendableCatalog::new(
                inventory.clone(),
                previewer.clone(),
            ));
            let model_evaluator = Arc::new(NativeModelEvaluator::new(
                inventory.clone(),
                inventory_hardware_assessor.clone(),
            ));
            let model_downloads = Arc::new(ManagedModelDownloads::open(inventory.clone()));
            let backends = BackendRegistry::empty();
            if fake {
                let model_id = "icn-fake".to_owned();
                let mut aliases = std::collections::BTreeSet::new();
                aliases.insert(model_id.clone());
                backends.replace(
                    Arc::new(FakeBackend::new(model_id.clone(), "Hello from ICN.")),
                    aliases,
                );
            }
            let native_build = build_identity::native_build();
            let identity = ServerIdentity {
                instance_id: instance_id.clone(),
                api_version: 1,
                native_build: native_build.clone(),
            };
            let runtime = (!fake).then(|| {
                Arc::new(NativeRuntimeController::new(
                    backends.clone(),
                    inventory.clone(),
                    native_backend,
                    native_executor_slot,
                    plan_defaults,
                    RuntimeState { resident: None },
                ))
            });
            let mut state = AppState::model_free(backends)
                .with_installed_packages(inventory)
                .with_hardware(inventory_hardware_assessor)
                .with_model_evaluator(model_evaluator)
                .with_model_downloads(model_downloads)
                .with_hugging_face_catalog(previewer)
                .with_recommendable_catalog(recommendable_catalog)
                .with_identity(identity);
            if let Some(runtime) = runtime {
                state = state.with_runtime(runtime);
            }
            if let Some(auth_token) = auth_token {
                state = state.with_authorization(auth_token);
            }
            let listener = tokio::net::TcpListener::bind(bind)
                .await
                .with_context(|| format!("failed to bind {bind}"))?;
            let address = listener
                .local_addr()
                .context("failed to read bound address")?;
            let origin = format!("http://{address}");
            println!(
                "MAGNITUDE_ICN_READY {}",
                serde_json::json!({
                    "type": "icn_ready",
                    "protocolVersion": 1,
                    "origin": origin,
                    "instanceId": instance_id,
                    "pid": std::process::id(),
                    "apiVersion": 1,
                    "nativeBuild": native_build,
                })
            );
            tracing::info!(
                service.name = telemetry::SERVICE_NAME,
                server.address = %address,
                "ICN server ready"
            );
            let app = app(state).layer(
                TraceLayer::new_for_http()
                    .make_span_with(telemetry::http_request_span)
                    .on_response(DefaultOnResponse::new().level(tracing::Level::INFO)),
            );
            let serve_result = axum::serve(listener, app)
                .with_graceful_shutdown(shutdown_signal(parent_pid))
                .await;
            serve_result?;
            tracing::info!("ICN server stopped");
        }
        Command::Doctor => println!("ICN runtime and native backend loaded successfully"),
        Command::Version { json } => {
            if json {
                println!("{}", build_identity::json());
            } else {
                println!("{}", env!("CARGO_PKG_VERSION"));
            }
        }
        Command::PlanWorker => run_planning_worker()?,
        Command::TemplateWorker => run_template_worker()?,
    }
    Ok(())
}

async fn shutdown_signal(parent_pid: Option<u32>) {
    tokio::select! {
        _ = interrupt_signal() => {},
        _ = parent_watchdog(parent_pid), if parent_pid.is_some() => {},
        _ = parent_stdin_eof(), if parent_pid.is_some() => {},
    }
}

async fn parent_stdin_eof() {
    // Tokio implements stdin reads on its blocking pool. A pending read then
    // prevents Runtime::drop from completing during an ordinary SIGTERM while
    // the parent still owns the pipe, creating a parent/child shutdown cycle.
    // A detached OS thread has the desired semantics: EOF wakes the async
    // watchdog after abrupt parent death, while orderly process exit does not
    // wait for the read to finish.
    let (eof, observed) = tokio::sync::oneshot::channel();
    std::thread::spawn(move || {
        use std::io::Read as _;

        let mut stdin = std::io::stdin().lock();
        let mut buffer = [0_u8; 1];
        loop {
            match stdin.read(&mut buffer) {
                Ok(0) | Err(_) => break,
                Ok(_) => {}
            }
        }
        let _ = eof.send(());
    });
    let _ = observed.await;
}

#[cfg(unix)]
async fn interrupt_signal() {
    use tokio::signal::unix::{SignalKind, signal};
    let mut terminate = signal(SignalKind::terminate()).expect("SIGTERM handler must install");
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {},
        _ = terminate.recv() => {},
    }
}

#[cfg(not(unix))]
async fn interrupt_signal() {
    let _ = tokio::signal::ctrl_c().await;
}

async fn parent_watchdog(parent_pid: Option<u32>) {
    let Some(parent_pid) = parent_pid else {
        std::future::pending::<()>().await;
        return;
    };
    let mut interval = tokio::time::interval(std::time::Duration::from_secs(1));
    loop {
        interval.tick().await;
        if !process_exists(parent_pid) {
            return;
        }
    }
}

#[cfg(unix)]
fn process_exists(pid: u32) -> bool {
    // Signal zero performs an existence/permission check without delivering a signal.
    let result = unsafe { libc::kill(pid as libc::pid_t, 0) };
    result == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

#[cfg(not(unix))]
fn process_exists(_pid: u32) -> bool {
    true
}

fn spawn_blocking_traced<F, R>(operation: F) -> tokio::task::JoinHandle<R>
where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
{
    let span = tracing::Span::current();
    tokio::task::spawn_blocking(move || span.in_scope(operation))
}

#[cfg(test)]
mod tests {
    use super::*;
    use icn_contracts::ModelInventory as _;

    #[test]
    fn performance_evidence_preserves_the_exact_requested_context_and_bounds() {
        let evidence = performance_evidence(
            GenerationPerformanceAssessment::Estimated {
                method: "native".to_owned(),
                confidence: icn_contracts::GenerationPerformanceConfidence::Moderate,
                workload: "baseline_single_sequence_decode".to_owned(),
                always_active_weight_bytes: 10,
                routed_expert_weight_bytes: 80,
                expert_count: 8,
                expert_used_count: 2,
                cross_memory_domain_placement: true,
                points: vec![
                    icn_contracts::GenerationSpeedPoint {
                        context_tokens: 100_000,
                        kv_bytes_read_per_token: 4_096,
                        lower_tokens_per_second: 20.0,
                        expected_tokens_per_second: 24.0,
                        upper_tokens_per_second: 28.0,
                    },
                    icn_contracts::GenerationSpeedPoint {
                        context_tokens: 200_000,
                        kv_bytes_read_per_token: 8_192,
                        lower_tokens_per_second: 15.0,
                        expected_tokens_per_second: 18.0,
                        upper_tokens_per_second: 21.0,
                    },
                ],
            },
            100_000,
        )
        .expect("matching performance evidence");

        assert_eq!(evidence.context_tokens, 100_000);
        assert_eq!(evidence.lower_tokens_per_second, 20.0);
        assert_eq!(evidence.estimated_tokens_per_second, 24.0);
        assert_eq!(evidence.upper_tokens_per_second, 28.0);
        assert_eq!(evidence.confidence, PerformanceConfidence::Moderate);
        assert_eq!(evidence.method, "native");
    }

    fn parity_test_defaults() -> RuntimePlanDefaults {
        RuntimePlanDefaults {
            context_size: 128,
            batch_size: 128,
            ubatch_size: 64,
            max_sequences: 1,
            prefill_quantum: 128,
            execution: ExecutionConfig::default(),
            projector_use_gpu: true,
            projector_warmup: true,
            image_min_tokens: None,
            image_max_tokens: None,
        }
    }

    #[test]
    fn available_and_preview_cache_keys_share_resolved_profile_identity() {
        let assessor = NativeHardwareAssessor {
            defaults: parity_test_defaults(),
            native_backend: test_native_backend(),
            native_executor: Arc::new(RwLock::new(None)),
            gate: tokio::sync::Mutex::new(()),
            planning_slots: Arc::new(tokio::sync::Semaphore::new(1)),
            calibration: tokio::sync::Mutex::new(CalibrationCache::default()),
        };
        let snapshot = HardwareSnapshot {
            captured_at: 1,
            platform: "test".to_owned(),
            architecture: "test".to_owned(),
            cpu_model: None,
            logical_cores: 1,
            system_memory: icn_contracts::HardwareSystemMemory {
                total_bytes: 1,
                current_available_bytes: Some(1),
            },
            native_build: "native".to_owned(),
            enabled_backends: vec!["cpu".to_owned()],
            topology_fingerprint: "topology".to_owned(),
            memory_domains: Vec::new(),
            resident_memory: None,
        };
        let equivalent_preview = ModelPreviewProfile {
            id: "caller-correlation-does-not-affect-fit".to_owned(),
            context_length: 128,
            parallel_sequences: 1,
        };
        assert_eq!(
            assessor.assessment_cache_key(None, &snapshot).unwrap(),
            assessor
                .assessment_cache_key(Some(&equivalent_preview), &snapshot)
                .unwrap()
        );
        assert_ne!(
            assessor.assessment_cache_key(None, &snapshot).unwrap(),
            assessor
                .assessment_cache_key(
                    Some(&ModelPreviewProfile {
                        context_length: 4096,
                        ..equivalent_preview.clone()
                    }),
                    &snapshot,
                )
                .unwrap()
        );
    }

    fn sparse_header_copy(source: &std::path::Path, destination: &std::path::Path) {
        use std::io::{Read, Write};

        let inspection = icn_models::gguf::inspect(source).expect("inspect complete fixture");
        let header_bytes = usize::try_from(inspection.header_bytes).expect("header fits usize");
        let mut input = std::fs::File::open(source).expect("open complete fixture");
        let mut header = vec![0_u8; header_bytes];
        input.read_exact(&mut header).expect("read complete header");
        let mut output = std::fs::File::create(destination).expect("create sparse preview");
        output.write_all(&header).expect("write preview header");
        output
            .set_len(input.metadata().expect("fixture metadata").len())
            .expect("preserve preview logical length");
    }

    /// This exercises the exact native assessor used by both inventory and preview models. The
    /// verified parity fixtures are optional in ordinary source checkouts, but CI/dev environments
    /// that stage them exercise both a tiny dense model and a production-scale MoE model.
    #[tokio::test]
    async fn available_and_sparse_preview_artifacts_have_identical_fit_assessments() {
        let inference_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
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

        let assessor = Arc::new(NativeHardwareAssessor {
            defaults: parity_test_defaults(),
            native_backend: test_native_backend(),
            native_executor: Arc::new(RwLock::new(None)),
            gate: tokio::sync::Mutex::new(()),
            planning_slots: Arc::new(tokio::sync::Semaphore::new(1)),
            calibration: tokio::sync::Mutex::new(CalibrationCache::default()),
        });
        let profile = ModelPreviewProfile {
            id: "parity".to_owned(),
            context_length: 128,
            parallel_sequences: 1,
        };

        for fixture in fixtures {
            let store = tempfile::tempdir().expect("temporary model store");
            let mut config = InventoryConfig::with_root(store.path().join("inventory"))
                .expect("inventory config");
            config.model_sources = vec![fixture.parent().expect("fixture parent").to_path_buf()];
            config.hf_cache_dirs.clear();
            let manager = ModelManager::open_with_template_assessor(
                config,
                Some(Arc::new(NativeTemplateAssessor)),
            )
            .await
            .expect("open inventory");
            manager
                .set_hardware_assessor(assessor.clone())
                .expect("configure inventory assessor");
            manager
                .ensure_model_inventory()
                .await
                .expect("inspect available fixture");
            let model = manager
                .list()
                .await
                .expect("list inventory")
                .into_iter()
                .find(|model| {
                    model
                        .location
                        .components()
                        .iter()
                        .any(|component| component.path.file_name() == fixture.file_name())
                })
                .expect("fixture inventory model");
            let inventory_assessment = model.hardware.clone();
            let available = manager
                .resolve_ready(&model.id)
                .await
                .expect("resolve available fixture");

            let sparse_root = store.path().join("sparse-preview");
            std::fs::create_dir_all(&sparse_root).expect("create sparse preview directory");
            let mut preview = available.clone();
            for component in &mut preview.components {
                let destination =
                    sparse_root.join(component.path.file_name().expect("component file name"));
                sparse_header_copy(&component.path, &destination);
                component.path = destination;
            }

            let default_preview_assessment = assessor
                .assess_resolved(preview.clone(), None)
                .await
                .expect("assess sparse preview with inventory defaults");
            assert_eq!(
                default_preview_assessment,
                inventory_assessment,
                "the inventory and preview paths diverged for {}",
                fixture.display()
            );

            let available_assessment = assessor
                .assess_resolved(available, Some(&profile))
                .await
                .expect("assess available fixture");
            let preview_assessment = assessor
                .assess_resolved(preview, Some(&profile))
                .await
                .expect("assess sparse preview fixture");
            assert_eq!(
                preview_assessment,
                available_assessment,
                "preview and available fitting diverged for {}",
                fixture.display()
            );
        }
    }

    #[test]
    fn inventory_flag_aliases_parse() {
        let aliases = Cli::try_parse_from([
            "magnitude-icn",
            "serve",
            "--fake",
            "--models-dir",
            "/tmp/models",
            "--hf-cache-dir",
            "/tmp/hf",
        ])
        .expect("documented inventory flag aliases should parse");
        let Command::Serve {
            model_store,
            hf_caches,
            ..
        } = aliases.command
        else {
            panic!("expected serve command")
        };
        assert_eq!(model_store, Some(PathBuf::from("/tmp/models")));
        assert_eq!(hf_caches, vec![PathBuf::from("/tmp/hf")]);
    }

    #[test]
    fn version_json_reports_native_and_build_provenance() {
        let value = build_identity::json();
        assert_eq!(value["native_build"], build_identity::native_build());
        assert!(value.get("bindings_revision").is_none());
        assert!(value.get("native_backend_revision").is_none());
        assert_eq!(value["target"], build_identity::TARGET);
        assert_eq!(value["profile"], build_identity::PROFILE);
        assert!(
            value["backends"]
                .as_array()
                .is_some_and(|values| !values.is_empty())
        );
    }
}
