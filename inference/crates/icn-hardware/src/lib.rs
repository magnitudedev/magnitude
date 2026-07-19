//! Model-free memory fitting over the exact pinned llama.cpp `common/fit` path.

use std::ffi::{CString, NulError};
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

pub use icn_contracts::{CacheType, FlashAttention as FitFlashAttention, GpuLayers, SplitMode};
use llama_cpp_2::context::params::{
    FlashAttentionPolicy, KvCacheType, LlamaContextParams, LlamaContextType,
};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::params::fit::{
    FitDeviceEstimate, FitGpuLayers, FitMemoryEstimate, FitStatus,
};
use llama_cpp_2::model::params::fit::{FitReport, FitReportError};

use icn_contracts::{
    HardwareAssessment, HardwareDeficit, HardwareMemory, HardwareProfile, HardwareRecommendation,
    MtpConfig, MtpSource, ResolvedExecutionPlan,
};

fn cache_type_into_native(cache_type: CacheType) -> KvCacheType {
    match cache_type {
        CacheType::F32 => KvCacheType::F32,
        CacheType::F16 => KvCacheType::F16,
        CacheType::Bf16 => KvCacheType::BF16,
        CacheType::Q8_0 => KvCacheType::Q8_0,
        CacheType::Q4_0 => KvCacheType::Q4_0,
        CacheType::Q4_1 => KvCacheType::Q4_1,
        CacheType::Iq4Nl => KvCacheType::IQ4_NL,
        CacheType::Q5_0 => KvCacheType::Q5_0,
        CacheType::Q5_1 => KvCacheType::Q5_1,
    }
}

fn flash_attention_into_native(policy: FitFlashAttention) -> FlashAttentionPolicy {
    match policy {
        FitFlashAttention::Auto => FlashAttentionPolicy::Auto,
        FitFlashAttention::Disabled => FlashAttentionPolicy::Disabled,
        FitFlashAttention::Enabled => FlashAttentionPolicy::Enabled,
    }
}

/// Inputs that affect llama.cpp's model, context, and compute estimates.
#[derive(Clone, Debug, serde::Serialize)]
pub struct FitOptions {
    /// Context length. `None` requests the model's trained context length.
    pub context_tokens: Option<NonZeroU32>,
    /// Minimum context length allowed when fitting. `u32::MAX` preserves the full context.
    pub minimum_context_tokens: u32,
    /// One margin to broadcast or one value per `llama_max_devices()`, in bytes.
    pub margins_bytes: Vec<u64>,
    /// Logical prompt batch size.
    pub batch_tokens: u32,
    /// Physical prompt micro-batch size.
    pub micro_batch_tokens: u32,
    /// Maximum parallel sequences sharing the context.
    pub sequence_count: u32,
    /// `None` leaves GPU layers in auto mode; `Some` pins an explicit count.
    pub gpu_layers: GpuLayers,
    /// Model distribution strategy.
    pub split_mode: SplitMode,
    /// Explicit per-device proportions, if configured.
    pub tensor_split: Option<Vec<f32>>,
    /// Whether tensors may be memory mapped.
    pub use_mmap: bool,
    /// Whether model pages should be locked in memory.
    pub use_mlock: bool,
    /// K-cache data type.
    pub cache_type_k: CacheType,
    /// V-cache data type.
    pub cache_type_v: CacheType,
    /// Flash Attention policy.
    pub flash_attention: FitFlashAttention,
    /// Whether K/Q/V operations and KV memory may be offloaded.
    pub offload_kqv: bool,
    /// Whether host tensor operations may be offloaded.
    pub operation_offload: bool,
    /// Whether to allocate the full sliding-window cache.
    pub swa_full: bool,
    /// Whether sequences share a unified KV cache.
    pub kv_unified: bool,
    /// Native context role.
    pub context_type: FitContextType,
    /// Recurrent-state rollback snapshots retained per sequence.
    pub recurrent_snapshots: u32,
    /// Maximum logits outputs allocated by the context.
    pub maximum_outputs: Option<NonZeroU32>,
}

/// Native context role used by no-allocation planning.
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FitContextType {
    /// Ordinary target inference context.
    #[default]
    Target,
    /// Multi-token-prediction draft context.
    Mtp,
}

impl Default for FitOptions {
    fn default() -> Self {
        Self {
            context_tokens: None,
            minimum_context_tokens: 4_096,
            margins_bytes: vec![1024 * 1024 * 1024],
            batch_tokens: 2_048,
            micro_batch_tokens: 512,
            sequence_count: 1,
            gpu_layers: GpuLayers::Auto,
            split_mode: SplitMode::Layer,
            tensor_split: None,
            use_mmap: true,
            use_mlock: false,
            cache_type_k: CacheType::F16,
            cache_type_v: CacheType::F16,
            flash_attention: FitFlashAttention::Auto,
            offload_kqv: true,
            operation_offload: true,
            swa_full: false,
            kv_unified: false,
            context_type: FitContextType::Target,
            recurrent_snapshots: 0,
            maximum_outputs: None,
        }
    }
}

/// Request for a no-allocation model fit.
#[derive(Clone, Debug, serde::Serialize)]
pub struct FitRequest {
    /// GGUF file to inspect.
    pub model: PathBuf,
    /// Planning parameters.
    pub options: FitOptions,
}

/// Validation, backend, or native bridge failure.
#[derive(Debug, thiserror::Error)]
pub enum EstimateError {
    /// The model path is not a regular file.
    #[error("model does not exist or is not a file: {0}")]
    InvalidModel(PathBuf),
    /// The model path contains an interior NUL byte.
    #[error("model path contains an interior NUL byte: {0}")]
    ModelPathNul(#[from] NulError),
    /// Invalid fit option.
    #[error("invalid fit options: {0}")]
    InvalidOptions(String),
    /// llama.cpp backend initialization failed.
    #[error("failed to initialize llama.cpp: {0}")]
    Backend(#[source] llama_cpp_2::LlamaCppError),
    /// Structured fit bridge failed.
    #[error(transparent)]
    Fit(#[from] FitReportError),
}

/// Stable Magnitude capacity policy. It intentionally uses total capacity,
/// not volatile process-external free memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct CapacityPolicy {
    pub reserve_bytes_per_domain: u64,
}

impl Default for CapacityPolicy {
    fn default() -> Self {
        Self {
            reserve_bytes_per_domain: 1536 * 1024 * 1024,
        }
    }
}

/// The exact plan selected for loading plus its consumer-facing assessment.
#[derive(Clone, Debug)]
pub struct AssessedExecutionPlan {
    pub plan: ResolvedExecutionPlan,
    pub assessment: HardwareAssessment,
    pub text_report: FitReport,
    /// No-allocation report for the MTP context and optional companion model.
    pub mtp_report: Option<FitReport>,
    #[cfg(feature = "mtmd")]
    pub projector_memory: Vec<llama_cpp_2::mtmd::MtmdDeviceMemoryEstimate>,
}

#[derive(Debug, thiserror::Error)]
pub enum AssessmentError {
    #[error(transparent)]
    Estimate(#[from] EstimateError),
    #[error("projector assessment requires the icn-hardware mtmd feature")]
    ProjectorUnsupported,
    #[cfg(feature = "mtmd")]
    #[error("projector preflight failed: {0}")]
    Projector(#[from] llama_cpp_2::mtmd::MtmdPreflightError),
    #[error("the native estimator omitted required memory measurements")]
    MissingMeasurements,
    #[error(
        "the native fitter selected tensor placement overrides that the execution plan cannot represent"
    )]
    UnrepresentableTensorPlacement,
}

/// Assess and resolve exactly the configuration that the engine will load.
///
/// The preferred plan is evaluated from the native initial measurement against
/// stable total capacity. If it does not fit, a native fitted fallback is used
/// only when every selected placement is representable in the public plan.
pub fn resolve_and_assess(
    backend: &LlamaBackend,
    requested: &ResolvedExecutionPlan,
    policy: CapacityPolicy,
) -> Result<AssessedExecutionPlan, AssessmentError> {
    let text_report = estimate_with_backend(backend, &fit_request(requested, false)?)?;
    let mut mtp_report = estimate_mtp_report(backend, requested)?;
    let projector_memory = projector_memory(requested)?;

    let preferred = capacity_summary(
        &text_report.devices,
        Measurement::Initial,
        mtp_report.as_ref().map(|report| report.devices.as_slice()),
        mtp_includes_model(requested),
        &projector_memory,
        requested.radix_cache.host_bytes,
        policy,
    )?;
    if preferred.fits {
        let plan = resolved_plan(requested, &text_report, Measurement::Initial)?;
        return Ok(AssessedExecutionPlan {
            assessment: fits_assessment(&plan, &preferred, HardwareRecommendation::Recommended),
            plan,
            text_report,
            mtp_report,
            #[cfg(feature = "mtmd")]
            projector_memory,
        });
    }

    let fallback_plan = (text_report.status == FitStatus::Success)
        .then(|| resolved_plan(requested, &text_report, Measurement::Fitted))
        .transpose()?;
    let fallback = fallback_plan
        .as_ref()
        .map(|plan| {
            mtp_report = estimate_mtp_report(backend, plan)?;
            capacity_summary(
                &text_report.devices,
                Measurement::Fitted,
                mtp_report.as_ref().map(|report| report.devices.as_slice()),
                mtp_includes_model(plan),
                &projector_memory,
                plan.radix_cache.host_bytes,
                policy,
            )
        })
        .transpose()?;
    if fallback.as_ref().is_some_and(|summary| summary.fits) {
        if !text_report.tensor_placements.is_empty() {
            return Err(AssessmentError::UnrepresentableTensorPlacement);
        }
        let plan = fallback_plan.expect("a fallback summary has a plan");
        let summary = fallback.expect("checked above");
        return Ok(AssessedExecutionPlan {
            assessment: fits_assessment(&plan, &summary, HardwareRecommendation::Constrained),
            plan,
            text_report,
            mtp_report,
            #[cfg(feature = "mtmd")]
            projector_memory,
        });
    }

    let profile = hardware_profile(requested, &preferred);
    let assessment = HardwareAssessment::DoesNotFit {
        profile,
        memory: HardwareDeficit {
            required_bytes: preferred.required_bytes,
            available_bytes: preferred.available_bytes,
            deficit_bytes: preferred
                .required_bytes
                .saturating_sub(preferred.available_bytes),
        },
        limiting_resource: preferred.limiting_resource,
        alternative: None,
    };
    Ok(AssessedExecutionPlan {
        plan: requested.clone(),
        assessment,
        text_report,
        mtp_report,
        #[cfg(feature = "mtmd")]
        projector_memory,
    })
}

fn fit_request(
    plan: &ResolvedExecutionPlan,
    mtp_context: bool,
) -> Result<FitRequest, AssessmentError> {
    let (model, cache_type_k, cache_type_v, context_type, recurrent_snapshots, maximum_outputs) =
        if mtp_context {
            let MtpConfig::Enabled {
                source,
                cache_type_k,
                cache_type_v,
                ..
            } = &plan.mtp
            else {
                return Err(AssessmentError::MissingMeasurements);
            };
            let model = match source {
                MtpSource::Bundled => plan.model_path.clone(),
                MtpSource::Separate { model_path } => model_path.clone(),
            };
            (
                model,
                *cache_type_k,
                *cache_type_v,
                FitContextType::Mtp,
                0,
                NonZeroU32::new(plan.max_sequences),
            )
        } else {
            let (snapshots, outputs) = match plan.mtp {
                MtpConfig::Enabled { n_max, .. } => (
                    n_max,
                    NonZeroU32::new(
                        plan.max_sequences
                            .saturating_mul(n_max.saturating_add(1))
                            .min(plan.batch_size),
                    ),
                ),
                MtpConfig::Disabled { .. } => (0, None),
            };
            (
                plan.model_path.clone(),
                plan.execution.cache_type_k,
                plan.execution.cache_type_v,
                FitContextType::Target,
                snapshots,
                outputs,
            )
        };
    Ok(FitRequest {
        model,
        options: FitOptions {
            context_tokens: NonZeroU32::new(plan.context_size),
            minimum_context_tokens: 4_096.min(plan.context_size).max(1),
            margins_bytes: vec![0],
            batch_tokens: plan.batch_size,
            micro_batch_tokens: plan.ubatch_size,
            sequence_count: plan.max_sequences,
            gpu_layers: plan.execution.gpu_layers,
            split_mode: plan.execution.split_mode,
            tensor_split: plan.execution.tensor_split.clone(),
            use_mmap: plan.execution.use_mmap,
            use_mlock: plan.execution.use_mlock,
            cache_type_k,
            cache_type_v,
            flash_attention: plan.execution.flash_attention,
            offload_kqv: plan.execution.offload_kqv,
            operation_offload: plan.execution.operation_offload,
            swa_full: plan.execution.swa_full,
            kv_unified: plan.execution.kv_unified,
            context_type,
            recurrent_snapshots,
            maximum_outputs,
        },
    })
}

fn estimate_mtp_report(
    backend: &LlamaBackend,
    plan: &ResolvedExecutionPlan,
) -> Result<Option<FitReport>, AssessmentError> {
    match plan.mtp {
        MtpConfig::Disabled { .. } => Ok(None),
        MtpConfig::Enabled { .. } => Ok(Some(estimate_linked_with_backend(
            backend,
            &fit_request(plan, true)?,
            &fit_request(plan, false)?,
        )?)),
    }
}

fn mtp_includes_model(plan: &ResolvedExecutionPlan) -> bool {
    matches!(
        plan.mtp,
        MtpConfig::Enabled {
            source: MtpSource::Separate { .. },
            ..
        }
    )
}

/// Initialize the pinned backend and assess a resolved execution plan.
pub fn assess(
    requested: &ResolvedExecutionPlan,
    policy: CapacityPolicy,
) -> Result<AssessedExecutionPlan, AssessmentError> {
    let backend = LlamaBackend::init().map_err(EstimateError::Backend)?;
    assess_with_backend(&backend, requested, policy)
}

/// Assess a plan using an existing initialized llama.cpp backend.
///
/// Serving processes must use this entry point from their serialized native executor because
/// llama.cpp backend initialization is process-global and `common/fit` temporarily owns global
/// diagnostic state.
pub fn assess_with_backend(
    backend: &LlamaBackend,
    requested: &ResolvedExecutionPlan,
    policy: CapacityPolicy,
) -> Result<AssessedExecutionPlan, AssessmentError> {
    resolve_and_assess(backend, requested, policy)
}

#[derive(Clone, Copy)]
enum Measurement {
    Initial,
    Fitted,
}

#[cfg(feature = "mtmd")]
type ProjectorMemory = llama_cpp_2::mtmd::MtmdDeviceMemoryEstimate;

#[cfg(not(feature = "mtmd"))]
#[derive(Clone, Debug)]
struct ProjectorMemory;

fn projector_memory(plan: &ResolvedExecutionPlan) -> Result<Vec<ProjectorMemory>, AssessmentError> {
    let Some(projector) = plan.projector.as_ref() else {
        return Ok(Vec::new());
    };
    #[cfg(not(feature = "mtmd"))]
    {
        let _ = projector;
        Err(AssessmentError::ProjectorUnsupported)
    }
    #[cfg(feature = "mtmd")]
    {
        use llama_cpp_2::context::params::FlashAttentionPolicy;
        use llama_cpp_2::mtmd::{MtmdContextParams, mtmd_default_marker, mtmd_memory_usage};
        let mut params = MtmdContextParams {
            use_gpu: projector.use_gpu,
            warmup: projector.warmup,
            image_min_tokens: projector.image_min_tokens,
            image_max_tokens: projector.image_max_tokens,
            media_marker: CString::new(mtmd_default_marker()).expect("native marker has no NUL"),
            flash_attention: match plan.execution.flash_attention {
                FitFlashAttention::Auto => FlashAttentionPolicy::Auto,
                FitFlashAttention::Disabled => FlashAttentionPolicy::Disabled,
                FitFlashAttention::Enabled => FlashAttentionPolicy::Enabled,
            },
            ..MtmdContextParams::default()
        };
        if let Some(threads) = plan.execution.threads {
            params.n_threads = i32::try_from(threads.get()).unwrap_or(i32::MAX);
        }
        Ok(mtmd_memory_usage(&projector.path, &params)?)
    }
}

#[derive(Debug)]
struct CapacitySummary {
    fits: bool,
    required_bytes: u64,
    available_bytes: u64,
    limiting_resource: String,
    device: String,
}

#[derive(Debug)]
struct CapacityDomain {
    name: String,
    total_bytes: u64,
    required_bytes: u64,
}

fn capacity_summary(
    devices: &[FitDeviceEstimate],
    measurement: Measurement,
    mtp_devices: Option<&[FitDeviceEstimate]>,
    mtp_includes_model: bool,
    projectors: &[ProjectorMemory],
    host_cache_bytes: u64,
    policy: CapacityPolicy,
) -> Result<CapacitySummary, AssessmentError> {
    #[cfg(not(feature = "mtmd"))]
    let _ = projectors;
    let unified = cfg!(target_os = "macos")
        && devices
            .iter()
            .any(|device| device.name.to_ascii_lowercase().contains("metal"));
    let mut domains = Vec::<CapacityDomain>::new();
    for device in devices {
        let estimate = match measurement {
            Measurement::Initial => device.initial,
            Measurement::Fitted => device.fitted,
        };
        let Some(estimate) = estimate else {
            continue;
        };
        add_device_estimate(&mut domains, device, estimate, unified)?;
    }
    if let Some(mtp_devices) = mtp_devices {
        for device in mtp_devices {
            let Some(estimate) = device.initial else {
                continue;
            };
            add_additional_device_estimate(
                &mut domains,
                device,
                estimate,
                unified,
                mtp_includes_model,
            )?;
        }
    }
    #[cfg(feature = "mtmd")]
    for projector in projectors {
        if unified {
            if let Some(domain) = domains.first_mut() {
                domain.required_bytes = domain.required_bytes.saturating_add(projector.bytes);
            }
        } else if let Some(domain_index) = projector
            .device_index
            .filter(|index| *index < domains.len())
            .or_else(|| {
                domains
                    .iter()
                    .position(|domain| domain.name == projector.device_name)
            })
        {
            let domain = &mut domains[domain_index];
            domain.required_bytes = domain.required_bytes.saturating_add(projector.bytes);
        } else {
            return Err(AssessmentError::MissingMeasurements);
        }
    }
    if domains.is_empty() {
        return Err(AssessmentError::MissingMeasurements);
    }
    if host_cache_bytes > 0 {
        let host_index = if unified {
            0
        } else {
            domains
                .iter()
                .position(|domain| domain.name.to_ascii_lowercase().contains("cpu"))
                .unwrap_or(0)
        };
        let host_domain = &mut domains[host_index];
        host_domain.required_bytes = host_domain.required_bytes.saturating_add(host_cache_bytes);
    }
    let mut required_bytes = 0_u64;
    let mut available_bytes = 0_u64;
    let mut limiting_resource = domains[0].name.clone();
    let mut largest_deficit = 0_u64;
    let mut fits = true;
    for domain in &domains {
        let available = domain
            .total_bytes
            .saturating_sub(policy.reserve_bytes_per_domain);
        let deficit = domain.required_bytes.saturating_sub(available);
        if deficit > 0 {
            fits = false;
        }
        if deficit >= largest_deficit {
            largest_deficit = deficit;
            limiting_resource.clone_from(&domain.name);
        }
        required_bytes = required_bytes.saturating_add(domain.required_bytes);
        available_bytes = available_bytes.saturating_add(available);
    }
    Ok(CapacitySummary {
        fits,
        required_bytes,
        available_bytes,
        limiting_resource,
        device: domains
            .iter()
            .map(|domain| domain.name.as_str())
            .collect::<Vec<_>>()
            .join(" + "),
    })
}

fn add_additional_device_estimate(
    domains: &mut [CapacityDomain],
    device: &FitDeviceEstimate,
    estimate: FitMemoryEstimate,
    unified: bool,
    include_model: bool,
) -> Result<(), AssessmentError> {
    let bytes = estimate
        .allocations
        .context_bytes
        .saturating_add(estimate.allocations.compute_bytes)
        .saturating_add(if include_model {
            estimate.allocations.model_bytes
        } else {
            0
        });
    let domain = if unified {
        domains.first_mut()
    } else {
        domains.iter_mut().find(|domain| domain.name == device.name)
    }
    .ok_or(AssessmentError::MissingMeasurements)?;
    domain.required_bytes = domain.required_bytes.saturating_add(bytes);
    Ok(())
}

fn add_device_estimate(
    domains: &mut Vec<CapacityDomain>,
    device: &FitDeviceEstimate,
    estimate: FitMemoryEstimate,
    unified: bool,
) -> Result<(), AssessmentError> {
    let total_bytes =
        u64::try_from(estimate.total_bytes).map_err(|_| AssessmentError::MissingMeasurements)?;
    if unified {
        if let Some(domain) = domains.first_mut() {
            domain.total_bytes = domain.total_bytes.max(total_bytes);
            domain.required_bytes = domain
                .required_bytes
                .saturating_add(estimate.allocations.total_bytes);
        } else {
            domains.push(CapacityDomain {
                name: "metal_unified_memory".to_owned(),
                total_bytes,
                required_bytes: estimate.allocations.total_bytes,
            });
        }
    } else {
        domains.push(CapacityDomain {
            name: device.name.clone(),
            total_bytes,
            required_bytes: estimate.allocations.total_bytes,
        });
    }
    Ok(())
}

fn resolved_plan(
    requested: &ResolvedExecutionPlan,
    report: &FitReport,
    measurement: Measurement,
) -> Result<ResolvedExecutionPlan, AssessmentError> {
    let configuration = match measurement {
        Measurement::Initial => report.requested,
        Measurement::Fitted => report.fitted,
    };
    let mut plan = requested.clone();
    plan.context_size = configuration.resolved_context_tokens;
    plan.execution.gpu_layers = match configuration.gpu_layers {
        FitGpuLayers::Count(value) => GpuLayers::Count(value),
        FitGpuLayers::All => GpuLayers::All,
        FitGpuLayers::Auto => GpuLayers::Count(configuration.resolved_gpu_layers),
    };
    if matches!(measurement, Measurement::Fitted) && !report.tensor_split.is_empty() {
        plan.execution.tensor_split = Some(report.tensor_split.clone());
    }
    Ok(plan)
}

fn hardware_profile(plan: &ResolvedExecutionPlan, summary: &CapacitySummary) -> HardwareProfile {
    HardwareProfile {
        context_length: plan.context_size,
        acceleration: if matches!(plan.execution.gpu_layers, GpuLayers::Count(0)) {
            "cpu".to_owned()
        } else {
            "accelerated".to_owned()
        },
        device: summary.device.clone(),
    }
}

fn fits_assessment(
    plan: &ResolvedExecutionPlan,
    summary: &CapacitySummary,
    recommendation: HardwareRecommendation,
) -> HardwareAssessment {
    HardwareAssessment::Fits {
        profile: hardware_profile(plan, summary),
        memory: HardwareMemory {
            required_bytes: summary.required_bytes,
            available_bytes: summary.available_bytes,
            headroom_bytes: summary
                .available_bytes
                .saturating_sub(summary.required_bytes),
        },
        recommendation,
    }
}

/// Initialize llama.cpp and estimate a model without allocating its tensor data.
///
/// # Errors
///
/// Returns [`EstimateError`] if validation, backend initialization, or native
/// diagnostics fail. A native fit `Failure`/`Error` is still returned as a
/// typed [`FitReport`] so callers can inspect its diagnostics.
pub fn estimate(request: &FitRequest) -> Result<FitReport, EstimateError> {
    let backend = LlamaBackend::init().map_err(EstimateError::Backend)?;
    estimate_with_backend(&backend, request)
}

/// Estimate a model using an already initialized llama.cpp backend.
///
/// The backend reference is an explicit lifetime proof for callers such as ICN;
/// the pinned C fitting function itself uses global backend registration.
///
/// # Errors
///
/// Returns [`EstimateError`] for invalid options or bridge/report failures.
pub fn estimate_with_backend(
    backend: &LlamaBackend,
    request: &FitRequest,
) -> Result<FitReport, EstimateError> {
    estimate_impl(backend, request, None)
}

/// Estimate an MTP model/context linked to the exact target execution context.
pub fn estimate_linked_with_backend(
    backend: &LlamaBackend,
    request: &FitRequest,
    target: &FitRequest,
) -> Result<FitReport, EstimateError> {
    estimate_impl(backend, request, Some(target))
}

fn estimate_impl(
    _backend: &LlamaBackend,
    request: &FitRequest,
    linked_target: Option<&FitRequest>,
) -> Result<FitReport, EstimateError> {
    validate(request)?;
    let model_path = path_c_string(&request.model)?;
    let max_devices = llama_cpp_2::max_devices();
    let mut margins = expand_margins(&request.options.margins_bytes, max_devices)?;

    let model_params = native_model_params(&request.options)?;
    let mut model_params = std::pin::pin!(model_params);
    let mut context_params = native_context_params(&request.options);

    if let Some(target) = linked_target {
        validate(target)?;
        let target_path = path_c_string(&target.model)?;
        let target_model_params = native_model_params(&target.options)?;
        let target_context_params = native_context_params(&target.options);
        return model_params
            .as_mut()
            .fit_params_report_linked(
                &model_path,
                &mut context_params,
                llama_cpp_2::model::params::fit::LinkedFitTarget {
                    model_path: &target_path,
                    model_params: &target_model_params,
                    context_params: &target_context_params,
                },
                &mut margins,
                request.options.minimum_context_tokens,
            )
            .map_err(EstimateError::Fit);
    }

    model_params
        .as_mut()
        .fit_params_report(
            &model_path,
            &mut context_params,
            &mut margins,
            request.options.minimum_context_tokens,
        )
        .map_err(EstimateError::Fit)
}

fn native_model_params(options: &FitOptions) -> Result<LlamaModelParams, EstimateError> {
    let params = LlamaModelParams::default()
        .with_gpu_layers(match options.gpu_layers {
            GpuLayers::Auto => llama_cpp_2::model::params::LlamaGpuLayers::Auto,
            GpuLayers::All => llama_cpp_2::model::params::LlamaGpuLayers::All,
            GpuLayers::Count(value) => llama_cpp_2::model::params::LlamaGpuLayers::Count(value),
        })
        .with_use_mmap(options.use_mmap)
        .with_use_mlock(options.use_mlock)
        .with_split_mode(match options.split_mode {
            SplitMode::None => llama_cpp_2::model::params::LlamaSplitMode::None,
            SplitMode::Layer => llama_cpp_2::model::params::LlamaSplitMode::Layer,
            SplitMode::Row => llama_cpp_2::model::params::LlamaSplitMode::Row,
            SplitMode::Tensor => llama_cpp_2::model::params::LlamaSplitMode::Tensor,
        });
    match &options.tensor_split {
        Some(weights) => params
            .with_tensor_split(weights)
            .map_err(|error| EstimateError::InvalidOptions(error.to_string())),
        None => Ok(params),
    }
}

fn native_context_params(options: &FitOptions) -> LlamaContextParams {
    LlamaContextParams::default()
        .with_n_ctx(options.context_tokens)
        .with_n_batch(options.batch_tokens)
        .with_n_ubatch(options.micro_batch_tokens)
        .with_n_seq_max(options.sequence_count)
        .with_type_k(cache_type_into_native(options.cache_type_k))
        .with_type_v(cache_type_into_native(options.cache_type_v))
        .with_flash_attention(flash_attention_into_native(options.flash_attention))
        .with_offload_kqv(options.offload_kqv)
        .with_op_offload(options.operation_offload)
        .with_swa_full(options.swa_full)
        .with_kv_unified(options.kv_unified)
        .with_context_type(match options.context_type {
            FitContextType::Target => LlamaContextType::Default,
            FitContextType::Mtp => LlamaContextType::Mtp,
        })
        .with_n_rs_seq(options.recurrent_snapshots)
        .with_n_outputs_max(options.maximum_outputs)
}

fn validate(request: &FitRequest) -> Result<(), EstimateError> {
    if !request.model.is_file() {
        return Err(EstimateError::InvalidModel(request.model.clone()));
    }
    let options = &request.options;
    if options.minimum_context_tokens == 0 {
        return Err(EstimateError::InvalidOptions(
            "minimum context must be greater than zero".to_owned(),
        ));
    }
    if options.batch_tokens == 0 || options.micro_batch_tokens == 0 {
        return Err(EstimateError::InvalidOptions(
            "batch and micro-batch sizes must be greater than zero".to_owned(),
        ));
    }
    if options.micro_batch_tokens > options.batch_tokens {
        return Err(EstimateError::InvalidOptions(
            "micro-batch size must not exceed batch size".to_owned(),
        ));
    }
    if options.sequence_count == 0 {
        return Err(EstimateError::InvalidOptions(
            "sequence count must be greater than zero".to_owned(),
        ));
    }
    if options
        .maximum_outputs
        .is_some_and(|outputs| outputs.get() > options.batch_tokens)
    {
        return Err(EstimateError::InvalidOptions(
            "maximum outputs must not exceed batch size".to_owned(),
        ));
    }
    if options.margins_bytes.is_empty() {
        return Err(EstimateError::InvalidOptions(
            "at least one memory margin is required".to_owned(),
        ));
    }
    if options.tensor_split.as_ref().is_some_and(|weights| {
        weights.is_empty()
            || weights
                .iter()
                .any(|weight| !weight.is_finite() || *weight < 0.0)
    }) {
        return Err(EstimateError::InvalidOptions(
            "tensor_split must contain finite, non-negative weights".to_owned(),
        ));
    }
    Ok(())
}

fn expand_margins(values: &[u64], count: usize) -> Result<Vec<usize>, EstimateError> {
    if values.len() != 1 && values.len() != count {
        return Err(EstimateError::InvalidOptions(format!(
            "provide one memory margin to broadcast or exactly {count}; received {}",
            values.len()
        )));
    }
    let convert = |value: u64| {
        usize::try_from(value).map_err(|_| {
            EstimateError::InvalidOptions(format!(
                "memory margin {value} does not fit this target's usize"
            ))
        })
    };
    if values.len() == 1 {
        return Ok(vec![convert(values[0])?; count]);
    }
    values.iter().copied().map(convert).collect()
}

fn path_c_string(path: &Path) -> Result<CString, EstimateError> {
    Ok(CString::new(path.to_string_lossy().as_bytes())?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use llama_cpp_2::model::params::fit::{FitAllocations, FitDeviceKind, FitMemoryEstimate};

    #[test]
    fn one_margin_is_broadcast() {
        assert_eq!(expand_margins(&[512], 3).expect("margins"), vec![512; 3]);
    }

    #[test]
    fn per_device_margin_count_is_exact() {
        assert!(expand_margins(&[1, 2], 3).is_err());
        assert_eq!(
            expand_margins(&[1, 2, 3], 3).expect("margins"),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn cache_types_match_upstream_spelling() {
        assert_eq!("iq4_nl".parse::<CacheType>(), Ok(CacheType::Iq4Nl));
        assert!("q6_k".parse::<CacheType>().is_err());
    }

    #[test]
    fn stable_capacity_ignores_volatile_free_memory() {
        let device = FitDeviceEstimate {
            index: 0,
            kind: FitDeviceKind::Host,
            backend_type: 0,
            name: "cpu".to_owned(),
            description: "host".to_owned(),
            initial: Some(FitMemoryEstimate {
                total_bytes: 1_000,
                free_bytes: 1,
                allocations: FitAllocations {
                    model_bytes: 300,
                    context_bytes: 50,
                    compute_bytes: 50,
                    total_bytes: 400,
                },
                target: None,
            }),
            fitted: None,
            margin_bytes: None,
        };
        let summary = capacity_summary(
            &[device],
            Measurement::Initial,
            None,
            false,
            &[],
            100,
            CapacityPolicy {
                reserve_bytes_per_domain: 100,
            },
        )
        .unwrap();
        assert!(summary.fits);
        assert_eq!(summary.available_bytes, 900);
        assert_eq!(summary.required_bytes, 500);
    }

    #[test]
    fn built_in_mtp_adds_context_and_compute_but_not_duplicate_model_bytes() {
        let device = |model_bytes, context_bytes, compute_bytes| FitDeviceEstimate {
            index: 0,
            kind: FitDeviceKind::Host,
            backend_type: 0,
            name: "cpu".to_owned(),
            description: "host".to_owned(),
            initial: Some(FitMemoryEstimate {
                total_bytes: 2_000,
                free_bytes: 1,
                allocations: FitAllocations {
                    model_bytes,
                    context_bytes,
                    compute_bytes,
                    total_bytes: model_bytes + context_bytes + compute_bytes,
                },
                target: None,
            }),
            fitted: None,
            margin_bytes: None,
        };
        let target = device(500, 100, 50);
        let mtp = device(500, 40, 10);
        let summary = capacity_summary(
            &[target],
            Measurement::Initial,
            Some(&[mtp]),
            false,
            &[],
            0,
            CapacityPolicy {
                reserve_bytes_per_domain: 0,
            },
        )
        .unwrap();
        assert_eq!(summary.required_bytes, 700);
    }
}
