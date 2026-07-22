//! Model-free memory fitting over the exact pinned llama.cpp `common/fit` path.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{CString, NulError};
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub use icn_contracts::{CacheType, FlashAttention as FitFlashAttention, GpuLayers, SplitMode};
use llama_cpp_2::context::params::{
    FlashAttentionPolicy, KvCacheType, LlamaContextParams, LlamaContextType,
};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::params::fit::{
    FitCalibration, FitCalibrationMetric, FitDecodeWorkload, FitDecodeWorkloadAssessment,
    FitDeviceEstimate, FitDeviceKind, FitMemoryEstimate, FitStatus, FitTensorWorkloadKind,
};
use llama_cpp_2::model::params::fit::{FitReport, FitReportError};

use icn_contracts::{
    ExecutionIntent, GenerationPerformanceAssessment, GenerationPerformanceConfidence,
    GenerationSpeedPoint, HardwareAssessment, HardwareDeficit, HardwareDevice, HardwareDeviceKind,
    HardwareDeviceMemoryAssessment, HardwareDeviceMemoryLimit, HardwareDeviceMemoryLimitKind,
    HardwareMemory, HardwareMemoryDomain, HardwareMemoryDomainAssessment, HardwareMemoryDomainKind,
    HardwareProfile, HardwareRecommendation, HardwareSnapshot, HardwareSystemMemory,
    ModelExecutionAssessment, MtpConfig, MtpSource,
};
use llama_cpp_2::LlamaBackendDeviceType;
use sha2::{Digest, Sha256};
use sysinfo::{MemoryRefreshKind, RefreshKind, System};

pub const GENERATION_PERFORMANCE_CONTEXTS: [u32; 4] = [8_192, 32_768, 100_000, 200_000];
pub const GENERATION_PERFORMANCE_METHOD: &str = "icn-hardware-calibrated-decode-v2";
// Versioned ICN policy for work not represented by the synthetic matrix-operation calibration.
// Changing any of these constants requires a new GENERATION_PERFORMANCE_METHOD identity.
const GENERATION_PERFORMANCE_WORKLOAD: &str = "baseline_single_sequence_decode";
const DENSE_DECODE_EFFICIENCY: f64 = 0.82;
const ROUTED_DECODE_EFFICIENCY: f64 = 0.75;
const CROSS_DOMAIN_PLACEMENT_EFFICIENCY: f64 = 0.88;
const CALIBRATION_SPREAD_WEIGHT: f64 = 1.5;
const MINIMUM_UNCERTAINTY: f64 = 0.12;
const MAXIMUM_CALIBRATION_UNCERTAINTY: f64 = 0.45;
const ROUTING_UNCERTAINTY: f64 = 0.08;
const MAXIMUM_ROUTED_UNCERTAINTY: f64 = 0.55;
const CROSS_DOMAIN_PLACEMENT_UNCERTAINTY: f64 = 0.12;
const MAXIMUM_CROSS_DOMAIN_UNCERTAINTY: f64 = 0.65;
const UPPER_BOUND_UNCERTAINTY_WEIGHT: f64 = 0.65;

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
    /// Explicit generation thread count. `None` uses the pinned native common default.
    pub threads: Option<NonZeroU32>,
    /// Explicit prompt thread count. `None` reuses the resolved generation count.
    pub threads_batch: Option<NonZeroU32>,
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
            threads: None,
            threads_batch: None,
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

#[derive(Clone, Debug)]
struct DiscoveredDevice {
    native_index: usize,
    backend: String,
    physical_id: Option<String>,
    name: String,
    description: String,
    kind: HardwareDeviceKind,
    total_bytes: u64,
    free_bytes: Option<u64>,
}

struct HardwareEnvironment {
    native_build: String,
    enabled_backends: Vec<String>,
    platform: String,
    architecture: String,
    logical_cores: usize,
    system_memory: HardwareSystemMemory,
}

impl Default for CapacityPolicy {
    fn default() -> Self {
        Self {
            reserve_bytes_per_domain: 1536 * 1024 * 1024,
        }
    }
}

/// Discover the non-overlapping memory domains exposed by the pinned native runtime.
#[must_use]
pub fn discover_hardware(
    _backend: &LlamaBackend,
    policy: CapacityPolicy,
    native_build: impl Into<String>,
    enabled_backends: Vec<String>,
) -> HardwareSnapshot {
    let mut system = System::new_with_specifics(
        RefreshKind::nothing().with_memory(MemoryRefreshKind::everything()),
    );
    system.refresh_memory();
    let mut system_memory = HardwareSystemMemory {
        total_bytes: system.total_memory(),
        current_available_bytes: Some(system.available_memory()),
    };
    let devices = llama_cpp_2::list_llama_ggml_backend_devices()
        .into_iter()
        .map(|device| DiscoveredDevice {
            native_index: device.index,
            backend: device.backend,
            physical_id: device.device_id,
            name: device.name,
            description: device.description,
            kind: match device.device_type {
                LlamaBackendDeviceType::Cpu => HardwareDeviceKind::Cpu,
                LlamaBackendDeviceType::Gpu => HardwareDeviceKind::Gpu,
                LlamaBackendDeviceType::IntegratedGpu => HardwareDeviceKind::IntegratedGpu,
                LlamaBackendDeviceType::Accelerator => HardwareDeviceKind::Accelerator,
                _ => HardwareDeviceKind::Unknown,
            },
            total_bytes: u64::try_from(device.memory_total).unwrap_or(u64::MAX),
            free_bytes: u64::try_from(device.memory_free).ok(),
        })
        .collect::<Vec<_>>();
    if system_memory.total_bytes == 0 {
        system_memory.total_bytes = devices
            .iter()
            .filter(|device| device.kind == HardwareDeviceKind::Cpu)
            .map(|device| device.total_bytes)
            .max()
            .unwrap_or(0);
    }
    hardware_snapshot_from_devices(
        devices,
        policy,
        HardwareEnvironment {
            native_build: native_build.into(),
            enabled_backends,
            platform: std::env::consts::OS.to_owned(),
            architecture: std::env::consts::ARCH.to_owned(),
            logical_cores: std::thread::available_parallelism().map_or(1, |value| value.get()),
            system_memory,
        },
    )
}

fn hardware_snapshot_from_devices(
    devices: Vec<DiscoveredDevice>,
    policy: CapacityPolicy,
    mut environment: HardwareEnvironment,
) -> HardwareSnapshot {
    let unified_platform = environment.platform == "macos" && environment.architecture == "aarch64";
    let mut shared = Vec::new();
    let mut dedicated = BTreeMap::<String, BTreeMap<String, Vec<DiscoveredDevice>>>::new();
    for device in devices {
        if unified_platform
            || matches!(
                device.kind,
                HardwareDeviceKind::Cpu | HardwareDeviceKind::IntegratedGpu
            )
        {
            shared.push(device);
        } else {
            let physical_key = device.physical_id.clone().map_or_else(
                || format!("backend:{}:{}", device.backend, device.native_index),
                |id| format!("physical:{id}"),
            );
            dedicated
                .entry(physical_key)
                .or_default()
                .entry(device.backend.to_ascii_lowercase())
                .or_default()
                .push(device);
        }
    }
    shared.sort_by(device_order);
    for backends in dedicated.values_mut() {
        for views in backends.values_mut() {
            views.sort_by(device_order);
        }
    }

    let mut domains = Vec::new();
    if !shared.is_empty() || environment.system_memory.total_bytes > 0 {
        let backend_total = shared
            .iter()
            .map(|device| device.total_bytes)
            .max()
            .unwrap_or(0);
        let total = if environment.system_memory.total_bytes > 0 {
            environment.system_memory.total_bytes
        } else {
            backend_total
        };
        let unified = unified_platform
            || shared
                .iter()
                .any(|device| device.kind == HardwareDeviceKind::IntegratedGpu);
        domains.push(HardwareMemoryDomain {
            id: "system".to_owned(),
            kind: if unified {
                HardwareMemoryDomainKind::UnifiedMemory
            } else {
                HardwareMemoryDomainKind::System
            },
            total_capacity_bytes: total,
            stable_capacity_bytes: total.saturating_sub(policy.reserve_bytes_per_domain),
            current_free_bytes: environment.system_memory.current_available_bytes,
            shares_system_memory: true,
            devices: shared
                .into_iter()
                .enumerate()
                .map(|(ordinal, device)| public_device(device, ordinal, unified_platform, policy))
                .collect(),
        });
    }

    // A physical accelerator can be exposed by more than one backend. Merge views only when the
    // backend reports the same exact physical identity. An id-less view remains backend-scoped;
    // display strings and capacities are never treated as identity evidence.
    for (physical_key, backends) in dedicated {
        let occurrences = backends.values().map(Vec::len).max().unwrap_or(0);
        for ordinal in 0..occurrences {
            let views = backends
                .values()
                .filter_map(|devices| devices.get(ordinal).cloned())
                .collect::<Vec<_>>();
            let total = views
                .iter()
                .map(|device| device.total_bytes)
                .max()
                .unwrap_or(0);
            if total == 0 {
                continue;
            }
            let free = views.iter().filter_map(|device| device.free_bytes).max();
            let id_material = format!("{physical_key}\0{ordinal}");
            let id = format!("device-{:x}", Sha256::digest(id_material.as_bytes()));
            domains.push(HardwareMemoryDomain {
                id,
                kind: HardwareMemoryDomainKind::PhysicalDevice,
                total_capacity_bytes: total,
                stable_capacity_bytes: total.saturating_sub(policy.reserve_bytes_per_domain),
                current_free_bytes: free,
                shares_system_memory: false,
                devices: views
                    .into_iter()
                    .map(|device| public_device(device, ordinal, false, policy))
                    .collect(),
            });
        }
    }

    environment.enabled_backends.sort();
    environment.enabled_backends.dedup();

    let topology_material = domains
        .iter()
        .map(|domain| {
            (
                &domain.id,
                &domain.kind,
                domain.total_capacity_bytes,
                domain.stable_capacity_bytes,
                domain.shares_system_memory,
                domain
                    .devices
                    .iter()
                    .map(|device| {
                        (
                            &device.backend,
                            &device.name,
                            &device.description,
                            &device.kind,
                            device
                                .memory_limit
                                .as_ref()
                                .map(|limit| (&limit.kind, limit.total_bytes, limit.stable_bytes)),
                        )
                    })
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    let topology_material = serde_json::to_vec(&topology_material).unwrap_or_default();
    let topology_fingerprint = format!("{:x}", Sha256::digest(topology_material));
    HardwareSnapshot {
        captured_at: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_secs()),
        platform: environment.platform,
        architecture: environment.architecture,
        cpu_model: domains
            .iter()
            .flat_map(|domain| &domain.devices)
            .find(|device| device.kind == HardwareDeviceKind::Cpu)
            .map(|device| device.description.clone()),
        logical_cores: environment.logical_cores,
        system_memory: environment.system_memory,
        native_build: environment.native_build,
        enabled_backends: environment.enabled_backends,
        topology_fingerprint,
        memory_domains: domains,
        resident_memory: None,
    }
}

fn device_order(left: &DiscoveredDevice, right: &DiscoveredDevice) -> std::cmp::Ordering {
    (
        left.physical_id.as_deref().unwrap_or(""),
        left.backend.to_ascii_lowercase(),
        left.native_index,
        left.name.to_ascii_lowercase(),
    )
        .cmp(&(
            right.physical_id.as_deref().unwrap_or(""),
            right.backend.to_ascii_lowercase(),
            right.native_index,
            right.name.to_ascii_lowercase(),
        ))
}

fn public_device(
    device: DiscoveredDevice,
    ordinal: usize,
    apple_unified: bool,
    policy: CapacityPolicy,
) -> HardwareDevice {
    let identity = format!(
        "{}\0{}\0{}\0{}\0{ordinal}",
        device.backend,
        device.physical_id.as_deref().unwrap_or(""),
        device.native_index,
        device.name
    );
    let memory_limit = (apple_unified
        && device.kind != HardwareDeviceKind::Cpu
        && device.total_bytes > 0)
        .then(|| HardwareDeviceMemoryLimit {
            kind: HardwareDeviceMemoryLimitKind::RecommendedWorkingSet,
            total_bytes: device.total_bytes,
            stable_bytes: device
                .total_bytes
                .saturating_sub(policy.reserve_bytes_per_domain),
            current_free_bytes: device.free_bytes,
        });
    HardwareDevice {
        id: format!("native-{:x}", Sha256::digest(identity.as_bytes())),
        native_index: device.native_index,
        backend: device.backend,
        physical_id: device.physical_id,
        name: device.name,
        description: device.description,
        kind: device.kind,
        memory_limit,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NativeMemoryLocation {
    Host,
    Device {
        backend: String,
        physical_id: Option<String>,
        native_index: usize,
    },
}

/// Resolve native allocation identity through the same logical-device topology used for fitting.
/// Display names and capacity similarity are intentionally not identity evidence.
pub fn resolve_memory_domain<'a>(
    snapshot: &'a HardwareSnapshot,
    location: &NativeMemoryLocation,
) -> Option<&'a str> {
    match location {
        NativeMemoryLocation::Host => snapshot
            .memory_domains
            .iter()
            .find(|domain| domain.shares_system_memory)
            .map(|domain| domain.id.as_str()),
        NativeMemoryLocation::Device {
            backend,
            physical_id,
            native_index,
        } => snapshot.memory_domains.iter().find_map(|domain| {
            domain
                .devices
                .iter()
                .any(
                    |device| match (physical_id.as_deref(), device.physical_id.as_deref()) {
                        (Some(expected), Some(actual)) => {
                            device.backend.eq_ignore_ascii_case(backend) && expected == actual
                        }
                        (None, _) => {
                            *native_index == device.native_index
                                && (backend.is_empty()
                                    || device.backend.eq_ignore_ascii_case(backend))
                        }
                        _ => false,
                    },
                )
                .then_some(domain.id.as_str())
        }),
    }
}

/// The exact plan selected for loading plus its consumer-facing assessment.
#[derive(Clone, Debug)]
pub struct AssessedExecutionPlan {
    pub plan: ExecutionIntent,
    pub assessment: HardwareAssessment,
    pub text_report: FitReport,
    /// No-allocation report for the MTP context and optional companion model.
    pub mtp_report: Option<FitReport>,
    #[cfg(feature = "mtmd")]
    pub projector_memory: Vec<llama_cpp_2::mtmd::MtmdDeviceMemoryEstimate>,
}

/// Process-local backend plan. Its fitted native parameters are consumed directly by loading and
/// are never serialized or persisted.
pub struct BackendLoadPlan {
    pub assessed: AssessedExecutionPlan,
    pub native: NativeParameterPlan,
    pub native_mtp: Option<NativeParameterPlan>,
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
    #[error("artifact is incompatible with the pinned native backend: {code}: {message}")]
    IncompatibleArtifact { code: String, message: String },
    #[error("artifact is invalid: {code}: {message}")]
    InvalidArtifact { code: String, message: String },
}

/// Assess execution intent using the same native planning implementation used by loading.
/// Preview retains only normalized evidence; it never projects native placement back into intent.
pub fn assess_intent_with_backend(
    backend: &LlamaBackend,
    requested: &ExecutionIntent,
    policy: CapacityPolicy,
) -> Result<AssessedExecutionPlan, AssessmentError> {
    Ok(plan_and_assess(backend, requested, policy, false)?.0)
}

fn assess_intent_with_decode_workload(
    backend: &LlamaBackend,
    requested: &ExecutionIntent,
    policy: CapacityPolicy,
) -> Result<AssessedExecutionPlan, AssessmentError> {
    Ok(plan_and_assess(backend, requested, policy, true)?.0)
}

/// Plan a load and retain the exact fitted native parameter object that produced its assessment.
pub fn plan_load_with_backend(
    backend: &LlamaBackend,
    requested: &ExecutionIntent,
    policy: CapacityPolicy,
) -> Result<BackendLoadPlan, AssessmentError> {
    let (assessed, native) = plan_and_assess(backend, requested, policy, false)?;
    let native = match (native, &assessed.assessment) {
        (Some(native), _) => native,
        (None, HardwareAssessment::IncompatibleArtifact { code, message }) => {
            return Err(AssessmentError::IncompatibleArtifact {
                code: code.clone(),
                message: message.clone(),
            });
        }
        (None, HardwareAssessment::InvalidArtifact { code, message }) => {
            return Err(AssessmentError::InvalidArtifact {
                code: code.clone(),
                message: message.clone(),
            });
        }
        (None, _) => return Err(AssessmentError::MissingMeasurements),
    };
    let native_mtp = match requested.mtp {
        MtpConfig::Disabled { .. } => None,
        MtpConfig::Enabled { .. } => Some(
            plan_fit_with_backend(
                backend,
                &fit_request(&assessed.plan, true)?,
                Some(&native),
                false,
            )?
            .native,
        ),
    };
    Ok(BackendLoadPlan {
        assessed,
        native,
        native_mtp,
    })
}

fn plan_and_assess(
    backend: &LlamaBackend,
    requested: &ExecutionIntent,
    policy: CapacityPolicy,
    capture_decode_workload: bool,
) -> Result<(AssessedExecutionPlan, Option<NativeParameterPlan>), AssessmentError> {
    let target_request = fit_request(requested, false)?;
    let target_fit =
        plan_fit_with_backend(backend, &target_request, None, capture_decode_workload)?;
    let text_report = target_fit.report.clone();
    if text_report.status == FitStatus::Error {
        return Ok((
            AssessedExecutionPlan {
                plan: requested.clone(),
                assessment: HardwareAssessment::IncompatibleArtifact {
                    code: "native_backend_incompatible".to_owned(),
                    message: "the pinned native backend cannot plan this valid artifact or execution intent"
                        .to_owned(),
                },
                text_report,
                mtp_report: None,
                #[cfg(feature = "mtmd")]
                projector_memory: Vec::new(),
            },
            None,
        ));
    }
    let mut mtp_report = estimate_mtp_report(backend, requested)?;
    let projector_memory = projector_memory(requested)?;

    let preferred = capacity_summary(
        &text_report.devices,
        Measurement::Initial,
        mtp_report.as_ref().map(|report| report.devices.as_slice()),
        mtp_includes_model(requested),
        &projector_memory,
        policy,
    )?;
    if preferred.fits {
        let plan = assessed_intent(requested, &text_report, Measurement::Initial);
        let native = native_parameter_plan(&target_request)?;
        return Ok((
            AssessedExecutionPlan {
                assessment: fits_assessment(&plan, &preferred, HardwareRecommendation::Recommended),
                plan,
                text_report,
                mtp_report,
                #[cfg(feature = "mtmd")]
                projector_memory,
            },
            Some(native),
        ));
    }

    let fallback_plan = (text_report.status == FitStatus::Success)
        .then(|| assessed_intent(requested, &text_report, Measurement::Fitted));
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
                policy,
            )
        })
        .transpose()?;
    if fallback.as_ref().is_some_and(|summary| summary.fits) {
        let plan = fallback_plan.expect("a fallback summary has a plan");
        let summary = fallback.expect("checked above");
        return Ok((
            AssessedExecutionPlan {
                assessment: fits_assessment(&plan, &summary, HardwareRecommendation::Constrained),
                plan,
                text_report,
                mtp_report,
                #[cfg(feature = "mtmd")]
                projector_memory,
            },
            Some(target_fit.native),
        ));
    }

    let profile = hardware_profile(requested, &preferred);
    let assessment = HardwareAssessment::DoesNotFit {
        profile,
        memory: HardwareDeficit {
            required_bytes: preferred.required_bytes,
            available_bytes: preferred.available_bytes,
            deficit_bytes: preferred.deficit_bytes,
            domains: preferred.domains.clone(),
            device_constraints: preferred.device_constraints.clone(),
        },
        limiting_resource: preferred.limiting_resource,
        alternative: None,
    };
    Ok((
        AssessedExecutionPlan {
            plan: requested.clone(),
            assessment,
            text_report,
            mtp_report,
            #[cfg(feature = "mtmd")]
            projector_memory,
        },
        None,
    ))
}

fn fit_request(plan: &ExecutionIntent, mtp_context: bool) -> Result<FitRequest, AssessmentError> {
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
            threads: plan.execution.threads,
            threads_batch: plan.execution.threads_batch,
        },
    })
}

fn estimate_mtp_report(
    backend: &LlamaBackend,
    plan: &ExecutionIntent,
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

fn mtp_includes_model(plan: &ExecutionIntent) -> bool {
    matches!(
        plan.mtp,
        MtpConfig::Enabled {
            source: MtpSource::Separate { .. },
            ..
        }
    )
}

/// Assess a plan using an existing initialized llama.cpp backend.
///
/// Serving processes must use this entry point from their serialized native executor because
/// llama.cpp backend initialization is process-global and `common/fit` temporarily owns global
/// diagnostic state.
pub fn assess_with_backend(
    backend: &LlamaBackend,
    requested: &ExecutionIntent,
    policy: CapacityPolicy,
) -> Result<AssessedExecutionPlan, AssessmentError> {
    assess_intent_with_backend(backend, requested, policy)
}

fn generation_performance(
    hardware: &HardwareAssessment,
    decode_workload: &FitDecodeWorkloadAssessment,
    devices: &[FitDeviceEstimate],
    unified_memory: bool,
    calibration: Option<&FitCalibration>,
    configured_context_tokens: u32,
) -> GenerationPerformanceAssessment {
    if !matches!(hardware, HardwareAssessment::Fits { .. }) {
        return unavailable_generation_performance(
            "configuration_does_not_fit",
            "generation performance is unavailable for a configuration that does not fit",
        );
    }
    let Some(calibration) = calibration else {
        return GenerationPerformanceAssessment::not_requested();
    };
    let workload = match decode_workload {
        FitDecodeWorkloadAssessment::Available { workload } => workload,
        FitDecodeWorkloadAssessment::Unavailable { reason } => {
            return unavailable_generation_performance(
                "native_workload_unavailable",
                reason.clone(),
            );
        }
    };
    let cross_memory_domain_placement =
        match workload_crosses_memory_domains(workload, devices, unified_memory) {
            Ok(value) => value,
            Err(failure) => {
                return unavailable_generation_performance(failure.code, failure.message);
            }
        };
    match estimate_generation_performance(
        workload,
        calibration,
        configured_context_tokens,
        &GENERATION_PERFORMANCE_CONTEXTS,
        cross_memory_domain_placement,
    ) {
        Ok(estimate) => estimate,
        Err(failure) => unavailable_generation_performance(failure.code, failure.message),
    }
}

fn unavailable_generation_performance(
    code: &str,
    message: impl Into<String>,
) -> GenerationPerformanceAssessment {
    GenerationPerformanceAssessment::Unavailable {
        method: GENERATION_PERFORMANCE_METHOD.to_owned(),
        code: code.to_owned(),
        message: message.into(),
    }
}

#[derive(Debug)]
struct PerformanceEstimateFailure {
    code: &'static str,
    message: String,
}

impl PerformanceEstimateFailure {
    fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

#[derive(Clone, Copy)]
struct CalibrationSelection<'a> {
    metric: &'a FitCalibrationMetric,
    exact: bool,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum PerformanceMemoryDomain {
    HostShared,
    Physical(String),
    NativeDevice(usize),
}

fn performance_memory_domain(
    device: &FitDeviceEstimate,
    unified_memory: bool,
) -> PerformanceMemoryDomain {
    if unified_memory
        || device.kind == FitDeviceKind::Host
        || device.backend_device_type() == LlamaBackendDeviceType::IntegratedGpu
    {
        PerformanceMemoryDomain::HostShared
    } else if let Some(physical_id) = &device.device_id {
        PerformanceMemoryDomain::Physical(physical_id.clone())
    } else {
        PerformanceMemoryDomain::NativeDevice(device.index)
    }
}

fn workload_crosses_memory_domains(
    workload: &FitDecodeWorkload,
    devices: &[FitDeviceEstimate],
    unified_memory: bool,
) -> Result<bool, PerformanceEstimateFailure> {
    let mut domains = BTreeSet::new();
    let mut record = |backend_type: i32, backend: &str, device_id: &Option<String>| {
        let device = devices.iter().find(|device| {
            device.backend_type == backend_type
                && device.backend == backend
                && device.device_id == *device_id
        });
        let Some(device) = device else {
            return Err(PerformanceEstimateFailure::new(
                "workload_device_unresolved",
                format!(
                    "native workload device {backend}/{} is absent from the fit topology",
                    device_id.as_deref().unwrap_or("<unknown>")
                ),
            ));
        };
        domains.insert(performance_memory_domain(device, unified_memory));
        Ok(())
    };
    for tensor in &workload.tensors {
        record(tensor.backend_type, &tensor.backend, &tensor.device_id)?;
    }
    for layer in &workload.kv_layers {
        record(layer.backend_type, &layer.backend, &layer.device_id)?;
    }
    Ok(domains.len() > 1)
}

fn validate_calibration(calibration: &FitCalibration) -> Result<(), PerformanceEstimateFailure> {
    if calibration.method != llama_cpp_2::model::params::fit::FIT_CALIBRATION_METHOD {
        return Err(PerformanceEstimateFailure::new(
            "unsupported_calibration_schema",
            "native calibration schema is not supported",
        ));
    }
    if calibration.metrics.is_empty() {
        return Err(PerformanceEstimateFailure::new(
            "invalid_calibration",
            "native calibration contains no metrics",
        ));
    }
    let mut identities = BTreeSet::new();
    for metric in &calibration.metrics {
        if metric.backend.is_empty()
            || metric.device_id.as_ref().is_some_and(String::is_empty)
            || !metric.bytes_per_second.is_finite()
            || metric.bytes_per_second <= 0.0
            || !metric.launch_microseconds.is_finite()
            || metric.launch_microseconds < 0.0
            || !metric.relative_spread.is_finite()
            || metric.relative_spread < 0.0
        {
            return Err(PerformanceEstimateFailure::new(
                "invalid_calibration",
                "native calibration contains an invalid identity or numeric value",
            ));
        }
        if !identities.insert((
            metric.backend_type,
            metric.backend.as_str(),
            metric.device_id.as_deref(),
            metric.tensor_type,
            metric.routed,
        )) {
            return Err(PerformanceEstimateFailure::new(
                "invalid_calibration",
                "native calibration contains duplicate operation metrics",
            ));
        }
    }
    Ok(())
}

fn calibration_for<'a>(
    calibration: &'a FitCalibration,
    backend_type: i32,
    backend: &str,
    device_id: &Option<String>,
    tensor_type: i32,
    routed: bool,
) -> Result<CalibrationSelection<'a>, PerformanceEstimateFailure> {
    let mut fallback = None;
    for metric in calibration.metrics.iter().filter(|metric| {
        metric.backend_type == backend_type
            && metric.backend == backend
            && metric.device_id == *device_id
            && metric.routed == routed
    }) {
        if metric.tensor_type == tensor_type {
            return Ok(CalibrationSelection {
                metric,
                exact: true,
            });
        }
        if fallback.is_none_or(|current: &FitCalibrationMetric| {
            metric.bytes_per_second < current.bytes_per_second
        }) {
            fallback = Some(metric);
        }
    }
    fallback
        .map(|metric| CalibrationSelection {
            metric,
            exact: false,
        })
        .ok_or_else(|| {
            PerformanceEstimateFailure::new(
                "calibration_coverage_missing",
                format!(
                    "no {} calibration covers backend {backend} device {}",
                    if routed { "routed" } else { "dense" },
                    device_id.as_deref().unwrap_or("<unknown>")
                ),
            )
        })
}

fn active_routed_bytes(
    bytes: u64,
    expert_count: u32,
    expert_used_count: u32,
) -> Result<u64, PerformanceEstimateFailure> {
    if expert_count == 0 || expert_used_count == 0 || expert_used_count > expert_count {
        return Err(PerformanceEstimateFailure::new(
            "invalid_expert_metadata",
            "routed tensors require a non-zero selected-expert count within the total expert count",
        ));
    }
    let numerator = u128::from(bytes)
        .checked_mul(u128::from(expert_used_count))
        .ok_or_else(|| {
            PerformanceEstimateFailure::new(
                "workload_overflow",
                "routed expert byte calculation overflowed",
            )
        })?;
    let scaled = numerator.div_ceil(u128::from(expert_count));
    u64::try_from(scaled).map_err(|_| {
        PerformanceEstimateFailure::new(
            "workload_overflow",
            "routed expert byte calculation exceeds u64",
        )
    })
}

fn operation_seconds(bytes: u64, metric: &FitCalibrationMetric) -> f64 {
    bytes as f64 / metric.bytes_per_second + metric.launch_microseconds / 1_000_000.0
}

fn estimate_generation_performance(
    workload: &FitDecodeWorkload,
    calibration: &FitCalibration,
    configured_context_tokens: u32,
    requested_contexts: &[u32],
    cross_memory_domain_placement: bool,
) -> Result<GenerationPerformanceAssessment, PerformanceEstimateFailure> {
    if workload.method != llama_cpp_2::model::params::fit::FIT_DECODE_WORKLOAD_METHOD {
        return Err(PerformanceEstimateFailure::new(
            "unsupported_workload_schema",
            "native decode workload schema is not supported",
        ));
    }
    validate_calibration(calibration)?;
    if workload.tensors.is_empty() || workload.kv_layers.is_empty() {
        return Err(PerformanceEstimateFailure::new(
            "incomplete_native_workload",
            "native decode workload omitted tensors or KV layers",
        ));
    }
    if configured_context_tokens == 0 || requested_contexts.is_empty() {
        return Err(PerformanceEstimateFailure::new(
            "invalid_context_curve",
            "no non-zero occupied context depths were requested",
        ));
    }

    let has_routed_tensors = workload
        .tensors
        .iter()
        .any(|tensor| tensor.kind == FitTensorWorkloadKind::RoutedExpert);
    if has_routed_tensors {
        active_routed_bytes(1, workload.expert_count, workload.expert_used_count)?;
    } else if workload.expert_count != 0 || workload.expert_used_count != 0 {
        return Err(PerformanceEstimateFailure::new(
            "invalid_expert_metadata",
            "expert metadata is present without routed expert tensors",
        ));
    }

    let mut always_active_weight_bytes = 0_u64;
    let mut routed_expert_weight_bytes = 0_u64;
    let mut weight_seconds = 0.0_f64;
    let mut weight_uncertainty_seconds = 0.0_f64;
    let mut used_fallback_calibration = false;
    for tensor in &workload.tensors {
        if tensor.stored_bytes == 0
            || tensor.operation_bytes == 0
            || tensor.operation_bytes > tensor.stored_bytes
        {
            return Err(PerformanceEstimateFailure::new(
                "invalid_native_workload",
                format!("tensor {} has invalid byte counts", tensor.name),
            ));
        }
        let routed = tensor.kind == FitTensorWorkloadKind::RoutedExpert;
        let active_bytes = if routed {
            routed_expert_weight_bytes = routed_expert_weight_bytes
                .checked_add(tensor.stored_bytes)
                .ok_or_else(|| {
                    PerformanceEstimateFailure::new(
                        "workload_overflow",
                        "routed expert tensor accounting overflowed",
                    )
                })?;
            active_routed_bytes(
                tensor.operation_bytes,
                workload.expert_count,
                workload.expert_used_count,
            )?
        } else {
            always_active_weight_bytes = always_active_weight_bytes
                .checked_add(tensor.operation_bytes)
                .ok_or_else(|| {
                    PerformanceEstimateFailure::new(
                        "workload_overflow",
                        "always-active tensor accounting overflowed",
                    )
                })?;
            tensor.operation_bytes
        };
        let selection = calibration_for(
            calibration,
            tensor.backend_type,
            &tensor.backend,
            &tensor.device_id,
            tensor.tensor_type,
            routed,
        )?;
        used_fallback_calibration |= !selection.exact;
        let seconds = operation_seconds(active_bytes, selection.metric);
        weight_seconds += seconds;
        weight_uncertainty_seconds += seconds * selection.metric.relative_spread.clamp(0.0, 1.0);
    }
    if !weight_seconds.is_finite() || weight_seconds <= 0.0 {
        return Err(PerformanceEstimateFailure::new(
            "invalid_native_workload",
            "native tensor workload produced no finite work",
        ));
    }

    let mut confidence = if has_routed_tensors || workload.hybrid_model {
        GenerationPerformanceConfidence::Moderate
    } else {
        GenerationPerformanceConfidence::High
    };
    if used_fallback_calibration || cross_memory_domain_placement || workload.recurrent_model {
        confidence = GenerationPerformanceConfidence::Low;
    }

    let mut contexts = requested_contexts
        .iter()
        .map(|context| (*context).min(configured_context_tokens))
        .filter(|context| *context > 0)
        .collect::<Vec<_>>();
    contexts.sort_unstable();
    contexts.dedup();
    if contexts.is_empty() {
        return Err(PerformanceEstimateFailure::new(
            "invalid_context_curve",
            "requested occupied context depths resolve to zero",
        ));
    }

    let mut expected_efficiency = if has_routed_tensors {
        ROUTED_DECODE_EFFICIENCY
    } else {
        DENSE_DECODE_EFFICIENCY
    };
    if cross_memory_domain_placement {
        expected_efficiency *= CROSS_DOMAIN_PLACEMENT_EFFICIENCY;
    }
    let mut points = Vec::with_capacity(contexts.len());
    for context_tokens in contexts {
        let mut kv_bytes_read_per_token = 0_u64;
        let mut kv_seconds = 0.0_f64;
        let mut kv_uncertainty_seconds = 0.0_f64;
        for layer in &workload.kv_layers {
            let attended_tokens = if layer.sliding_window_tokens == 0 {
                context_tokens
            } else {
                context_tokens.min(layer.sliding_window_tokens)
            };
            for (tensor_type, row_bytes) in [
                (layer.key_type, layer.key_bytes_per_token),
                (layer.value_type, layer.value_bytes_per_token),
            ] {
                let bytes = row_bytes
                    .checked_mul(u64::from(attended_tokens))
                    .ok_or_else(|| {
                        PerformanceEstimateFailure::new(
                            "workload_overflow",
                            "KV traffic calculation overflowed",
                        )
                    })?;
                kv_bytes_read_per_token =
                    kv_bytes_read_per_token.checked_add(bytes).ok_or_else(|| {
                        PerformanceEstimateFailure::new(
                            "workload_overflow",
                            "KV traffic accounting overflowed",
                        )
                    })?;
                let selection = calibration_for(
                    calibration,
                    layer.backend_type,
                    &layer.backend,
                    &layer.device_id,
                    tensor_type,
                    false,
                )?;
                if !selection.exact {
                    confidence = GenerationPerformanceConfidence::Low;
                }
                let seconds = operation_seconds(bytes, selection.metric);
                kv_seconds += seconds;
                kv_uncertainty_seconds +=
                    seconds * selection.metric.relative_spread.clamp(0.0, 1.0);
            }
        }
        let raw_seconds = weight_seconds + kv_seconds;
        if !raw_seconds.is_finite() || raw_seconds <= 0.0 {
            return Err(PerformanceEstimateFailure::new(
                "invalid_estimate",
                "generation calculation produced a non-finite token time",
            ));
        }
        let observed_spread = (weight_uncertainty_seconds + kv_uncertainty_seconds) / raw_seconds;
        let mut uncertainty = (observed_spread * CALIBRATION_SPREAD_WEIGHT)
            .clamp(MINIMUM_UNCERTAINTY, MAXIMUM_CALIBRATION_UNCERTAINTY);
        if has_routed_tensors {
            uncertainty = (uncertainty + ROUTING_UNCERTAINTY).min(MAXIMUM_ROUTED_UNCERTAINTY);
        }
        if cross_memory_domain_placement {
            uncertainty = (uncertainty + CROSS_DOMAIN_PLACEMENT_UNCERTAINTY)
                .min(MAXIMUM_CROSS_DOMAIN_UNCERTAINTY);
        }
        let lower_tokens_per_second = expected_efficiency * (1.0 - uncertainty) / raw_seconds;
        let expected_tokens_per_second = expected_efficiency / raw_seconds;
        let upper_tokens_per_second =
            (expected_efficiency * (1.0 + uncertainty * UPPER_BOUND_UNCERTAINTY_WEIGHT)).min(1.0)
                / raw_seconds;
        if [
            lower_tokens_per_second,
            expected_tokens_per_second,
            upper_tokens_per_second,
        ]
        .iter()
        .any(|rate| !rate.is_finite() || *rate <= 0.0)
        {
            return Err(PerformanceEstimateFailure::new(
                "invalid_estimate",
                "generation calculation produced a non-finite token rate",
            ));
        }
        points.push(GenerationSpeedPoint {
            context_tokens,
            kv_bytes_read_per_token,
            lower_tokens_per_second,
            expected_tokens_per_second,
            upper_tokens_per_second,
        });
    }

    Ok(GenerationPerformanceAssessment::Estimated {
        method: GENERATION_PERFORMANCE_METHOD.to_owned(),
        confidence,
        workload: GENERATION_PERFORMANCE_WORKLOAD.to_owned(),
        always_active_weight_bytes,
        routed_expert_weight_bytes,
        expert_count: workload.expert_count,
        expert_used_count: workload.expert_used_count,
        cross_memory_domain_placement,
        points,
    })
}

/// Assess several product profiles for one model. The requested no-allocation model is
/// constructed once and llama.cpp constructs an exact context graph for every profile.
/// Upstream fitting is invoked only for profiles whose requested plan does not fit and
/// whose exact tensor storage could still fit across the available memory domains.
pub fn assess_profiles_with_backend(
    backend: &LlamaBackend,
    requested: &[ExecutionIntent],
    policy: CapacityPolicy,
) -> Result<Vec<HardwareAssessment>, AssessmentError> {
    Ok(assess_profiles_impl(backend, requested, policy, None)?
        .into_iter()
        .map(|assessment| assessment.hardware)
        .collect())
}

/// Assess several product profiles and attach native baseline-decode performance evidence.
pub fn assess_execution_profiles_with_backend(
    backend: &LlamaBackend,
    requested: &[ExecutionIntent],
    policy: CapacityPolicy,
    calibration: &FitCalibration,
) -> Result<Vec<ModelExecutionAssessment>, AssessmentError> {
    assess_profiles_impl(backend, requested, policy, Some(calibration))
}

fn assess_profiles_impl(
    backend: &LlamaBackend,
    requested: &[ExecutionIntent],
    policy: CapacityPolicy,
    calibration: Option<&FitCalibration>,
) -> Result<Vec<ModelExecutionAssessment>, AssessmentError> {
    if requested.is_empty() {
        return Ok(Vec::new());
    }
    let requests = requested
        .iter()
        .map(|intent| fit_request(intent, false))
        .collect::<Result<Vec<_>, _>>()?;
    if requests
        .iter()
        .any(|request| request.model != requests[0].model)
    {
        return Err(AssessmentError::MissingMeasurements);
    }

    let native = requests
        .iter()
        .map(native_parameter_plan)
        .collect::<Result<Vec<_>, _>>()?;
    let model_path = path_c_string(&requests[0].model)?;
    let margins = expand_margins(
        &requests[0].options.margins_bytes,
        llama_cpp_2::max_devices(),
    )?;
    let contexts = native
        .iter()
        .map(|plan| plan.context_params.clone())
        .collect::<Vec<_>>();
    let reports = match calibration {
        Some(_) => native[0]
            .model_params
            .as_ref()
            .get_ref()
            .measure_contexts_with_decode_workload(&model_path, &contexts, &margins),
        None => native[0].model_params.as_ref().get_ref().measure_contexts(
            &model_path,
            &contexts,
            &margins,
        ),
    }
    .map_err(EstimateError::Fit)?;
    let unified_memory = cfg!(all(target_os = "macos", target_arch = "aarch64"));

    requested
        .iter()
        .zip(reports)
        .map(|(intent, report)| {
            let projectors = projector_memory(intent)?;
            let preferred = capacity_summary(
                &report.devices,
                Measurement::Initial,
                None,
                false,
                &projectors,
                policy,
            )?;
            if preferred.fits {
                let plan = assessed_intent(intent, &report, Measurement::Initial);
                let hardware =
                    fits_assessment(&plan, &preferred, HardwareRecommendation::Recommended);
                return Ok(ModelExecutionAssessment {
                    performance: generation_performance(
                        &hardware,
                        &report.decode_workload,
                        &report.devices,
                        unified_memory,
                        calibration,
                        report.fitted.resolved_context_tokens,
                    ),
                    hardware,
                });
            }

            // Every load must store every tensor exactly once in some physical memory
            // domain. llama_model_size is the native model's exact tensor storage, so
            // exceeding aggregate stable capacity is a proof of non-fit, not an estimate.
            if report.model.tensor_bytes > preferred.available_bytes {
                let hardware = HardwareAssessment::DoesNotFit {
                    profile: hardware_profile(intent, &preferred),
                    memory: HardwareDeficit {
                        required_bytes: preferred.required_bytes,
                        available_bytes: preferred.available_bytes,
                        deficit_bytes: preferred.deficit_bytes,
                        domains: preferred.domains,
                        device_constraints: preferred.device_constraints,
                    },
                    limiting_resource: preferred.limiting_resource,
                    alternative: None,
                };
                return Ok(ModelExecutionAssessment {
                    performance: generation_performance(
                        &hardware,
                        &report.decode_workload,
                        &report.devices,
                        unified_memory,
                        calibration,
                        report.fitted.resolved_context_tokens,
                    ),
                    hardware,
                });
            }

            let assessed = match calibration {
                Some(_) => assess_intent_with_decode_workload(backend, intent, policy)?,
                None => assess_with_backend(backend, intent, policy)?,
            };
            let performance = generation_performance(
                &assessed.assessment,
                &assessed.text_report.decode_workload,
                &assessed.text_report.devices,
                unified_memory,
                calibration,
                assessed.text_report.fitted.resolved_context_tokens,
            );
            Ok(ModelExecutionAssessment {
                performance,
                hardware: assessed.assessment,
            })
        })
        .collect()
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

fn projector_memory(plan: &ExecutionIntent) -> Result<Vec<ProjectorMemory>, AssessmentError> {
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
    deficit_bytes: u64,
    limiting_resource: String,
    device: String,
    domains: Vec<HardwareMemoryDomainAssessment>,
    device_constraints: Vec<HardwareDeviceMemoryAssessment>,
}

#[derive(Debug)]
struct CapacityDomain {
    native_indices: Vec<usize>,
    physical_id: Option<String>,
    shares_host_memory: bool,
    kind: FitDeviceKind,
    name: String,
    total_bytes: u64,
    model_bytes: u64,
    context_bytes: u64,
    compute_bytes: u64,
    auxiliary_bytes: u64,
    required_bytes: u64,
}

#[derive(Debug)]
struct CapacityDeviceConstraint {
    native_index: usize,
    name: String,
    kind: HardwareDeviceMemoryLimitKind,
    total_bytes: u64,
    model_bytes: u64,
    context_bytes: u64,
    compute_bytes: u64,
    auxiliary_bytes: u64,
    required_bytes: u64,
}

fn capacity_summary(
    devices: &[FitDeviceEstimate],
    measurement: Measurement,
    mtp_devices: Option<&[FitDeviceEstimate]>,
    mtp_includes_model: bool,
    projectors: &[ProjectorMemory],
    policy: CapacityPolicy,
) -> Result<CapacitySummary, AssessmentError> {
    // Every accelerator exposed by the supported macOS llama.cpp build shares the process's
    // unified physical memory with the host. Native device display names such as `MTL0` are not a
    // reliable backend discriminator and previously caused the same RAM to be counted twice.
    capacity_summary_for_topology(
        devices,
        measurement,
        mtp_devices,
        mtp_includes_model,
        projectors,
        policy,
        cfg!(all(target_os = "macos", target_arch = "aarch64")),
    )
}

#[allow(clippy::too_many_arguments)]
fn capacity_summary_for_topology(
    devices: &[FitDeviceEstimate],
    measurement: Measurement,
    mtp_devices: Option<&[FitDeviceEstimate]>,
    mtp_includes_model: bool,
    projectors: &[ProjectorMemory],
    policy: CapacityPolicy,
    unified: bool,
) -> Result<CapacitySummary, AssessmentError> {
    #[cfg(not(feature = "mtmd"))]
    let _ = projectors;
    let mut domains = Vec::<CapacityDomain>::new();
    let mut device_constraints = Vec::<CapacityDeviceConstraint>::new();
    for device in devices {
        let estimate = match measurement {
            Measurement::Initial => device.initial,
            Measurement::Fitted => device.fitted,
        };
        let Some(estimate) = estimate else {
            continue;
        };
        add_device_estimate(
            &mut domains,
            &mut device_constraints,
            device,
            estimate,
            unified,
        )?;
    }
    if let Some(mtp_devices) = mtp_devices {
        for device in mtp_devices {
            let Some(estimate) = device.initial else {
                continue;
            };
            add_additional_device_estimate(
                &mut domains,
                &mut device_constraints,
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
                domain.auxiliary_bytes = domain.auxiliary_bytes.saturating_add(projector.bytes);
                domain.required_bytes = domain.required_bytes.saturating_add(projector.bytes);
            }
            if let Some(constraint) = projector.device_index.and_then(|index| {
                device_constraints
                    .iter_mut()
                    .find(|constraint| constraint.native_index == index)
            }) {
                constraint.auxiliary_bytes =
                    constraint.auxiliary_bytes.saturating_add(projector.bytes);
                constraint.required_bytes =
                    constraint.required_bytes.saturating_add(projector.bytes);
            }
        } else if let Some(domain_index) = projector.device_index.and_then(|index| {
            domains
                .iter()
                .position(|domain| domain.native_indices.contains(&index))
        }) {
            let domain = &mut domains[domain_index];
            domain.auxiliary_bytes = domain.auxiliary_bytes.saturating_add(projector.bytes);
            domain.required_bytes = domain.required_bytes.saturating_add(projector.bytes);
        } else {
            return Err(AssessmentError::MissingMeasurements);
        }
    }
    if domains.is_empty() {
        return Err(AssessmentError::MissingMeasurements);
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
    for constraint in &device_constraints {
        let available = constraint
            .total_bytes
            .saturating_sub(policy.reserve_bytes_per_domain);
        let deficit = constraint.required_bytes.saturating_sub(available);
        if deficit > 0 {
            fits = false;
        }
        if deficit > largest_deficit {
            largest_deficit = deficit;
            limiting_resource.clone_from(&constraint.name);
        }
    }
    Ok(CapacitySummary {
        fits,
        required_bytes,
        available_bytes,
        deficit_bytes: largest_deficit,
        limiting_resource,
        device: domains
            .iter()
            .map(|domain| domain.name.as_str())
            .collect::<Vec<_>>()
            .join(" + "),
        domains: domains
            .into_iter()
            .map(|domain| {
                let available_bytes = domain
                    .total_bytes
                    .saturating_sub(policy.reserve_bytes_per_domain);
                HardwareMemoryDomainAssessment {
                    memory_domain: domain.name,
                    model_bytes: domain.model_bytes,
                    context_bytes: domain.context_bytes,
                    compute_bytes: domain.compute_bytes,
                    auxiliary_bytes: domain.auxiliary_bytes,
                    required_bytes: domain.required_bytes,
                    available_bytes,
                    margin_bytes: (i128::from(available_bytes) - i128::from(domain.required_bytes))
                        .clamp(i128::from(i64::MIN), i128::from(i64::MAX))
                        as i64,
                }
            })
            .collect(),
        device_constraints: device_constraints
            .into_iter()
            .map(|constraint| {
                let available_bytes = constraint
                    .total_bytes
                    .saturating_sub(policy.reserve_bytes_per_domain);
                HardwareDeviceMemoryAssessment {
                    device: constraint.name,
                    kind: constraint.kind,
                    model_bytes: constraint.model_bytes,
                    context_bytes: constraint.context_bytes,
                    compute_bytes: constraint.compute_bytes,
                    auxiliary_bytes: constraint.auxiliary_bytes,
                    required_bytes: constraint.required_bytes,
                    available_bytes,
                    margin_bytes: (i128::from(available_bytes)
                        - i128::from(constraint.required_bytes))
                    .clamp(i128::from(i64::MIN), i128::from(i64::MAX))
                        as i64,
                }
            })
            .collect(),
    })
}

fn add_additional_device_estimate(
    domains: &mut [CapacityDomain],
    device_constraints: &mut [CapacityDeviceConstraint],
    device: &FitDeviceEstimate,
    estimate: FitMemoryEstimate,
    unified: bool,
    include_model: bool,
) -> Result<(), AssessmentError> {
    let domain = if unified {
        domains.first_mut()
    } else {
        domains.iter_mut().find(|domain| {
            (device.device_id.is_some() && domain.physical_id == device.device_id)
                || (domain.native_indices.contains(&device.index) && domain.kind == device.kind)
        })
    }
    .ok_or(AssessmentError::MissingMeasurements)?;
    if include_model {
        domain.model_bytes = domain
            .model_bytes
            .saturating_add(estimate.allocations.model_bytes);
    }
    domain.context_bytes = domain
        .context_bytes
        .saturating_add(estimate.allocations.context_bytes);
    domain.compute_bytes = domain
        .compute_bytes
        .saturating_add(estimate.allocations.compute_bytes);
    domain.required_bytes = domain
        .model_bytes
        .saturating_add(domain.context_bytes)
        .saturating_add(domain.compute_bytes)
        .saturating_add(domain.auxiliary_bytes);
    if unified
        && let Some(constraint) = device_constraints
            .iter_mut()
            .find(|constraint| constraint.native_index == device.index)
    {
        if include_model {
            constraint.model_bytes = constraint
                .model_bytes
                .saturating_add(estimate.allocations.model_bytes);
        }
        constraint.context_bytes = constraint
            .context_bytes
            .saturating_add(estimate.allocations.context_bytes);
        constraint.compute_bytes = constraint
            .compute_bytes
            .saturating_add(estimate.allocations.compute_bytes);
        constraint.required_bytes = constraint
            .model_bytes
            .saturating_add(constraint.context_bytes)
            .saturating_add(constraint.compute_bytes)
            .saturating_add(constraint.auxiliary_bytes);
    }
    Ok(())
}

fn add_device_estimate(
    domains: &mut Vec<CapacityDomain>,
    device_constraints: &mut Vec<CapacityDeviceConstraint>,
    device: &FitDeviceEstimate,
    estimate: FitMemoryEstimate,
    unified: bool,
) -> Result<(), AssessmentError> {
    let total_bytes =
        u64::try_from(estimate.total_bytes).map_err(|_| AssessmentError::MissingMeasurements)?;
    let shares_host_memory = unified
        || device.kind == FitDeviceKind::Host
        || device.backend_device_type() == LlamaBackendDeviceType::IntegratedGpu;
    if unified
        && device.kind == FitDeviceKind::Accelerator
        && (device.backend.eq_ignore_ascii_case("metal")
            || device.backend.eq_ignore_ascii_case("mtl"))
        && total_bytes > 0
    {
        device_constraints.push(CapacityDeviceConstraint {
            native_index: device.index,
            name: device.name.clone(),
            kind: HardwareDeviceMemoryLimitKind::RecommendedWorkingSet,
            total_bytes,
            model_bytes: estimate.allocations.model_bytes,
            context_bytes: estimate.allocations.context_bytes,
            compute_bytes: estimate.allocations.compute_bytes,
            auxiliary_bytes: 0,
            required_bytes: estimate.allocations.total_bytes,
        });
    }
    if shares_host_memory {
        if let Some(domain) = domains.iter_mut().find(|domain| domain.shares_host_memory) {
            domain.native_indices.push(device.index);
            domain.total_bytes = domain.total_bytes.max(total_bytes);
            domain.model_bytes = domain
                .model_bytes
                .saturating_add(estimate.allocations.model_bytes);
            domain.context_bytes = domain
                .context_bytes
                .saturating_add(estimate.allocations.context_bytes);
            domain.compute_bytes = domain
                .compute_bytes
                .saturating_add(estimate.allocations.compute_bytes);
            domain.required_bytes = domain
                .model_bytes
                .saturating_add(domain.context_bytes)
                .saturating_add(domain.compute_bytes)
                .saturating_add(domain.auxiliary_bytes);
        } else {
            domains.push(CapacityDomain {
                native_indices: vec![device.index],
                physical_id: None,
                shares_host_memory: true,
                kind: device.kind,
                name: if unified {
                    "unified_memory"
                } else {
                    "system_memory"
                }
                .to_owned(),
                total_bytes,
                model_bytes: estimate.allocations.model_bytes,
                context_bytes: estimate.allocations.context_bytes,
                compute_bytes: estimate.allocations.compute_bytes,
                auxiliary_bytes: 0,
                required_bytes: estimate.allocations.total_bytes,
            });
        }
    } else if let Some(domain) = device.device_id.as_ref().and_then(|physical_id| {
        domains
            .iter_mut()
            .find(|domain| domain.physical_id.as_ref() == Some(physical_id))
    }) {
        domain.native_indices.push(device.index);
        domain.total_bytes = domain.total_bytes.max(total_bytes);
        domain.model_bytes = domain
            .model_bytes
            .saturating_add(estimate.allocations.model_bytes);
        domain.context_bytes = domain
            .context_bytes
            .saturating_add(estimate.allocations.context_bytes);
        domain.compute_bytes = domain
            .compute_bytes
            .saturating_add(estimate.allocations.compute_bytes);
        domain.required_bytes = domain
            .model_bytes
            .saturating_add(domain.context_bytes)
            .saturating_add(domain.compute_bytes)
            .saturating_add(domain.auxiliary_bytes);
    } else {
        domains.push(CapacityDomain {
            native_indices: vec![device.index],
            physical_id: device.device_id.clone(),
            shares_host_memory: false,
            kind: device.kind,
            name: device.name.clone(),
            total_bytes,
            model_bytes: estimate.allocations.model_bytes,
            context_bytes: estimate.allocations.context_bytes,
            compute_bytes: estimate.allocations.compute_bytes,
            auxiliary_bytes: 0,
            required_bytes: estimate.allocations.total_bytes,
        });
    }
    Ok(())
}

fn assessed_intent(
    requested: &ExecutionIntent,
    report: &FitReport,
    measurement: Measurement,
) -> ExecutionIntent {
    let configuration = match measurement {
        Measurement::Initial => report.requested,
        Measurement::Fitted => report.fitted,
    };
    let mut plan = requested.clone();
    plan.context_size = configuration.resolved_context_tokens;
    plan
}

fn hardware_profile(plan: &ExecutionIntent, summary: &CapacitySummary) -> HardwareProfile {
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
    plan: &ExecutionIntent,
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
            domains: summary.domains.clone(),
            device_constraints: summary.device_constraints.clone(),
        },
        recommendation,
    }
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
    Ok(plan_fit_with_backend(backend, request, None, false)?.report)
}

/// Estimate an MTP model/context linked to the exact target execution context.
pub fn estimate_linked_with_backend(
    backend: &LlamaBackend,
    request: &FitRequest,
    target: &FitRequest,
) -> Result<FitReport, EstimateError> {
    let target = plan_fit_with_backend(backend, target, None, false)?;
    Ok(plan_fit_with_backend(backend, request, Some(&target.native), false)?.report)
}

pub struct NativeParameterPlan {
    model: PathBuf,
    model_params: std::pin::Pin<Box<LlamaModelParams>>,
    context_params: LlamaContextParams,
    threads: NonZeroU32,
    threads_batch: NonZeroU32,
}

/// Exact native parameter objects used only for MTP capability preflight. Construction lives
/// beside ordinary fit/load planning so speculative discovery cannot drift from runtime defaults.
pub struct MtpPreflightParameters {
    pub model_params: std::pin::Pin<Box<LlamaModelParams>>,
    pub target_context: LlamaContextParams,
    pub draft_context: LlamaContextParams,
}

struct NativeFitPlan {
    native: NativeParameterPlan,
    report: FitReport,
}

impl NativeParameterPlan {
    pub fn into_parts(
        self,
    ) -> (
        PathBuf,
        std::pin::Pin<Box<LlamaModelParams>>,
        LlamaContextParams,
        NonZeroU32,
        NonZeroU32,
    ) {
        (
            self.model,
            self.model_params,
            self.context_params,
            self.threads,
            self.threads_batch,
        )
    }
}

pub fn mtp_preflight_parameters(
    intent: &ExecutionIntent,
    recurrent_snapshots: u32,
) -> Result<MtpPreflightParameters, AssessmentError> {
    let mut preflight = intent.clone();
    preflight.mtp = MtpConfig::Enabled {
        source: MtpSource::Bundled,
        n_max: recurrent_snapshots,
        n_min: 0,
        p_min: 0.0,
        cache_type_k: CacheType::F16,
        cache_type_v: CacheType::F16,
    };
    let target = native_parameter_plan(&fit_request(&preflight, false)?)?;
    let draft = native_parameter_plan(&fit_request(&preflight, true)?)?;
    Ok(MtpPreflightParameters {
        model_params: target.model_params,
        target_context: target.context_params,
        draft_context: draft.context_params,
    })
}

/// Build native parameters from intent without fitting. Intended for native parity tooling; model
/// serving must use [`plan_load_with_backend`] so fitted placement is retained.
pub fn native_parameters_for_intent(
    intent: &ExecutionIntent,
) -> Result<NativeParameterPlan, AssessmentError> {
    Ok(native_parameter_plan(&fit_request(intent, false)?)?)
}

fn native_parameter_plan(request: &FitRequest) -> Result<NativeParameterPlan, EstimateError> {
    validate(request)?;
    let (threads, threads_batch) = native_thread_counts(&request.options);
    Ok(NativeParameterPlan {
        model: request.model.clone(),
        model_params: Box::pin(native_model_params(&request.options)?),
        context_params: native_context_params(&request.options),
        threads,
        threads_batch,
    })
}

fn plan_fit_with_backend(
    _backend: &LlamaBackend,
    request: &FitRequest,
    linked_target: Option<&NativeParameterPlan>,
    capture_decode_workload: bool,
) -> Result<NativeFitPlan, EstimateError> {
    let mut native = native_parameter_plan(request)?;
    let model_path = path_c_string(&request.model)?;
    let max_devices = llama_cpp_2::max_devices();
    let mut margins = expand_margins(&request.options.margins_bytes, max_devices)?;

    let report = if let Some(target) = linked_target {
        let target_path = path_c_string(&target.model)?;
        native
            .model_params
            .as_mut()
            .fit_params_report_linked(
                &model_path,
                &mut native.context_params,
                llama_cpp_2::model::params::fit::LinkedFitTarget {
                    model_path: &target_path,
                    model_params: target.model_params.as_ref().get_ref(),
                    context_params: &target.context_params,
                },
                &mut margins,
                request.options.minimum_context_tokens,
            )
            .map_err(EstimateError::Fit)?
    } else if capture_decode_workload {
        native
            .model_params
            .as_mut()
            .fit_params_report_with_decode_workload(
                &model_path,
                &mut native.context_params,
                &mut margins,
                request.options.minimum_context_tokens,
            )
            .map_err(EstimateError::Fit)?
    } else {
        native
            .model_params
            .as_mut()
            .fit_params_report(
                &model_path,
                &mut native.context_params,
                &mut margins,
                request.options.minimum_context_tokens,
            )
            .map_err(EstimateError::Fit)?
    };

    Ok(NativeFitPlan { native, report })
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
    let (threads, threads_batch) = native_thread_counts(options);
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
        .with_n_threads(threads.get().min(i32::MAX as u32) as i32)
        .with_n_threads_batch(threads_batch.get().min(i32::MAX as u32) as i32)
}

fn native_thread_counts(options: &FitOptions) -> (NonZeroU32, NonZeroU32) {
    let threads = options.threads.unwrap_or_else(|| {
        NonZeroU32::new(llama_cpp_2::model::params::fit::default_math_threads())
            .expect("native math-thread default is positive")
    });
    let threads_batch = options.threads_batch.unwrap_or(threads);
    (threads, threads_batch)
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

    fn discovered_device(
        backend: &str,
        name: &str,
        description: &str,
        kind: HardwareDeviceKind,
        total_bytes: u64,
        free_bytes: u64,
    ) -> DiscoveredDevice {
        DiscoveredDevice {
            native_index: 0,
            backend: backend.to_owned(),
            physical_id: None,
            name: name.to_owned(),
            description: description.to_owned(),
            kind,
            total_bytes,
            free_bytes: Some(free_bytes),
        }
    }

    #[test]
    fn hardware_snapshot_deduplicates_backend_aliases_without_merging_duplicate_cards() {
        let gpu = |backend: &str, name: &str| {
            let mut device = discovered_device(
                backend,
                name,
                "Example GPU",
                HardwareDeviceKind::Gpu,
                16_000,
                12_000,
            );
            device.native_index = name.bytes().last().unwrap_or(b'0').saturating_sub(b'0') as usize;
            device.physical_id = Some(format!("0000:01:0{}.0", device.native_index));
            device
        };
        let snapshot = hardware_snapshot_from_devices(
            vec![
                discovered_device(
                    "CPU",
                    "CPU",
                    "Example CPU",
                    HardwareDeviceKind::Cpu,
                    64_000,
                    32_000,
                ),
                gpu("CUDA", "CUDA0"),
                gpu("CUDA", "CUDA1"),
                gpu("Vulkan", "Vulkan0"),
                gpu("Vulkan", "Vulkan1"),
            ],
            CapacityPolicy {
                reserve_bytes_per_domain: 1_000,
            },
            HardwareEnvironment {
                native_build: "build".to_owned(),
                enabled_backends: vec!["vulkan".to_owned(), "cuda".to_owned()],
                platform: "linux".to_owned(),
                architecture: "x86_64".to_owned(),
                logical_cores: 8,
                system_memory: HardwareSystemMemory {
                    total_bytes: 64_000,
                    current_available_bytes: Some(32_000),
                },
            },
        );
        assert_eq!(snapshot.memory_domains.len(), 3);
        assert_eq!(snapshot.memory_domains[0].id, "system");
        for domain in &snapshot.memory_domains[1..] {
            assert_eq!(domain.total_capacity_bytes, 16_000);
            assert_eq!(domain.stable_capacity_bytes, 15_000);
            assert_eq!(domain.devices.len(), 2);
        }
        assert_eq!(snapshot.enabled_backends, vec!["cuda", "vulkan"]);
        assert_eq!(
            resolve_memory_domain(&snapshot, &NativeMemoryLocation::Host),
            Some("system"),
        );
        assert_eq!(
            resolve_memory_domain(
                &snapshot,
                &NativeMemoryLocation::Device {
                    backend: "CUDA".to_owned(),
                    physical_id: Some("0000:01:00.0".to_owned()),
                    native_index: 0,
                },
            ),
            Some(snapshot.memory_domains[1].id.as_str()),
        );
    }

    #[test]
    fn idless_backend_views_are_not_merged_and_zero_capacity_domains_are_omitted() {
        let snapshot = hardware_snapshot_from_devices(
            vec![
                discovered_device(
                    "CPU",
                    "CPU",
                    "Example CPU",
                    HardwareDeviceKind::Cpu,
                    64_000,
                    32_000,
                ),
                discovered_device(
                    "CUDA",
                    "CUDA0",
                    "Example GPU",
                    HardwareDeviceKind::Gpu,
                    16_000,
                    12_000,
                ),
                discovered_device(
                    "Vulkan",
                    "Vulkan0",
                    "Example GPU",
                    HardwareDeviceKind::Gpu,
                    16_000,
                    12_000,
                ),
                discovered_device(
                    "BLAS",
                    "BLAS",
                    "Accelerate",
                    HardwareDeviceKind::Accelerator,
                    0,
                    0,
                ),
            ],
            CapacityPolicy {
                reserve_bytes_per_domain: 1_000,
            },
            HardwareEnvironment {
                native_build: "build".to_owned(),
                enabled_backends: vec!["cpu".to_owned(), "cuda".to_owned(), "vulkan".to_owned()],
                platform: "linux".to_owned(),
                architecture: "x86_64".to_owned(),
                logical_cores: 8,
                system_memory: HardwareSystemMemory {
                    total_bytes: 64_000,
                    current_available_bytes: Some(32_000),
                },
            },
        );

        assert_eq!(snapshot.memory_domains.len(), 3);
        assert_eq!(
            snapshot
                .memory_domains
                .iter()
                .filter(|domain| domain.kind == HardwareMemoryDomainKind::PhysicalDevice)
                .count(),
            2
        );
        assert!(
            snapshot
                .memory_domains
                .iter()
                .all(|domain| { domain.devices.iter().all(|device| device.backend != "BLAS") })
        );
    }

    #[test]
    fn hardware_topology_is_stable_across_order_and_free_memory_changes() {
        let devices = vec![
            discovered_device(
                "CPU",
                "CPU",
                "Example CPU",
                HardwareDeviceKind::Cpu,
                64_000,
                40_000,
            ),
            discovered_device(
                "Metal",
                "MTL0",
                "Example GPU",
                HardwareDeviceKind::Gpu,
                48_000,
                20_000,
            ),
        ];
        let first = hardware_snapshot_from_devices(
            devices.clone(),
            CapacityPolicy::default(),
            HardwareEnvironment {
                native_build: "build".to_owned(),
                enabled_backends: vec!["metal".to_owned(), "cpu".to_owned()],
                platform: "macos".to_owned(),
                architecture: "aarch64".to_owned(),
                logical_cores: 8,
                system_memory: HardwareSystemMemory {
                    total_bytes: 64_000,
                    current_available_bytes: Some(40_000),
                },
            },
        );
        let mut changed = devices.into_iter().rev().collect::<Vec<_>>();
        changed[0].free_bytes = Some(1);
        changed[1].free_bytes = Some(2);
        let second = hardware_snapshot_from_devices(
            changed,
            CapacityPolicy::default(),
            HardwareEnvironment {
                native_build: "build".to_owned(),
                enabled_backends: vec!["cpu".to_owned(), "metal".to_owned()],
                platform: "macos".to_owned(),
                architecture: "aarch64".to_owned(),
                logical_cores: 8,
                system_memory: HardwareSystemMemory {
                    total_bytes: 64_000,
                    current_available_bytes: Some(1),
                },
            },
        );
        assert_eq!(first.topology_fingerprint, second.topology_fingerprint);
        assert_eq!(first.memory_domains.len(), 1);
        assert_eq!(first.memory_domains[0].total_capacity_bytes, 64_000);
        assert_eq!(first.memory_domains[0].devices.len(), 2);
        let metal = first.memory_domains[0]
            .devices
            .iter()
            .find(|device| device.backend == "Metal")
            .expect("Metal device");
        assert_eq!(
            metal.memory_limit,
            Some(HardwareDeviceMemoryLimit {
                kind: HardwareDeviceMemoryLimitKind::RecommendedWorkingSet,
                total_bytes: 48_000,
                stable_bytes: 0,
                current_free_bytes: Some(20_000),
            })
        );
        assert_eq!(first.system_memory.total_bytes, 64_000);
        assert_ne!(
            first.memory_domains[0].current_free_bytes,
            second.memory_domains[0].current_free_bytes
        );
    }

    #[test]
    fn macos_unifies_devices_only_on_apple_silicon() {
        let devices = vec![
            discovered_device(
                "CPU",
                "CPU",
                "Example CPU",
                HardwareDeviceKind::Cpu,
                64_000,
                40_000,
            ),
            discovered_device(
                "Metal",
                "MTL0",
                "Discrete GPU",
                HardwareDeviceKind::Gpu,
                16_000,
                12_000,
            ),
        ];
        let snapshot = hardware_snapshot_from_devices(
            devices,
            CapacityPolicy::default(),
            HardwareEnvironment {
                native_build: "build".to_owned(),
                enabled_backends: vec!["cpu".to_owned(), "metal".to_owned()],
                platform: "macos".to_owned(),
                architecture: "x86_64".to_owned(),
                logical_cores: 8,
                system_memory: HardwareSystemMemory {
                    total_bytes: 64_000,
                    current_available_bytes: Some(40_000),
                },
            },
        );
        assert_eq!(snapshot.memory_domains.len(), 2);
        assert_eq!(
            snapshot.memory_domains[0].kind,
            HardwareMemoryDomainKind::System
        );
        assert_eq!(
            snapshot.memory_domains[1].kind,
            HardwareMemoryDomainKind::PhysicalDevice
        );
    }

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
            backend: "CPU".to_owned(),
            device_id: None,
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
            CapacityPolicy {
                reserve_bytes_per_domain: 100,
            },
        )
        .unwrap();
        assert!(summary.fits);
        assert_eq!(summary.available_bytes, 900);
        assert_eq!(summary.required_bytes, 400);
    }

    #[test]
    fn fit_capacity_uses_exact_physical_identity_across_backend_views() {
        let device = |index, backend: &str, required_bytes| FitDeviceEstimate {
            index,
            kind: FitDeviceKind::Accelerator,
            backend_type: 0,
            backend: backend.to_owned(),
            device_id: Some("0000:01:00.0".to_owned()),
            name: format!("{backend}0"),
            description: "Example GPU".to_owned(),
            initial: Some(FitMemoryEstimate {
                total_bytes: 16_000,
                free_bytes: 12_000,
                allocations: FitAllocations {
                    model_bytes: required_bytes,
                    context_bytes: 0,
                    compute_bytes: 0,
                    total_bytes: required_bytes,
                },
                target: None,
            }),
            fitted: None,
            margin_bytes: None,
        };
        let summary = capacity_summary_for_topology(
            &[device(0, "CUDA", 4_000), device(1, "Vulkan", 3_000)],
            Measurement::Initial,
            None,
            false,
            &[],
            CapacityPolicy {
                reserve_bytes_per_domain: 1_000,
            },
            false,
        )
        .expect("capacity summary");

        assert_eq!(summary.available_bytes, 15_000);
        assert_eq!(summary.required_bytes, 7_000);
        assert_eq!(summary.domains.len(), 1);
    }

    #[test]
    fn integrated_gpu_and_host_share_one_fit_capacity_domain() {
        let device = |index, kind, backend_type, name: &str, total_bytes, required_bytes| {
            FitDeviceEstimate {
                index,
                kind,
                backend_type,
                backend: if kind == FitDeviceKind::Host {
                    "CPU".to_owned()
                } else {
                    "Vulkan".to_owned()
                },
                device_id: None,
                name: name.to_owned(),
                description: name.to_owned(),
                initial: Some(FitMemoryEstimate {
                    total_bytes,
                    free_bytes: total_bytes,
                    allocations: FitAllocations {
                        model_bytes: required_bytes,
                        context_bytes: 0,
                        compute_bytes: 0,
                        total_bytes: required_bytes,
                    },
                    target: None,
                }),
                fitted: None,
                margin_bytes: None,
            }
        };
        let summary = capacity_summary_for_topology(
            &[
                device(0, FitDeviceKind::Accelerator, 2, "iGPU", 8_000, 4_000),
                device(1, FitDeviceKind::Host, 0, "CPU", 32_000, 5_000),
            ],
            Measurement::Initial,
            None,
            false,
            &[],
            CapacityPolicy {
                reserve_bytes_per_domain: 2_000,
            },
            false,
        )
        .expect("capacity summary");

        assert_eq!(summary.available_bytes, 30_000);
        assert_eq!(summary.required_bytes, 9_000);
        assert_eq!(summary.domains.len(), 1);
    }

    #[test]
    fn unified_capacity_counts_physical_memory_once() {
        let device = |index, kind, name: &str, total_bytes, required_bytes| FitDeviceEstimate {
            index,
            kind,
            backend_type: 0,
            backend: if kind == FitDeviceKind::Accelerator {
                "Metal"
            } else {
                "CPU"
            }
            .to_owned(),
            device_id: None,
            name: name.to_owned(),
            description: name.to_owned(),
            initial: Some(FitMemoryEstimate {
                total_bytes,
                free_bytes: 1,
                allocations: FitAllocations {
                    model_bytes: required_bytes,
                    context_bytes: 0,
                    compute_bytes: 0,
                    total_bytes: required_bytes,
                },
                target: None,
            }),
            fitted: None,
            margin_bytes: None,
        };
        let summary = capacity_summary_for_topology(
            &[
                device(0, FitDeviceKind::Accelerator, "MTL0", 48_000, 20_000),
                device(1, FitDeviceKind::Host, "CPU", 64_000, 5_000),
            ],
            Measurement::Initial,
            None,
            false,
            &[],
            CapacityPolicy {
                reserve_bytes_per_domain: 4_000,
            },
            true,
        )
        .unwrap();
        assert_eq!(summary.available_bytes, 60_000);
        assert_eq!(summary.required_bytes, 25_000);
        assert_eq!(summary.device_constraints.len(), 1);
        assert_eq!(summary.device_constraints[0].available_bytes, 44_000);
        assert_eq!(summary.device_constraints[0].required_bytes, 20_000);
    }

    #[test]
    fn apple_metal_working_set_can_limit_an_otherwise_valid_unified_fit() {
        let device = |index, kind, backend: &str, total_bytes, required_bytes| FitDeviceEstimate {
            index,
            kind,
            backend_type: 0,
            backend: backend.to_owned(),
            device_id: None,
            name: if kind == FitDeviceKind::Host {
                "CPU".to_owned()
            } else {
                "MTL0".to_owned()
            },
            description: "Apple M4 Max".to_owned(),
            initial: Some(FitMemoryEstimate {
                total_bytes,
                free_bytes: total_bytes,
                allocations: FitAllocations {
                    model_bytes: required_bytes,
                    context_bytes: 0,
                    compute_bytes: 0,
                    total_bytes: required_bytes,
                },
                target: None,
            }),
            fitted: None,
            margin_bytes: None,
        };
        let summary = capacity_summary_for_topology(
            &[
                device(0, FitDeviceKind::Accelerator, "MTL", 48_000, 47_000),
                device(1, FitDeviceKind::Host, "CPU", 64_000, 1_000),
            ],
            Measurement::Initial,
            None,
            false,
            &[],
            CapacityPolicy {
                reserve_bytes_per_domain: 2_000,
            },
            true,
        )
        .expect("capacity summary");

        assert_eq!(summary.required_bytes, 48_000);
        assert_eq!(summary.available_bytes, 62_000);
        assert!(!summary.fits);
        assert_eq!(summary.deficit_bytes, 1_000);
        assert_eq!(summary.limiting_resource, "MTL0");
        assert_eq!(summary.device_constraints[0].margin_bytes, -1_000);
    }

    #[test]
    fn built_in_mtp_adds_context_and_compute_but_not_duplicate_model_bytes() {
        let device = |model_bytes, context_bytes, compute_bytes| FitDeviceEstimate {
            index: 0,
            kind: FitDeviceKind::Host,
            backend_type: 0,
            backend: "CPU".to_owned(),
            device_id: None,
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
            CapacityPolicy {
                reserve_bytes_per_domain: 0,
            },
        )
        .unwrap();
        assert_eq!(summary.required_bytes, 700);
    }

    fn calibration_metric(
        backend: &str,
        device_id: &str,
        tensor_type: i32,
        routed: bool,
        bytes_per_second: f64,
    ) -> FitCalibrationMetric {
        FitCalibrationMetric {
            backend_type: 2,
            backend: backend.to_owned(),
            device_id: Some(device_id.to_owned()),
            tensor_type,
            routed,
            bytes_per_second,
            launch_microseconds: 0.0,
            relative_spread: 0.0,
        }
    }

    fn calibration(metrics: Vec<FitCalibrationMetric>) -> FitCalibration {
        FitCalibration {
            method: llama_cpp_2::model::params::fit::FIT_CALIBRATION_METHOD.to_owned(),
            metrics,
            elapsed_microseconds: 1,
        }
    }

    fn tensor(
        name: &str,
        kind: FitTensorWorkloadKind,
        stored_bytes: u64,
        operation_bytes: u64,
        tensor_type: i32,
        backend: &str,
        device_id: &str,
    ) -> llama_cpp_2::model::params::fit::FitTensorWorkload {
        llama_cpp_2::model::params::fit::FitTensorWorkload {
            name: name.to_owned(),
            backend_type: 2,
            backend: backend.to_owned(),
            device_id: Some(device_id.to_owned()),
            tensor_type,
            kind,
            stored_bytes,
            operation_bytes,
        }
    }

    fn kv_layer(
        layer: u32,
        row_bytes: u64,
        sliding_window_tokens: u32,
        recurrent: bool,
        backend: &str,
        device_id: &str,
    ) -> llama_cpp_2::model::params::fit::FitKvLayerWorkload {
        llama_cpp_2::model::params::fit::FitKvLayerWorkload {
            layer,
            backend_type: 2,
            backend: backend.to_owned(),
            device_id: Some(device_id.to_owned()),
            key_type: 1,
            value_type: 1,
            key_bytes_per_token: row_bytes,
            value_bytes_per_token: row_bytes,
            sliding_window_tokens,
            recurrent,
        }
    }

    fn workload(
        tensors: Vec<llama_cpp_2::model::params::fit::FitTensorWorkload>,
        kv_layers: Vec<llama_cpp_2::model::params::fit::FitKvLayerWorkload>,
        expert_count: u32,
        expert_used_count: u32,
    ) -> FitDecodeWorkload {
        FitDecodeWorkload {
            method: llama_cpp_2::model::params::fit::FIT_DECODE_WORKLOAD_METHOD.to_owned(),
            expert_count,
            expert_used_count,
            hybrid_model: false,
            recurrent_model: kv_layers.iter().any(|layer| layer.recurrent),
            tensors,
            kv_layers,
        }
    }

    fn performance_device(
        index: usize,
        kind: FitDeviceKind,
        backend_type: i32,
        backend: &str,
        device_id: &str,
    ) -> FitDeviceEstimate {
        FitDeviceEstimate {
            index,
            kind,
            backend_type,
            backend: backend.to_owned(),
            device_id: Some(device_id.to_owned()),
            name: device_id.to_owned(),
            description: device_id.to_owned(),
            initial: None,
            fitted: None,
            margin_bytes: None,
        }
    }

    #[test]
    fn apple_cpu_and_metal_workload_share_one_performance_memory_domain() {
        let workload = workload(
            vec![tensor(
                "token_embd.weight",
                FitTensorWorkloadKind::RowLookup,
                10_000,
                100,
                1,
                "CPU",
                "CPU",
            )],
            vec![kv_layer(0, 10, 0, false, "Metal", "MTL0")],
            0,
            0,
        );
        let devices = [
            performance_device(0, FitDeviceKind::Host, 2, "CPU", "CPU"),
            performance_device(1, FitDeviceKind::Accelerator, 2, "Metal", "MTL0"),
        ];

        assert!(!workload_crosses_memory_domains(&workload, &devices, true).unwrap());
    }

    #[test]
    fn workload_on_distinct_accelerator_memory_domains_is_cross_domain() {
        let mut workload = workload(
            vec![tensor(
                "output.weight",
                FitTensorWorkloadKind::AlwaysActive,
                100,
                100,
                1,
                "CUDA",
                "GPU0",
            )],
            vec![kv_layer(0, 10, 0, false, "CUDA", "GPU1")],
            0,
            0,
        );
        workload.tensors[0].backend_type = 1;
        workload.kv_layers[0].backend_type = 1;
        let devices = [
            performance_device(0, FitDeviceKind::Accelerator, 1, "CUDA", "GPU0"),
            performance_device(1, FitDeviceKind::Accelerator, 1, "CUDA", "GPU1"),
        ];

        assert!(workload_crosses_memory_domains(&workload, &devices, false).unwrap());
    }

    fn estimated_parts(
        assessment: GenerationPerformanceAssessment,
    ) -> (
        GenerationPerformanceConfidence,
        u64,
        u64,
        bool,
        Vec<GenerationSpeedPoint>,
    ) {
        let GenerationPerformanceAssessment::Estimated {
            confidence,
            always_active_weight_bytes,
            routed_expert_weight_bytes,
            cross_memory_domain_placement,
            points,
            ..
        } = assessment
        else {
            panic!("expected generation estimate")
        };
        (
            confidence,
            always_active_weight_bytes,
            routed_expert_weight_bytes,
            cross_memory_domain_placement,
            points,
        )
    }

    #[test]
    fn dense_estimator_uses_native_operation_bytes_and_context_kv_traffic() {
        let workload = workload(
            vec![tensor(
                "output.weight",
                FitTensorWorkloadKind::AlwaysActive,
                800,
                800,
                1,
                "Metal",
                "MTL0",
            )],
            vec![kv_layer(0, 10, 0, false, "Metal", "MTL0")],
            0,
            0,
        );
        let calibration = calibration(vec![calibration_metric("Metal", "MTL0", 1, false, 1_000.0)]);
        let (confidence, always_bytes, routed_bytes, hybrid, points) = estimated_parts(
            estimate_generation_performance(&workload, &calibration, 20, &[10, 20], false).unwrap(),
        );
        assert_eq!(confidence, GenerationPerformanceConfidence::High);
        assert_eq!(always_bytes, 800);
        assert_eq!(routed_bytes, 0);
        assert!(!hybrid);
        assert_eq!(points.len(), 2);
        assert_eq!(points[0].kv_bytes_read_per_token, 200);
        assert_eq!(points[1].kv_bytes_read_per_token, 400);
        assert!((points[0].expected_tokens_per_second - 0.82).abs() < 1e-12);
        assert!((points[1].expected_tokens_per_second - (0.82 / 1.2)).abs() < 1e-12);
        assert!(points.iter().all(|point| {
            point.lower_tokens_per_second <= point.expected_tokens_per_second
                && point.expected_tokens_per_second <= point.upper_tokens_per_second
        }));
    }

    #[test]
    fn row_lookups_do_not_charge_the_complete_embedding_table() {
        let workload = workload(
            vec![tensor(
                "token_embd.weight",
                FitTensorWorkloadKind::RowLookup,
                10_000,
                100,
                1,
                "Metal",
                "MTL0",
            )],
            vec![kv_layer(0, 1, 0, false, "Metal", "MTL0")],
            0,
            0,
        );
        let calibration = calibration(vec![calibration_metric("Metal", "MTL0", 1, false, 1_000.0)]);
        let (_, always_bytes, _, _, _) = estimated_parts(
            estimate_generation_performance(&workload, &calibration, 10, &[10], false).unwrap(),
        );
        assert_eq!(always_bytes, 100);
    }

    #[test]
    fn moe_estimator_scales_only_routed_pools_by_selected_experts() {
        let workload = workload(
            vec![
                tensor(
                    "blk.0.ffn_gate.weight",
                    FitTensorWorkloadKind::AlwaysActive,
                    400,
                    400,
                    1,
                    "Metal",
                    "MTL0",
                ),
                tensor(
                    "blk.0.ffn_gate_exps.weight",
                    FitTensorWorkloadKind::RoutedExpert,
                    800,
                    800,
                    1,
                    "Metal",
                    "MTL0",
                ),
            ],
            vec![kv_layer(0, 5, 0, false, "Metal", "MTL0")],
            8,
            2,
        );
        let calibration = calibration(vec![
            calibration_metric("Metal", "MTL0", 1, false, 1_000.0),
            calibration_metric("Metal", "MTL0", 1, true, 1_000.0),
        ]);
        let (confidence, always_bytes, routed_bytes, _, points) = estimated_parts(
            estimate_generation_performance(&workload, &calibration, 10, &[10], false).unwrap(),
        );
        assert_eq!(confidence, GenerationPerformanceConfidence::Moderate);
        assert_eq!(always_bytes, 400);
        assert_eq!(routed_bytes, 800);
        // 400 always-active + 200 selected expert + 100 KV bytes at 1,000 B/s.
        assert!((points[0].expected_tokens_per_second - (0.75 / 0.7)).abs() < 1e-12);
    }

    #[test]
    fn sliding_window_caps_only_its_own_layer() {
        let workload = workload(
            vec![tensor(
                "output.weight",
                FitTensorWorkloadKind::AlwaysActive,
                100,
                100,
                1,
                "Metal",
                "MTL0",
            )],
            vec![
                kv_layer(0, 5, 0, false, "Metal", "MTL0"),
                kv_layer(1, 5, 10, false, "Metal", "MTL0"),
            ],
            0,
            0,
        );
        let calibration = calibration(vec![calibration_metric("Metal", "MTL0", 1, false, 1_000.0)]);
        let (_, _, _, _, points) = estimated_parts(
            estimate_generation_performance(&workload, &calibration, 20, &[10, 20], false).unwrap(),
        );
        assert_eq!(points[0].kv_bytes_read_per_token, 200);
        assert_eq!(points[1].kv_bytes_read_per_token, 300);
        assert!(points[1].expected_tokens_per_second < points[0].expected_tokens_per_second);
    }

    #[test]
    fn cross_domain_placement_and_calibration_fallback_are_conservative() {
        let workload = workload(
            vec![tensor(
                "output.weight",
                FitTensorWorkloadKind::AlwaysActive,
                100,
                100,
                7,
                "Metal",
                "MTL0",
            )],
            vec![kv_layer(0, 5, 0, false, "CPU", "CPU")],
            0,
            0,
        );
        let calibration = calibration(vec![
            calibration_metric("Metal", "MTL0", 1, false, 1_000.0),
            calibration_metric("Metal", "MTL0", 2, false, 500.0),
            calibration_metric("CPU", "CPU", 1, false, 500.0),
        ]);
        let (confidence, _, _, hybrid, points) = estimated_parts(
            estimate_generation_performance(&workload, &calibration, 10, &[10], true).unwrap(),
        );
        assert_eq!(confidence, GenerationPerformanceConfidence::Low);
        assert!(hybrid);
        // Unknown Metal type 7 uses the slower same-operation fallback (type 2 at 500 B/s).
        assert!((points[0].expected_tokens_per_second - (0.82 * 0.88 / 0.4)).abs() < 1e-12);
    }

    #[test]
    fn more_active_weights_or_experts_never_improve_the_estimate() {
        let calibration = calibration(vec![
            calibration_metric("Metal", "MTL0", 1, false, 1_000.0),
            calibration_metric("Metal", "MTL0", 1, true, 1_000.0),
        ]);
        let make_workload = |always_active_bytes, expert_used_count| {
            workload(
                vec![
                    tensor(
                        "output.weight",
                        FitTensorWorkloadKind::AlwaysActive,
                        always_active_bytes,
                        always_active_bytes,
                        1,
                        "Metal",
                        "MTL0",
                    ),
                    tensor(
                        "blk.0.ffn_exps.weight",
                        FitTensorWorkloadKind::RoutedExpert,
                        800,
                        800,
                        1,
                        "Metal",
                        "MTL0",
                    ),
                ],
                vec![kv_layer(0, 1, 0, false, "Metal", "MTL0")],
                8,
                expert_used_count,
            )
        };
        let rate = |workload: &FitDecodeWorkload| {
            estimated_parts(
                estimate_generation_performance(workload, &calibration, 10, &[10], false).unwrap(),
            )
            .4[0]
                .expected_tokens_per_second
        };

        let baseline = rate(&make_workload(100, 1));
        assert!(rate(&make_workload(200, 1)) < baseline);
        assert!(rate(&make_workload(100, 4)) < baseline);
    }

    #[test]
    fn incomplete_moe_and_calibration_evidence_fail_typed() {
        let invalid_moe = workload(
            vec![tensor(
                "blk.0.ffn_exps.weight",
                FitTensorWorkloadKind::RoutedExpert,
                100,
                100,
                1,
                "Metal",
                "MTL0",
            )],
            vec![kv_layer(0, 1, 0, false, "Metal", "MTL0")],
            8,
            0,
        );
        let dense_only = calibration(vec![calibration_metric("Metal", "MTL0", 1, false, 1_000.0)]);
        let error = estimate_generation_performance(&invalid_moe, &dense_only, 10, &[10], false)
            .expect_err("invalid expert metadata must fail");
        assert_eq!(error.code, "invalid_expert_metadata");

        let valid_moe = FitDecodeWorkload {
            expert_used_count: 2,
            ..invalid_moe
        };
        let error = estimate_generation_performance(&valid_moe, &dense_only, 10, &[10], false)
            .expect_err("dense calibration must not substitute for routed work");
        assert_eq!(error.code, "calibration_coverage_missing");

        let dense = workload(
            vec![tensor(
                "output.weight",
                FitTensorWorkloadKind::AlwaysActive,
                100,
                100,
                1,
                "Metal",
                "MTL0",
            )],
            vec![kv_layer(0, 1, 0, false, "Metal", "MTL0")],
            0,
            0,
        );
        let mut malformed = dense_only.clone();
        malformed.metrics.push(calibration_metric(
            "unused-backend",
            "unused-device",
            1,
            false,
            f64::NAN,
        ));
        let error = estimate_generation_performance(&dense, &malformed, 10, &[10], false)
            .expect_err("every calibration metric must be validated before use");
        assert_eq!(error.code, "invalid_calibration");
    }

    #[test]
    fn kv_overflow_and_recurrent_workloads_are_handled_conservatively() {
        let mut recurrent = workload(
            vec![tensor(
                "output.weight",
                FitTensorWorkloadKind::AlwaysActive,
                100,
                100,
                1,
                "Metal",
                "MTL0",
            )],
            vec![kv_layer(0, 1, 0, true, "Metal", "MTL0")],
            0,
            0,
        );
        let calibration = calibration(vec![calibration_metric("Metal", "MTL0", 1, false, 1_000.0)]);
        let (confidence, _, _, _, _) = estimated_parts(
            estimate_generation_performance(&recurrent, &calibration, 10, &[10], false).unwrap(),
        );
        assert_eq!(confidence, GenerationPerformanceConfidence::Low);

        recurrent.kv_layers[0].key_bytes_per_token = u64::MAX;
        let error = estimate_generation_performance(&recurrent, &calibration, 2, &[2], false)
            .expect_err("KV byte overflow must fail");
        assert_eq!(error.code, "workload_overflow");
    }

    #[test]
    fn non_fitting_configuration_never_exposes_a_speed_estimate() {
        let hardware = HardwareAssessment::DoesNotFit {
            profile: HardwareProfile {
                context_length: 200_000,
                acceleration: "cpu".to_owned(),
                device: "system".to_owned(),
            },
            memory: HardwareDeficit {
                required_bytes: 2,
                available_bytes: 1,
                deficit_bytes: 1,
                domains: Vec::new(),
                device_constraints: Vec::new(),
            },
            limiting_resource: "system".to_owned(),
            alternative: None,
        };
        let performance = generation_performance(
            &hardware,
            &FitDecodeWorkloadAssessment::Available {
                workload: workload(
                    vec![tensor(
                        "output.weight",
                        FitTensorWorkloadKind::AlwaysActive,
                        1,
                        1,
                        1,
                        "CPU",
                        "CPU",
                    )],
                    vec![kv_layer(0, 1, 0, false, "CPU", "CPU")],
                    0,
                    0,
                ),
            },
            &[],
            false,
            Some(&calibration(vec![calibration_metric(
                "CPU", "CPU", 1, false, 1_000.0,
            )])),
            8_192,
        );
        assert!(matches!(
            performance,
            GenerationPerformanceAssessment::Unavailable { ref code, .. }
                if code == "configuration_does_not_fit"
        ));
    }
}
