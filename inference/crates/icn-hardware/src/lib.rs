//! Model assessment over the exact pinned llama.cpp native-planning path.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{CString, NulError};
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub use icn_contracts::{
    CacheType, FlashAttention as PlanningFlashAttention, GpuLayers, SplitMode,
};
use llama_cpp_2::context::params::{
    FlashAttentionPolicy, KvCacheType, LlamaContextParams, LlamaContextType,
};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::params::fit::{
    FitCalibration as HardwareCalibration, FitCalibrationMetric as HardwareCalibrationMetric,
    FitDecodeWorkload as DecodeWorkload, FitDecodeWorkloadAssessment as DecodeWorkloadAssessment,
    FitDeviceEstimate, FitDeviceKind, FitMemoryEstimate, FitStatus, FitTensorWorkloadKind,
};
use llama_cpp_2::model::params::fit::{FitReport, FitReportError};

use icn_contracts::{
    ExecutionIntent, GenerationPerformanceAssessment, GenerationPerformanceConfidence,
    HardwareAssessment, HardwareDeficit, HardwareDevice, HardwareDeviceId, HardwareDeviceKind,
    HardwareDeviceMemoryAssessment, HardwareDeviceMemoryLimit, HardwareDeviceMemoryLimitKind,
    HardwareMemory, HardwareMemoryDomain, HardwareMemoryDomainAssessment, HardwareMemoryDomainKind,
    HardwareProfile, HardwareRecommendation, HardwareSnapshot, HardwareSystemMemory,
    MemoryAccountant, MemoryAccounting, MemoryAccountingError, MemoryBreakdown, MemoryCharge,
    MemoryChargeOwner, MemoryLocation, MemoryTopology, ModelExecutionAssessment,
    NativeDeviceIdentity, NativeDeviceLocator, SpeculativeDecodingConfig, SpeculativeDraftSource,
};
use llama_cpp_2::LlamaBackendDeviceType;
use sha2::{Digest, Sha256};
use sysinfo::{MemoryRefreshKind, RefreshKind, System};

/// Canonical identity for physical system memory in topology and plan assessments.
use icn_contracts::MemoryDomainId;
const GIB: u64 = 1024 * 1024 * 1024;

/// The system-memory policy shared by model assessment, load admission, and runtime supervision.
///
/// Each reserve is the larger of a fraction of physical memory and an absolute floor. Keeping
/// these values in the hardware layer gives every caller one authoritative policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SystemMemoryThresholds {
    pub assess_reserve_bytes: u64,
    pub abort_reserve_bytes: u64,
}

#[must_use]
pub fn system_memory_thresholds(total_bytes: u64) -> SystemMemoryThresholds {
    SystemMemoryThresholds {
        assess_reserve_bytes: (total_bytes / 10).max(2 * GIB),
        abort_reserve_bytes: (total_bytes / 20).max(GIB),
    }
}

/// Cross-platform system-memory facts used by inference allocation.
///
/// Platform-specific constraints are collapsed into the binding allocation capacity and
/// headroom. Callers never need to know which operating-system mechanism supplied the limit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SystemMemoryObservation {
    pub physical_capacity_bytes: u64,
    pub physical_available_bytes: u64,
    pub allocation_capacity_bytes: u64,
    pub allocation_headroom_bytes: u64,
}

pub fn observe_system_memory() -> Result<SystemMemoryObservation, String> {
    let mut system = System::new_with_specifics(
        RefreshKind::nothing().with_memory(MemoryRefreshKind::everything()),
    );
    system.refresh_memory_specifics(MemoryRefreshKind::everything());
    normalize_system_memory(system.total_memory(), system.available_memory())
}

/// Normalizes an already-sampled physical-memory observation into ICN's allocation contract.
///
/// Long-lived observers can reuse their platform sampler while this function keeps operating-
/// system allocation constraints private to the hardware layer.
pub fn normalize_system_memory(
    physical_capacity_bytes: u64,
    physical_available_bytes: u64,
) -> Result<SystemMemoryObservation, String> {
    if physical_capacity_bytes == 0 || physical_available_bytes > physical_capacity_bytes {
        return Err(format!(
            "invalid system memory observation: total={physical_capacity_bytes}, available={physical_available_bytes}"
        ));
    }
    let platform_limit = platform_allocation_limit()?;
    Ok(SystemMemoryObservation {
        physical_capacity_bytes,
        physical_available_bytes,
        allocation_capacity_bytes: platform_limit.map_or(physical_capacity_bytes, |limit| {
            physical_capacity_bytes.min(limit.capacity)
        }),
        allocation_headroom_bytes: platform_limit.map_or(physical_available_bytes, |limit| {
            physical_available_bytes.min(limit.headroom)
        }),
    })
}

#[derive(Clone, Copy)]
struct PlatformAllocationLimit {
    capacity: u64,
    headroom: u64,
}

#[cfg(not(windows))]
fn platform_allocation_limit() -> Result<Option<PlatformAllocationLimit>, String> {
    Ok(None)
}

#[cfg(windows)]
fn platform_allocation_limit() -> Result<Option<PlatformAllocationLimit>, String> {
    use std::mem::{size_of, zeroed};
    use windows_sys::Win32::System::ProcessStatus::{GetPerformanceInfo, PERFORMANCE_INFORMATION};

    // SAFETY: the structure is initialized to its documented size and remains valid for the call.
    unsafe {
        let mut performance: PERFORMANCE_INFORMATION = zeroed();
        performance.cb = size_of::<PERFORMANCE_INFORMATION>() as u32;
        if GetPerformanceInfo(&mut performance, performance.cb) == 0 {
            return Err(format!(
                "GetPerformanceInfo failed: {}",
                std::io::Error::last_os_error()
            ));
        }
        let page_size = performance.PageSize as u64;
        let capacity = (performance.CommitLimit as u64).saturating_mul(page_size);
        let allocated = (performance.CommitTotal as u64).saturating_mul(page_size);
        Ok(Some(PlatformAllocationLimit {
            capacity,
            headroom: capacity.saturating_sub(allocated),
        }))
    }
}
// ICN policy for work not represented by the synthetic matrix-operation calibration.
const GENERATION_PERFORMANCE_WORKLOAD: &str = "baseline_single_sequence_decode";
const DENSE_DECODE_EFFICIENCY: f64 = 0.82;
const ROUTED_DECODE_EFFICIENCY: f64 = 0.75;
const RECURRENT_DECODE_EFFICIENCY: f64 = 0.72;
const SPARSE_ATTENTION_DECODE_EFFICIENCY: f64 = 0.68;
const COMPRESSED_ATTENTION_DECODE_EFFICIENCY: f64 = 0.65;
const RECURRENT_STATE_READ_WRITE_MULTIPLIER: u64 = 2;
const DEEPSEEK4_CSA_STATE_ROWS: u64 = 8;
const DEEPSEEK4_HCA_STATE_ROWS: u64 = 128;
const DEEPSEEK4_STATE_TENSORS: u64 = 2;
const DEEPSEEK4_CSA_EMBEDDING_MULTIPLIER: u64 = 2;
const DEEPSEEK4_STATE_WRITE_ROWS_PER_TOKEN: u64 = 1;
const F32_BYTES: u64 = 4;
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

fn flash_attention_into_native(policy: PlanningFlashAttention) -> FlashAttentionPolicy {
    match policy {
        PlanningFlashAttention::Auto => FlashAttentionPolicy::Auto,
        PlanningFlashAttention::Disabled => FlashAttentionPolicy::Disabled,
        PlanningFlashAttention::Enabled => FlashAttentionPolicy::Enabled,
    }
}

/// Inputs that affect llama.cpp's model, context, and compute estimates.
#[derive(Clone, Debug, serde::Serialize)]
pub struct PlanningOptions {
    /// Context length. `None` requests the model's trained context length.
    pub context_tokens: Option<NonZeroU32>,
    /// Minimum context length allowed during native planning. `u32::MAX` preserves full context.
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
    /// Whether model loading includes multi-token-prediction layers.
    pub load_mtp: bool,
    /// K-cache data type.
    pub cache_type_k: CacheType,
    /// V-cache data type.
    pub cache_type_v: CacheType,
    /// Flash Attention policy.
    pub flash_attention: PlanningFlashAttention,
    /// Whether K/Q/V operations and KV memory may be offloaded.
    pub offload_kqv: bool,
    /// Whether host tensor operations may be offloaded.
    pub operation_offload: bool,
    /// Whether to allocate the full sliding-window cache.
    pub swa_full: bool,
    /// Whether sequences share a unified KV cache.
    pub kv_unified: bool,
    /// Native context role.
    pub context_type: PlanningContextType,
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
pub enum PlanningContextType {
    /// Ordinary target inference context.
    #[default]
    Target,
    /// Multi-token-prediction draft context.
    Mtp,
    /// DFlash or DSpark draft context using the ordinary decoder graph type.
    SpeculativeDraft,
}

impl Default for PlanningOptions {
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
            load_mtp: false,
            cache_type_k: CacheType::F16,
            cache_type_v: CacheType::F16,
            flash_attention: PlanningFlashAttention::Auto,
            offload_kqv: true,
            operation_offload: true,
            swa_full: false,
            kv_unified: false,
            context_type: PlanningContextType::Target,
            recurrent_snapshots: 0,
            maximum_outputs: None,
            threads: None,
            threads_batch: None,
        }
    }
}

/// Request for no-allocation native model planning.
#[derive(Clone, Debug, serde::Serialize)]
pub struct PlanningRequest {
    /// GGUF file to inspect.
    pub model: PathBuf,
    /// Planning parameters.
    pub options: PlanningOptions,
}

/// Validation, backend, or native bridge failure.
#[derive(Debug, thiserror::Error)]
pub enum NativePlanningError {
    /// The model path is not a regular file.
    #[error("model does not exist or is not a file: {0}")]
    InvalidModel(PathBuf),
    /// The model path contains an interior NUL byte.
    #[error("model path contains an interior NUL byte: {0}")]
    ModelPathNul(#[from] NulError),
    /// Invalid native-planning option.
    #[error("invalid native-planning options: {0}")]
    InvalidOptions(String),
    /// llama.cpp backend initialization failed.
    #[error("failed to initialize llama.cpp: {0}")]
    Backend(#[source] llama_cpp_2::LlamaCppError),
    /// Structured native-planning bridge failed.
    #[error(transparent)]
    NativeBridge(#[from] FitReportError),
}

/// Stable ICN assessment-capacity policy. It intentionally uses total capacity,
/// not volatile process-external free memory or application warning policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub struct CapacityPolicy {
    pub reserve_bytes_per_domain: u64,
    #[serde(default)]
    pub system_reserve_bytes: Option<u64>,
}

impl CapacityPolicy {
    fn reserve_for_domain(self, domain: &MemoryDomainId) -> u64 {
        if domain.is_system() {
            self.system_reserve_bytes
                .unwrap_or(self.reserve_bytes_per_domain)
        } else {
            self.reserve_bytes_per_domain
        }
    }
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
    system_product_name: Option<String>,
    logical_cores: usize,
    system_memory: HardwareSystemMemory,
}

impl Default for CapacityPolicy {
    fn default() -> Self {
        Self {
            reserve_bytes_per_domain: 1536 * 1024 * 1024,
            system_reserve_bytes: None,
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
    let total_bytes = system.total_memory();
    let available_bytes = system.available_memory();
    let thresholds = system_memory_thresholds(total_bytes);
    let mut system_memory = HardwareSystemMemory {
        physical_capacity_bytes: total_bytes,
        physical_available_bytes: available_bytes,
        allocation_capacity_bytes: total_bytes,
        allocation_headroom_bytes: available_bytes,
        assess_reserve_bytes: thresholds.assess_reserve_bytes,
        abort_reserve_bytes: thresholds.abort_reserve_bytes,
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
    if system_memory.physical_capacity_bytes == 0 {
        system_memory.physical_capacity_bytes = devices
            .iter()
            .filter(|device| device.kind == HardwareDeviceKind::Cpu)
            .map(|device| device.total_bytes)
            .max()
            .unwrap_or(0);
        system_memory.allocation_capacity_bytes = system_memory.physical_capacity_bytes;
        let thresholds = system_memory_thresholds(system_memory.physical_capacity_bytes);
        system_memory.assess_reserve_bytes = thresholds.assess_reserve_bytes;
        system_memory.abort_reserve_bytes = thresholds.abort_reserve_bytes;
    }
    hardware_snapshot_from_devices(
        devices,
        policy,
        HardwareEnvironment {
            native_build: native_build.into(),
            enabled_backends,
            platform: std::env::consts::OS.to_owned(),
            architecture: std::env::consts::ARCH.to_owned(),
            system_product_name: discover_system_product_name(std::env::consts::OS),
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
                HardwareDeviceKind::Cpu
                    | HardwareDeviceKind::IntegratedGpu
                    | HardwareDeviceKind::Accelerator
            )
        {
            shared.push(device);
        } else {
            let identity = NativeDeviceIdentity::new(
                &device.backend,
                device.physical_id.clone(),
                device.native_index,
            );
            let physical_key = dedicated_physical_key(&identity);
            dedicated
                .entry(physical_key)
                .or_default()
                .entry(identity.backend().to_owned())
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
    if !shared.is_empty() || environment.system_memory.physical_capacity_bytes > 0 {
        let backend_total = shared
            .iter()
            .map(|device| device.total_bytes)
            .max()
            .unwrap_or(0);
        let total = if environment.system_memory.physical_capacity_bytes > 0 {
            environment.system_memory.physical_capacity_bytes
        } else {
            backend_total
        };
        let unified = unified_platform
            || shared
                .iter()
                .any(|device| device.kind == HardwareDeviceKind::IntegratedGpu);
        domains.push(HardwareMemoryDomain {
            id: MemoryDomainId::system(),
            kind: if unified {
                HardwareMemoryDomainKind::UnifiedMemory
            } else {
                HardwareMemoryDomainKind::System
            },
            total_capacity_bytes: total,
            stable_capacity_bytes: total
                .saturating_sub(policy.reserve_for_domain(&MemoryDomainId::system())),
            current_free_bytes: Some(environment.system_memory.physical_available_bytes),
            shares_system_memory: true,
            devices: shared
                .into_iter()
                .map(|device| public_device(device, unified_platform, policy))
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
            let id = dedicated_memory_domain_id(&physical_key, ordinal);
            domains.push(HardwareMemoryDomain {
                id,
                kind: HardwareMemoryDomainKind::PhysicalDevice,
                total_capacity_bytes: total,
                stable_capacity_bytes: total.saturating_sub(policy.reserve_bytes_per_domain),
                current_free_bytes: free,
                shares_system_memory: false,
                devices: views
                    .into_iter()
                    .map(|device| public_device(device, false, policy))
                    .collect(),
            });
        }
    }

    environment.enabled_backends.sort();
    environment.enabled_backends.dedup();

    let topology_fingerprint = topology_fingerprint(&domains);
    HardwareSnapshot {
        captured_at: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_secs()),
        platform: environment.platform,
        architecture: environment.architecture,
        system_product_name: environment.system_product_name,
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
    }
}

fn topology_fingerprint(domains: &[HardwareMemoryDomain]) -> String {
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
                            &device.id,
                            device.native_index,
                            &device.backend,
                            &device.physical_id,
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
    format!("{:x}", Sha256::digest(topology_material))
}

/// Apply one capacity policy to an inventory observation before constructing its topology.
///
/// Assessment consumers receive only the resulting snapshot/topology; policy never participates
/// in location resolution or accounting.
#[must_use]
pub fn with_capacity_policy(
    mut snapshot: HardwareSnapshot,
    policy: CapacityPolicy,
) -> HardwareSnapshot {
    for domain in &mut snapshot.memory_domains {
        domain.stable_capacity_bytes = domain
            .total_capacity_bytes
            .saturating_sub(policy.reserve_for_domain(&domain.id));
        for device in &mut domain.devices {
            if let Some(limit) = &mut device.memory_limit {
                limit.stable_bytes = limit
                    .total_bytes
                    .saturating_sub(policy.reserve_for_domain(&domain.id));
            }
        }
    }
    snapshot.topology_fingerprint = topology_fingerprint(&snapshot.memory_domains);
    snapshot
}

/// Replaces volatile system-memory fields with one freshly normalized observation.
#[must_use]
pub fn with_system_memory_observation(
    mut snapshot: HardwareSnapshot,
    observation: SystemMemoryObservation,
) -> HardwareSnapshot {
    let thresholds = system_memory_thresholds(observation.physical_capacity_bytes);
    snapshot.system_memory = HardwareSystemMemory {
        physical_capacity_bytes: observation.physical_capacity_bytes,
        physical_available_bytes: observation.physical_available_bytes,
        allocation_capacity_bytes: observation.allocation_capacity_bytes,
        allocation_headroom_bytes: observation.allocation_headroom_bytes,
        assess_reserve_bytes: thresholds.assess_reserve_bytes,
        abort_reserve_bytes: thresholds.abort_reserve_bytes,
    };
    for domain in &mut snapshot.memory_domains {
        if domain.id.is_system() {
            domain.total_capacity_bytes = observation.physical_capacity_bytes;
            domain.stable_capacity_bytes = observation
                .physical_capacity_bytes
                .saturating_sub(thresholds.assess_reserve_bytes);
            domain.current_free_bytes = Some(observation.physical_available_bytes);
        }
    }
    snapshot.topology_fingerprint = topology_fingerprint(&snapshot.memory_domains);
    snapshot
}

fn discover_system_product_name(platform: &str) -> Option<String> {
    match platform {
        "linux" => [
            "/sys/devices/virtual/dmi/id/product_name",
            "/sys/class/dmi/id/product_name",
            "/proc/device-tree/model",
        ]
        .into_iter()
        .find_map(|path| {
            std::fs::read(path)
                .ok()
                .and_then(|value| normalize_product_name(&value))
        }),
        "macos" => std::process::Command::new("/usr/sbin/system_profiler")
            .args(["SPHardwareDataType", "-json"])
            .output()
            .ok()
            .filter(|output| output.status.success())
            .and_then(|output| {
                serde_json::from_slice::<serde_json::Value>(&output.stdout)
                    .ok()
                    .and_then(|document| {
                        document
                            .get("SPHardwareDataType")
                            .and_then(serde_json::Value::as_array)
                            .and_then(|entries| entries.first())
                            .and_then(|entry| entry.get("machine_name"))
                            .and_then(serde_json::Value::as_str)
                            .and_then(|name| normalize_product_name(name.as_bytes()))
                    })
            }),
        _ => None,
    }
}

fn normalize_product_name(value: &[u8]) -> Option<String> {
    let name = String::from_utf8_lossy(value)
        .replace('_', " ")
        .trim_matches(|character: char| character == '\0' || character.is_whitespace())
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if name.is_empty()
        || [
            "default string",
            "not specified",
            "system product name",
            "to be filled by o.e.m.",
            "unknown",
        ]
        .iter()
        .any(|placeholder| name.eq_ignore_ascii_case(placeholder))
    {
        None
    } else {
        Some(name)
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

fn dedicated_memory_domain_id(physical_key: &str, ordinal: usize) -> MemoryDomainId {
    let identity = format!("{physical_key}\0{ordinal}");
    MemoryDomainId::new(format!("device-{:x}", Sha256::digest(identity.as_bytes())))
}

fn dedicated_physical_key(identity: &NativeDeviceIdentity) -> String {
    identity.physical_id().map_or_else(
        || format!("backend:{}:{}", identity.backend(), identity.native_index()),
        |id| format!("physical:{id}"),
    )
}

fn public_device(
    device: DiscoveredDevice,
    apple_unified: bool,
    policy: CapacityPolicy,
) -> HardwareDevice {
    let identity = NativeDeviceIdentity::new(
        &device.backend,
        device.physical_id.clone(),
        device.native_index,
    );
    let id = native_device_id(&identity);
    let memory_limit = (apple_unified
        && device.kind != HardwareDeviceKind::Cpu
        && device.total_bytes > 0)
        .then(|| HardwareDeviceMemoryLimit {
            kind: HardwareDeviceMemoryLimitKind::RecommendedWorkingSet,
            total_bytes: device.total_bytes,
            stable_bytes: device
                .total_bytes
                .saturating_sub(policy.reserve_for_domain(&MemoryDomainId::system())),
            current_free_bytes: device.free_bytes,
        });
    HardwareDevice {
        id,
        native_index: device.native_index,
        backend: device.backend,
        physical_id: device.physical_id,
        name: device.name,
        description: device.description,
        kind: device.kind,
        memory_limit,
    }
}

fn native_device_id(identity: &NativeDeviceIdentity) -> HardwareDeviceId {
    let identity = format!(
        "{}\0{}\0{}",
        identity.backend(),
        identity.physical_id().unwrap_or(""),
        identity.native_index()
    );
    HardwareDeviceId::new(format!("native-{:x}", Sha256::digest(identity.as_bytes())))
}

/// The exact plan selected for loading plus its consumer-facing assessment.
#[derive(Clone, Debug)]
pub struct AssessedExecutionPlan {
    pub plan: ExecutionIntent,
    pub assessment: HardwareAssessment,
    pub text_report: FitReport,
    /// No-allocation report for the speculative context and optional companion model.
    pub speculative_report: Option<FitReport>,
    #[cfg(feature = "mtmd")]
    pub projector_memory: Vec<llama_cpp_2::mtmd::MtmdDeviceMemoryEstimate>,
}

/// Process-local backend plan. Its selected native parameters are consumed directly by loading and
/// are never serialized or persisted.
pub struct BackendLoadPlan {
    pub assessed: AssessedExecutionPlan,
    pub native: NativeParameterPlan,
    pub native_speculative: Option<NativeParameterPlan>,
}

/// Exact load planning either produces executable native parameters or a complete rejection.
pub enum BackendLoadPlanningOutcome {
    Planned(BackendLoadPlan),
    Rejected(AssessedExecutionPlan),
}

#[derive(Debug, thiserror::Error)]
pub enum AssessmentError {
    #[error(transparent)]
    Planning(#[from] NativePlanningError),
    #[error("projector assessment requires the icn-hardware mtmd feature")]
    ProjectorUnsupported,
    #[cfg(feature = "mtmd")]
    #[error("projector preflight failed: {0}")]
    Projector(#[from] llama_cpp_2::mtmd::MtmdPreflightError),
    #[error("the native planner omitted required memory measurements")]
    MissingMeasurements,
    #[error("failed to open the model without tensor allocation: {0}")]
    ModelLoad(String),
    #[error(
        "{owner:?} native device is absent from the supplied memory topology: backend={backend:?}, physical_id={physical_id:?}, native_index={native_index}"
    )]
    TopologyMismatch {
        owner: MemoryChargeOwner,
        backend: Option<String>,
        physical_id: Option<String>,
        native_index: usize,
    },
    #[error("artifact is incompatible with the pinned native backend: {code}: {message}")]
    IncompatibleArtifact { code: String, message: String },
    #[error("artifact is invalid: {code}: {message}")]
    InvalidArtifact { code: String, message: String },
    #[error("generation performance assessment failed: {code}: {message}")]
    PerformanceEstimate { code: &'static str, message: String },
}

/// Open the target once without allocating tensor data so one worker request can reuse it for
/// template inspection and profile measurement.
pub fn open_no_alloc_model(
    backend: &LlamaBackend,
    requested: &[ExecutionIntent],
) -> Result<LlamaModel, AssessmentError> {
    let (model_path, model_params, _, _) = profile_measurement_inputs(requested)?;
    LlamaModel::load_from_file(backend, model_path, model_params.as_ref().get_ref())
        .map_err(|error| AssessmentError::ModelLoad(error.to_string()))
}

fn profile_measurement_inputs(
    requested: &[ExecutionIntent],
) -> Result<
    (
        PathBuf,
        std::pin::Pin<Box<LlamaModelParams>>,
        Vec<LlamaContextParams>,
        Vec<usize>,
    ),
    AssessmentError,
> {
    let requests = requested
        .iter()
        .map(|intent| planning_request(intent, false))
        .collect::<Result<Vec<_>, _>>()?;
    let Some(first) = requests.first() else {
        return Err(AssessmentError::MissingMeasurements);
    };
    if requests.iter().any(|request| request.model != first.model) {
        return Err(AssessmentError::MissingMeasurements);
    }
    let native = requests
        .iter()
        .map(native_parameter_plan)
        .collect::<Result<Vec<_>, _>>()?;
    let model_params = Box::pin(
        native_model_params(&first.options)?
            .with_use_mlock(false)
            .with_no_alloc(true),
    );
    let margins = expand_margins(&first.options.margins_bytes, llama_cpp_2::max_devices())?;
    Ok((
        first.model.clone(),
        model_params,
        native.into_iter().map(|plan| plan.context_params).collect(),
        margins,
    ))
}

/// Assess execution intent using the same native planning implementation used by loading.
/// Preview retains only normalized evidence; it never projects native placement back into intent.
pub fn assess_intent_with_backend(
    backend: &LlamaBackend,
    topology: &MemoryTopology,
    requested: &ExecutionIntent,
) -> Result<AssessedExecutionPlan, AssessmentError> {
    Ok(plan_and_assess(backend, topology, requested, false)?.0)
}

fn assess_intent_with_decode_workload(
    backend: &LlamaBackend,
    topology: &MemoryTopology,
    requested: &ExecutionIntent,
) -> Result<AssessedExecutionPlan, AssessmentError> {
    Ok(plan_and_assess(backend, topology, requested, true)?.0)
}

/// Plan a load and retain the exact native parameter object that produced its assessment.
pub fn plan_load_with_backend(
    backend: &LlamaBackend,
    topology: &MemoryTopology,
    requested: &ExecutionIntent,
) -> Result<BackendLoadPlanningOutcome, AssessmentError> {
    let (assessed, native) = plan_and_assess(backend, topology, requested, false)?;
    let Some(native) = native else {
        return match assessed.assessment {
            HardwareAssessment::Fits { .. } => Err(AssessmentError::MissingMeasurements),
            _ => Ok(BackendLoadPlanningOutcome::Rejected(assessed)),
        };
    };
    let native_speculative = match requested.speculative {
        SpeculativeDecodingConfig::Disabled { .. } => None,
        SpeculativeDecodingConfig::Enabled { .. } => Some(
            resolve_native_plan(
                backend,
                &planning_request(&assessed.plan, true)?,
                Some(&native),
                false,
            )?
            .native,
        ),
    };
    Ok(BackendLoadPlanningOutcome::Planned(BackendLoadPlan {
        assessed,
        native,
        native_speculative,
    }))
}

fn plan_and_assess(
    backend: &LlamaBackend,
    topology: &MemoryTopology,
    requested: &ExecutionIntent,
    capture_decode_workload: bool,
) -> Result<(AssessedExecutionPlan, Option<NativeParameterPlan>), AssessmentError> {
    let target_request = planning_request(requested, false)?;
    let target_fit = resolve_native_plan(backend, &target_request, None, capture_decode_workload)?;
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
                speculative_report: None,
                #[cfg(feature = "mtmd")]
                projector_memory: Vec::new(),
            },
            None,
        ));
    }
    let mut speculative_report = estimate_speculative_report(backend, requested)?;
    let projector_memory = projector_memory(requested)?;

    let preferred = capacity_summary(
        topology,
        &text_report.devices,
        Measurement::Initial,
        speculative_report
            .as_ref()
            .map(|report| report.devices.as_slice()),
        speculative_includes_model(requested),
        &projector_memory,
    )?;
    if preferred.fits {
        let plan = assessed_intent(requested, &text_report, Measurement::Initial);
        let native = native_parameter_plan(&target_request)?;
        return Ok((
            AssessedExecutionPlan {
                assessment: fits_assessment(&plan, &preferred, HardwareRecommendation::Recommended),
                plan,
                text_report,
                speculative_report,
                #[cfg(feature = "mtmd")]
                projector_memory,
            },
            Some(native),
        ));
    }

    let fallback_plan = (text_report.status == FitStatus::Success)
        .then(|| assessed_intent(requested, &text_report, Measurement::Selected));
    let fallback = fallback_plan
        .as_ref()
        .map(|plan| {
            speculative_report = estimate_speculative_report(backend, plan)?;
            capacity_summary(
                topology,
                &text_report.devices,
                Measurement::Selected,
                speculative_report
                    .as_ref()
                    .map(|report| report.devices.as_slice()),
                speculative_includes_model(plan),
                &projector_memory,
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
                speculative_report,
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
            usable_capacity_bytes: preferred.usable_capacity_bytes,
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
            speculative_report,
            #[cfg(feature = "mtmd")]
            projector_memory,
        },
        None,
    ))
}

fn planning_request(
    plan: &ExecutionIntent,
    speculative_context: bool,
) -> Result<PlanningRequest, AssessmentError> {
    let (model, cache_type_k, cache_type_v, context_type, recurrent_snapshots, maximum_outputs) =
        if speculative_context {
            let SpeculativeDecodingConfig::Enabled {
                source,
                method,
                cache_type_k,
                cache_type_v,
                ..
            } = &plan.speculative
            else {
                return Err(AssessmentError::MissingMeasurements);
            };
            let model = match source {
                SpeculativeDraftSource::Embedded => plan.model_path.clone(),
                SpeculativeDraftSource::Separate { model_path } => model_path.clone(),
            };
            (
                model,
                *cache_type_k,
                *cache_type_v,
                match method {
                    icn_contracts::SpeculativeMethodConfig::Mtp { .. } => PlanningContextType::Mtp,
                    icn_contracts::SpeculativeMethodConfig::DFlash { .. }
                    | icn_contracts::SpeculativeMethodConfig::DSpark { .. } => {
                        PlanningContextType::SpeculativeDraft
                    }
                },
                0,
                NonZeroU32::new(plan.max_sequences),
            )
        } else {
            let (snapshots, outputs) = match plan.speculative {
                SpeculativeDecodingConfig::Enabled { n_max, .. } => (
                    n_max,
                    NonZeroU32::new(
                        plan.max_sequences
                            .saturating_mul(n_max.saturating_add(1))
                            .min(plan.batch_size),
                    ),
                ),
                SpeculativeDecodingConfig::Disabled { .. } => (0, None),
            };
            (
                plan.model_path.clone(),
                plan.execution.cache_type_k,
                plan.execution.cache_type_v,
                PlanningContextType::Target,
                snapshots,
                outputs,
            )
        };
    Ok(PlanningRequest {
        model,
        options: PlanningOptions {
            context_tokens: NonZeroU32::new(plan.physical_context_size),
            minimum_context_tokens: plan.physical_context_size,
            margins_bytes: vec![0],
            batch_tokens: plan.batch_size,
            micro_batch_tokens: plan.ubatch_size,
            sequence_count: plan.max_sequences,
            gpu_layers: plan.execution.gpu_layers,
            split_mode: plan.execution.split_mode,
            tensor_split: plan.execution.tensor_split.clone(),
            use_mmap: plan.execution.use_mmap,
            use_mlock: plan.execution.use_mlock,
            load_mtp: matches!(
                plan.speculative,
                SpeculativeDecodingConfig::Enabled {
                    source: SpeculativeDraftSource::Embedded,
                    method: icn_contracts::SpeculativeMethodConfig::Mtp { .. },
                    ..
                }
            ) || (speculative_context
                && matches!(
                    plan.speculative,
                    SpeculativeDecodingConfig::Enabled {
                        method: icn_contracts::SpeculativeMethodConfig::Mtp { .. },
                        ..
                    }
                )),
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

fn estimate_speculative_report(
    backend: &LlamaBackend,
    plan: &ExecutionIntent,
) -> Result<Option<FitReport>, AssessmentError> {
    match plan.speculative {
        SpeculativeDecodingConfig::Disabled { .. } => Ok(None),
        SpeculativeDecodingConfig::Enabled { .. } => Ok(Some(assess_linked_model_with_backend(
            backend,
            &planning_request(plan, true)?,
            &planning_request(plan, false)?,
        )?)),
    }
}

fn speculative_includes_model(plan: &ExecutionIntent) -> bool {
    matches!(
        plan.speculative,
        SpeculativeDecodingConfig::Enabled {
            source: SpeculativeDraftSource::Separate { .. },
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
    topology: &MemoryTopology,
    requested: &ExecutionIntent,
) -> Result<AssessedExecutionPlan, AssessmentError> {
    assess_intent_with_backend(backend, topology, requested)
}

fn generation_performance(
    decode_workload: &DecodeWorkloadAssessment,
    devices: &[FitDeviceEstimate],
    topology: &MemoryTopology,
    calibration: &HardwareCalibration,
    context_tokens: &[u32],
) -> Result<Vec<GenerationPerformanceAssessment>, PerformanceEstimateFailure> {
    let workload = match decode_workload {
        DecodeWorkloadAssessment::Available { workload } => workload,
        DecodeWorkloadAssessment::Unavailable { reason } => {
            return Err(PerformanceEstimateFailure::new(
                "native_workload_unavailable",
                reason.clone(),
            ));
        }
    };
    let cross_memory_domain_placement =
        workload_crosses_memory_domains(workload, devices, topology)?;
    context_tokens
        .iter()
        .map(|context_tokens| {
            estimate_generation_performance(
                workload,
                calibration,
                *context_tokens,
                cross_memory_domain_placement,
            )
        })
        .collect()
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
    metric: &'a HardwareCalibrationMetric,
    exact: bool,
}

fn workload_crosses_memory_domains(
    workload: &DecodeWorkload,
    devices: &[FitDeviceEstimate],
    topology: &MemoryTopology,
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
                    "native workload device {backend}/{} is absent from the hardware topology",
                    device_id.as_deref().unwrap_or("<unknown>")
                ),
            ));
        };
        let location = if device.kind == FitDeviceKind::Host {
            MemoryLocation::Host
        } else {
            MemoryLocation::NativeDevice(NativeDeviceLocator::exact(
                &device.backend,
                device.device_id.clone(),
                device.index,
            ))
        };
        let resolved = topology.resolve(&location).ok_or_else(|| {
            PerformanceEstimateFailure::new(
                "workload_device_unresolved",
                format!(
                    "native workload device {backend}/{} is absent from the memory topology",
                    device_id.as_deref().unwrap_or("<unknown>")
                ),
            )
        })?;
        domains.insert(resolved.memory_domain.clone());
        Ok(())
    };
    for tensor in &workload.tensors {
        if tensor.baseline_executed {
            record(tensor.backend_type, &tensor.backend, &tensor.device_id)?;
        }
    }
    for layer in &workload.kv_layers {
        record(layer.backend_type, &layer.backend, &layer.device_id)?;
    }
    Ok(domains.len() > 1)
}

fn validate_calibration(
    calibration: &HardwareCalibration,
) -> Result<(), PerformanceEstimateFailure> {
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
    calibration: &'a HardwareCalibration,
    backend_type: i32,
    backend: &str,
    device_id: &Option<String>,
    tensor_type: i32,
    routed: bool,
) -> Result<CalibrationSelection<'a>, PerformanceEstimateFailure> {
    let mut same_operation_fallback = None;
    let mut dense_fallback = None;
    for metric in calibration.metrics.iter().filter(|metric| {
        metric.backend_type == backend_type
            && metric.backend == backend
            && metric.device_id == *device_id
    }) {
        if metric.routed == routed && metric.tensor_type == tensor_type {
            return Ok(CalibrationSelection {
                metric,
                exact: true,
            });
        }
        if metric.routed == routed
            && same_operation_fallback.is_none_or(|current: &HardwareCalibrationMetric| {
                metric.bytes_per_second < current.bytes_per_second
            })
        {
            same_operation_fallback = Some(metric);
        }
        if routed
            && !metric.routed
            && dense_fallback.is_none_or(|current: &HardwareCalibrationMetric| {
                let metric_exact_type = metric.tensor_type == tensor_type;
                let current_exact_type = current.tensor_type == tensor_type;
                metric_exact_type && !current_exact_type
                    || metric_exact_type == current_exact_type
                        && metric.bytes_per_second < current.bytes_per_second
            })
        {
            dense_fallback = Some(metric);
        }
    }
    same_operation_fallback
        .or(dense_fallback)
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

fn operation_seconds(bytes: u64, metric: &HardwareCalibrationMetric) -> f64 {
    bytes as f64 / metric.bytes_per_second + metric.launch_microseconds / 1_000_000.0
}

fn deepseek4_attention_state_bytes(
    workload: &DecodeWorkload,
    layer: &llama_cpp_2::model::params::fit::FitKvLayerWorkload,
) -> Result<u64, PerformanceEstimateFailure> {
    if workload.architecture != "deepseek4" || layer.compression_ratio == 0 {
        return Ok(0);
    }
    let (state_rows, embedding_width) = match layer.compression_ratio {
        4 => (
            DEEPSEEK4_CSA_STATE_ROWS,
            u64::from(layer.attention_head_size)
                .checked_mul(DEEPSEEK4_CSA_EMBEDDING_MULTIPLIER)
                .ok_or_else(|| {
                    PerformanceEstimateFailure::new(
                        "workload_overflow",
                        "compressed-attention state width overflowed",
                    )
                })?,
        ),
        128 => (
            DEEPSEEK4_HCA_STATE_ROWS,
            u64::from(layer.attention_head_size),
        ),
        _ => return Ok(0),
    };
    let attention_state = DEEPSEEK4_STATE_TENSORS
        .checked_mul(
            state_rows
                .checked_add(DEEPSEEK4_STATE_WRITE_ROWS_PER_TOKEN)
                .ok_or_else(|| {
                    PerformanceEstimateFailure::new(
                        "workload_overflow",
                        "compressed-attention state row calculation overflowed",
                    )
                })?,
        )
        .and_then(|value| value.checked_mul(embedding_width))
        .and_then(|value| value.checked_mul(F32_BYTES))
        .ok_or_else(|| {
            PerformanceEstimateFailure::new(
                "workload_overflow",
                "compressed-attention state calculation overflowed",
            )
        })?;
    let index_state = if layer.compression_ratio == 4 && layer.sparse_index {
        DEEPSEEK4_STATE_TENSORS
            .checked_mul(
                DEEPSEEK4_CSA_STATE_ROWS
                    .checked_add(DEEPSEEK4_STATE_WRITE_ROWS_PER_TOKEN)
                    .ok_or_else(|| {
                        PerformanceEstimateFailure::new(
                            "workload_overflow",
                            "compressed-attention index state row calculation overflowed",
                        )
                    })?,
            )
            .and_then(|value| {
                value.checked_mul(
                    u64::from(workload.indexer_head_size)
                        .checked_mul(DEEPSEEK4_CSA_EMBEDDING_MULTIPLIER)?,
                )
            })
            .and_then(|value| value.checked_mul(F32_BYTES))
            .ok_or_else(|| {
                PerformanceEstimateFailure::new(
                    "workload_overflow",
                    "compressed-attention index state calculation overflowed",
                )
            })?
    } else {
        0
    };
    attention_state.checked_add(index_state).ok_or_else(|| {
        PerformanceEstimateFailure::new(
            "workload_overflow",
            "compressed-attention state traffic overflowed",
        )
    })
}

#[derive(Clone, Copy)]
struct AttentionRow {
    tensor_type: i32,
    bytes_per_token: u64,
}

#[derive(Clone, Copy)]
enum AttentionRows {
    None,
    Conventional {
        key: AttentionRow,
        value: AttentionRow,
    },
    Mla {
        latent: AttentionRow,
    },
}

fn attention_rows(layer: &llama_cpp_2::model::params::fit::FitKvLayerWorkload) -> AttentionRows {
    match layer.attention {
        llama_cpp_2::model::params::fit::FitAttentionWorkload::None => AttentionRows::None,
        llama_cpp_2::model::params::fit::FitAttentionWorkload::Conventional { key, value } => {
            AttentionRows::Conventional {
                key: AttentionRow {
                    tensor_type: key.tensor_type,
                    bytes_per_token: key.bytes_per_token,
                },
                value: AttentionRow {
                    tensor_type: value.tensor_type,
                    bytes_per_token: value.bytes_per_token,
                },
            }
        }
        llama_cpp_2::model::params::fit::FitAttentionWorkload::Mla { latent } => {
            AttentionRows::Mla {
                latent: AttentionRow {
                    tensor_type: latent.tensor_type,
                    bytes_per_token: latent.bytes_per_token,
                },
            }
        }
    }
}

fn estimate_generation_performance(
    workload: &DecodeWorkload,
    calibration: &HardwareCalibration,
    context_tokens: u32,
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
    let mut kv_layer_ids = BTreeSet::new();
    for layer in &workload.kv_layers {
        if !kv_layer_ids.insert(layer.layer) {
            return Err(PerformanceEstimateFailure::new(
                "invalid_native_workload",
                format!(
                    "native decode workload contains duplicate KV layer {}",
                    layer.layer
                ),
            ));
        }
    }
    if context_tokens == 0 {
        return Err(PerformanceEstimateFailure::new(
            "invalid_context",
            "the assessed context must be non-zero",
        ));
    }

    if workload.architecture.is_empty() {
        return Err(PerformanceEstimateFailure::new(
            "incomplete_native_workload",
            "native decode workload omitted the model architecture",
        ));
    }

    let has_routed_tensors = workload.tensors.iter().any(|tensor| {
        tensor.baseline_executed && tensor.kind == FitTensorWorkloadKind::RoutedExpert
    });
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
    let mut used_unstable_calibration = false;
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
        if !tensor.baseline_executed {
            continue;
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
        used_unstable_calibration |= !selection.metric.stable;
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

    let specialized_attention = workload
        .kv_layers
        .iter()
        .any(|layer| layer.compression_ratio > 0 || layer.sparse_index);
    let mut confidence = if has_routed_tensors
        || workload.hybrid_model
        || workload.recurrent_model
        || specialized_attention
    {
        GenerationPerformanceConfidence::Moderate
    } else {
        GenerationPerformanceConfidence::High
    };
    if used_fallback_calibration || used_unstable_calibration || cross_memory_domain_placement {
        confidence = GenerationPerformanceConfidence::Low;
    }

    let mut expected_efficiency = if has_routed_tensors {
        ROUTED_DECODE_EFFICIENCY
    } else {
        DENSE_DECODE_EFFICIENCY
    };
    if workload.recurrent_model {
        expected_efficiency = expected_efficiency.min(RECURRENT_DECODE_EFFICIENCY);
    }
    if workload.kv_layers.iter().any(|layer| layer.sparse_index) {
        expected_efficiency = expected_efficiency.min(SPARSE_ATTENTION_DECODE_EFFICIENCY);
    }
    if workload
        .kv_layers
        .iter()
        .any(|layer| layer.compression_ratio > 0)
    {
        expected_efficiency = expected_efficiency.min(COMPRESSED_ATTENTION_DECODE_EFFICIENCY);
    }
    if cross_memory_domain_placement {
        expected_efficiency *= CROSS_DOMAIN_PLACEMENT_EFFICIENCY;
    }
    let mut kv_bytes_read_per_token = 0_u64;
    let mut kv_seconds = 0.0_f64;
    let mut kv_uncertainty_seconds = 0.0_f64;
    for layer in &workload.kv_layers {
        if layer.recurrent {
            let state_bytes = layer
                .recurrent_conv_bytes
                .checked_add(layer.recurrent_state_bytes)
                .and_then(|bytes| bytes.checked_mul(RECURRENT_STATE_READ_WRITE_MULTIPLIER))
                .ok_or_else(|| {
                    PerformanceEstimateFailure::new(
                        "workload_overflow",
                        "recurrent-state traffic calculation overflowed",
                    )
                })?;
            if state_bytes == 0 {
                confidence = GenerationPerformanceConfidence::Low;
                continue;
            }
            let selection = calibration_for(
                calibration,
                layer.backend_type,
                &layer.backend,
                &layer.device_id,
                layer.recurrent_type,
                false,
            )?;
            if !selection.exact {
                confidence = GenerationPerformanceConfidence::Low;
            }
            if !selection.metric.stable {
                confidence = GenerationPerformanceConfidence::Low;
            }
            let seconds = operation_seconds(state_bytes, selection.metric);
            kv_seconds += seconds;
            kv_uncertainty_seconds += seconds * selection.metric.relative_spread.clamp(0.0, 1.0);
            continue;
        }

        let specialized_attention = layer.compression_ratio > 0 || layer.sparse_index;
        let attention_rows = attention_rows(layer);
        if matches!(attention_rows, AttentionRows::None) {
            if specialized_attention {
                confidence = GenerationPerformanceConfidence::Low;
            } else {
                continue;
            }
        }

        let stored_tokens = if layer.compression_ratio > 0 {
            context_tokens.div_ceil(layer.compression_ratio)
        } else {
            context_tokens
        };
        let attended_tokens = if layer.sparse_index && workload.indexer_top_k > 0 {
            stored_tokens.min(workload.indexer_top_k)
        } else if layer.sliding_window_tokens > 0 {
            context_tokens.min(layer.sliding_window_tokens)
        } else {
            stored_tokens
        };
        let calibration_tensor_type = match attention_rows {
            AttentionRows::None => None,
            AttentionRows::Conventional { key, .. } => Some(key.tensor_type),
            AttentionRows::Mla { latent } => Some(latent.tensor_type),
        };
        let rows = match attention_rows {
            AttentionRows::None => [None, None],
            AttentionRows::Conventional { key, value } => [Some(key), Some(value)],
            AttentionRows::Mla { latent } => [Some(latent), None],
        };
        for row in rows.into_iter().flatten() {
            let tensor_type = row.tensor_type;
            let row_bytes = row.bytes_per_token;
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
            if !selection.metric.stable {
                confidence = GenerationPerformanceConfidence::Low;
            }
            let seconds = operation_seconds(bytes, selection.metric);
            kv_seconds += seconds;
            kv_uncertainty_seconds += seconds * selection.metric.relative_spread.clamp(0.0, 1.0);
        }

        if layer.sparse_index {
            let index_depth = if layer.compression_ratio > 0 {
                stored_tokens
            } else {
                context_tokens
            };
            let index_bytes = layer
                .indexer_bytes_per_token
                .checked_mul(u64::from(index_depth))
                .ok_or_else(|| {
                    PerformanceEstimateFailure::new(
                        "workload_overflow",
                        "sparse-index traffic calculation overflowed",
                    )
                })?;
            if index_bytes == 0 {
                confidence = GenerationPerformanceConfidence::Low;
            } else {
                kv_bytes_read_per_token = kv_bytes_read_per_token
                    .checked_add(index_bytes)
                    .ok_or_else(|| {
                        PerformanceEstimateFailure::new(
                            "workload_overflow",
                            "sparse-index traffic accounting overflowed",
                        )
                    })?;
                let selection = calibration_for(
                    calibration,
                    layer.backend_type,
                    &layer.backend,
                    &layer.device_id,
                    calibration_tensor_type.ok_or_else(|| {
                        PerformanceEstimateFailure::new(
                            "invalid_native_workload",
                            "sparse attention has no attention storage type",
                        )
                    })?,
                    false,
                )?;
                if !selection.exact {
                    confidence = GenerationPerformanceConfidence::Low;
                }
                if !selection.metric.stable {
                    confidence = GenerationPerformanceConfidence::Low;
                }
                let seconds = operation_seconds(index_bytes, selection.metric);
                kv_seconds += seconds;
                kv_uncertainty_seconds +=
                    seconds * selection.metric.relative_spread.clamp(0.0, 1.0);
            }
        }

        let attention_state_bytes = deepseek4_attention_state_bytes(workload, layer)?;
        if attention_state_bytes > 0 {
            if layer.attention_head_size == 0 {
                confidence = GenerationPerformanceConfidence::Low;
            } else {
                let selection = calibration_for(
                    calibration,
                    layer.backend_type,
                    &layer.backend,
                    &layer.device_id,
                    layer.attention_state_type,
                    false,
                )?;
                if !selection.exact {
                    confidence = GenerationPerformanceConfidence::Low;
                }
                if !selection.metric.stable {
                    confidence = GenerationPerformanceConfidence::Low;
                }
                let seconds = operation_seconds(attention_state_bytes, selection.metric);
                kv_seconds += seconds;
                kv_uncertainty_seconds +=
                    seconds * selection.metric.relative_spread.clamp(0.0, 1.0);
            }
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
    Ok(GenerationPerformanceAssessment {
        confidence,
        workload: GENERATION_PERFORMANCE_WORKLOAD.to_owned(),
        always_active_weight_bytes,
        routed_expert_weight_bytes,
        expert_count: workload.expert_count,
        expert_used_count: workload.expert_used_count,
        cross_memory_domain_placement,
        context_tokens,
        kv_bytes_read_per_token,
        lower_tokens_per_second,
        expected_tokens_per_second,
        upper_tokens_per_second,
    })
}

/// Assess several product profiles for one model. The requested no-allocation model is
/// constructed once and llama.cpp constructs an exact context graph for every profile.
/// Native placement selection is invoked only for profiles whose requested plan does not fit and
/// whose exact tensor storage could still fit across the available memory domains.
pub fn assess_profiles_with_backend(
    backend: &LlamaBackend,
    topology: &MemoryTopology,
    requested: &[ExecutionIntent],
) -> Result<Vec<HardwareAssessment>, AssessmentError> {
    if requested.is_empty() {
        return Ok(Vec::new());
    }
    let model = open_no_alloc_model(backend, requested)?;
    assess_profiles_with_model(backend, topology, requested, &model)
}

/// Assess capacity profiles using an already-open no-allocation model.
pub fn assess_profiles_with_model(
    backend: &LlamaBackend,
    topology: &MemoryTopology,
    requested: &[ExecutionIntent],
    model: &LlamaModel,
) -> Result<Vec<HardwareAssessment>, AssessmentError> {
    Ok(
        assess_profiles_impl(backend, topology, requested, model, None, &[])?
            .into_iter()
            .map(|assessment| assessment.hardware)
            .collect(),
    )
}

/// Assess several product profiles and attach native baseline-decode performance evidence.
pub fn assess_execution_profiles_with_backend(
    backend: &LlamaBackend,
    topology: &MemoryTopology,
    requested: &[ExecutionIntent],
    calibration: &HardwareCalibration,
    performance_context_tokens: &[Vec<u32>],
) -> Result<Vec<ModelExecutionAssessment>, AssessmentError> {
    if requested.is_empty() {
        return Ok(Vec::new());
    }
    let model = open_no_alloc_model(backend, requested)?;
    assess_execution_profiles_with_model(
        backend,
        topology,
        requested,
        &model,
        calibration,
        performance_context_tokens,
    )
}

/// Assess execution profiles using an already-open no-allocation model.
pub fn assess_execution_profiles_with_model(
    backend: &LlamaBackend,
    topology: &MemoryTopology,
    requested: &[ExecutionIntent],
    model: &LlamaModel,
    calibration: &HardwareCalibration,
    performance_context_tokens: &[Vec<u32>],
) -> Result<Vec<ModelExecutionAssessment>, AssessmentError> {
    assess_profiles_impl(
        backend,
        topology,
        requested,
        model,
        Some(calibration),
        performance_context_tokens,
    )?
    .into_iter()
    .map(|assessment| match assessment.performance {
        Some(performance) => Ok(ModelExecutionAssessment::Executable {
            hardware: assessment.hardware,
            performance,
        }),
        None => Ok(ModelExecutionAssessment::NotExecutable {
            hardware: assessment.hardware,
        }),
    })
    .collect()
}

struct ProfileAssessment {
    hardware: HardwareAssessment,
    performance: Option<Vec<GenerationPerformanceAssessment>>,
}

fn assess_profiles_impl(
    backend: &LlamaBackend,
    topology: &MemoryTopology,
    requested: &[ExecutionIntent],
    model: &LlamaModel,
    calibration: Option<&HardwareCalibration>,
    performance_context_tokens: &[Vec<u32>],
) -> Result<Vec<ProfileAssessment>, AssessmentError> {
    if calibration.is_some() && performance_context_tokens.len() != requested.len() {
        return Err(AssessmentError::PerformanceEstimate {
            code: "invalid_performance_contexts",
            message: "performance sample contexts did not match assessed profiles".to_owned(),
        });
    }
    let (_, model_params, contexts, margins) = profile_measurement_inputs(requested)?;
    let reports = match calibration {
        Some(_) => model_params
            .as_ref()
            .get_ref()
            .measure_loaded_contexts_with_decode_workload(model, &contexts, &margins),
        None => model_params
            .as_ref()
            .get_ref()
            .measure_loaded_contexts(model, &contexts, &margins),
    }
    .map_err(NativePlanningError::NativeBridge)?;
    let aggregate_stable_capacity = topology.aggregate_stable_capacity();
    requested
        .iter()
        .zip(reports)
        .enumerate()
        .map(|(index, (intent, report))| {
            let projectors = projector_memory(intent)?;
            let preferred = capacity_summary(
                topology,
                &report.devices,
                Measurement::Initial,
                None,
                false,
                &projectors,
            )?;
            if preferred.fits {
                let plan = assessed_intent(intent, &report, Measurement::Initial);
                let hardware =
                    fits_assessment(&plan, &preferred, HardwareRecommendation::Recommended);
                return Ok(ProfileAssessment {
                    performance: calibration
                        .map(|calibration| {
                            generation_performance(
                                &report.decode_workload,
                                &report.devices,
                                topology,
                                calibration,
                                &performance_context_tokens[index],
                            )
                        })
                        .transpose()
                        .map_err(|failure| AssessmentError::PerformanceEstimate {
                            code: failure.code,
                            message: failure.message,
                        })?,
                    hardware,
                });
            }

            // Every load must store every tensor exactly once in some physical memory
            // domain. llama_model_size is the native model's exact tensor storage, so
            // exceeding aggregate stable capacity is a proof of non-fit, not an estimate.
            if report.model.tensor_bytes > aggregate_stable_capacity {
                let hardware = HardwareAssessment::DoesNotFit {
                    profile: hardware_profile(intent, &preferred),
                    memory: HardwareDeficit {
                        required_bytes: preferred.required_bytes,
                        usable_capacity_bytes: preferred.usable_capacity_bytes,
                        deficit_bytes: preferred.deficit_bytes,
                        domains: preferred.domains,
                        device_constraints: preferred.device_constraints,
                    },
                    limiting_resource: preferred.limiting_resource,
                    alternative: None,
                };
                return Ok(ProfileAssessment {
                    performance: None,
                    hardware,
                });
            }

            let assessed = match calibration {
                Some(_) => assess_intent_with_decode_workload(backend, topology, intent)?,
                None => assess_with_backend(backend, topology, intent)?,
            };
            let performance = if matches!(assessed.assessment, HardwareAssessment::Fits { .. }) {
                calibration
                    .map(|calibration| {
                        generation_performance(
                            &assessed.text_report.decode_workload,
                            &assessed.text_report.devices,
                            topology,
                            calibration,
                            &performance_context_tokens[index],
                        )
                    })
                    .transpose()
                    .map_err(|failure| AssessmentError::PerformanceEstimate {
                        code: failure.code,
                        message: failure.message,
                    })?
            } else {
                None
            };
            Ok(ProfileAssessment {
                performance,
                hardware: assessed.assessment,
            })
        })
        .collect()
}

#[derive(Clone, Copy)]
enum Measurement {
    Initial,
    Selected,
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
                PlanningFlashAttention::Auto => FlashAttentionPolicy::Auto,
                PlanningFlashAttention::Disabled => FlashAttentionPolicy::Disabled,
                PlanningFlashAttention::Enabled => FlashAttentionPolicy::Enabled,
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
    usable_capacity_bytes: u64,
    deficit_bytes: u64,
    limiting_resource: String,
    device: String,
    domains: Vec<HardwareMemoryDomainAssessment>,
    device_constraints: Vec<HardwareDeviceMemoryAssessment>,
}

fn native_memory_charge(
    owner: MemoryChargeOwner,
    device: &FitDeviceEstimate,
    estimate: FitMemoryEstimate,
    include_model: bool,
) -> MemoryCharge {
    let location = if device.kind == FitDeviceKind::Host {
        MemoryLocation::Host
    } else {
        MemoryLocation::NativeDevice(NativeDeviceLocator::exact(
            &device.backend,
            device.device_id.clone(),
            device.index,
        ))
    };
    let memory = MemoryBreakdown::new(
        estimate.allocations.model_bytes,
        estimate.allocations.context_bytes,
        estimate.allocations.compute_bytes,
        0,
    );
    MemoryCharge::new(
        owner,
        location,
        if include_model {
            memory
        } else {
            memory.without_model()
        },
    )
}

#[cfg(feature = "mtmd")]
fn projector_memory_charge(projector: &ProjectorMemory) -> Result<MemoryCharge, AssessmentError> {
    Ok(MemoryCharge::new(
        MemoryChargeOwner::Projector,
        MemoryLocation::NativeDevice(NativeDeviceLocator::by_index(
            projector
                .device_index
                .ok_or(AssessmentError::MissingMeasurements)?,
        )),
        MemoryBreakdown::new(0, 0, 0, projector.bytes),
    ))
}

fn capacity_summary_from_accounting(accounting: MemoryAccounting) -> CapacitySummary {
    let domains = accounting.domains;
    let device_constraints = accounting.device_constraints;
    let mut required_bytes = 0_u64;
    let mut usable_capacity_bytes = 0_u64;
    let mut limiting_resource = domains[0].id.to_string();
    let mut largest_deficit = 0_u64;
    let mut fits = true;
    for domain in &domains {
        let usable_capacity = domain.usable_capacity_bytes;
        let required = domain.memory.total_bytes();
        let deficit = required.saturating_sub(usable_capacity);
        if deficit > 0 {
            fits = false;
        }
        if deficit >= largest_deficit {
            largest_deficit = deficit;
            limiting_resource = domain.id.to_string();
        }
        required_bytes = required_bytes.saturating_add(required);
        usable_capacity_bytes = usable_capacity_bytes.saturating_add(usable_capacity);
    }
    for constraint in &device_constraints {
        let usable_capacity = constraint.usable_capacity_bytes;
        let deficit = constraint
            .memory
            .total_bytes()
            .saturating_sub(usable_capacity);
        if deficit > 0 {
            fits = false;
        }
        if deficit > largest_deficit {
            largest_deficit = deficit;
            limiting_resource.clone_from(&constraint.name);
        }
    }
    CapacitySummary {
        fits,
        required_bytes,
        usable_capacity_bytes,
        deficit_bytes: largest_deficit,
        limiting_resource,
        device: domains
            .iter()
            .map(|domain| domain.id.as_str())
            .collect::<Vec<_>>()
            .join(" + "),
        domains: domains
            .into_iter()
            .map(|domain| {
                let usable_capacity_bytes = domain.usable_capacity_bytes;
                let required_bytes = domain.memory.total_bytes();
                HardwareMemoryDomainAssessment {
                    memory_domain: domain.id,
                    model_bytes: domain.memory.model_bytes,
                    context_bytes: domain.memory.context_bytes,
                    compute_bytes: domain.memory.compute_bytes,
                    auxiliary_bytes: domain.memory.auxiliary_bytes,
                    required_bytes,
                    usable_capacity_bytes,
                    margin_bytes: (i128::from(usable_capacity_bytes) - i128::from(required_bytes))
                        .clamp(i128::from(i64::MIN), i128::from(i64::MAX))
                        as i64,
                }
            })
            .collect(),
        device_constraints: device_constraints
            .into_iter()
            .map(|constraint| {
                let usable_capacity_bytes = constraint.usable_capacity_bytes;
                let required_bytes = constraint.memory.total_bytes();
                HardwareDeviceMemoryAssessment {
                    device_id: constraint.device_id,
                    device: constraint.name,
                    kind: constraint.kind,
                    model_bytes: constraint.memory.model_bytes,
                    context_bytes: constraint.memory.context_bytes,
                    compute_bytes: constraint.memory.compute_bytes,
                    auxiliary_bytes: constraint.memory.auxiliary_bytes,
                    required_bytes,
                    usable_capacity_bytes,
                    margin_bytes: (i128::from(usable_capacity_bytes) - i128::from(required_bytes))
                        .clamp(i128::from(i64::MIN), i128::from(i64::MAX))
                        as i64,
                }
            })
            .collect(),
    }
}

fn accounting_error(error: MemoryAccountingError) -> AssessmentError {
    match error.location {
        MemoryLocation::Host => AssessmentError::MissingMeasurements,
        MemoryLocation::NativeDevice(locator) => AssessmentError::TopologyMismatch {
            owner: error.owner,
            backend: locator.backend().map(str::to_owned),
            physical_id: locator.physical_id().map(str::to_owned),
            native_index: locator.native_index(),
        },
    }
}

fn capacity_summary(
    topology: &MemoryTopology,
    devices: &[FitDeviceEstimate],
    measurement: Measurement,
    speculative_devices: Option<&[FitDeviceEstimate]>,
    speculative_includes_model: bool,
    projectors: &[ProjectorMemory],
) -> Result<CapacitySummary, AssessmentError> {
    #[cfg(not(feature = "mtmd"))]
    let _ = projectors;
    let mut accountant = MemoryAccountant::new(topology);
    for device in devices {
        let estimate = match measurement {
            Measurement::Initial => device.initial,
            Measurement::Selected => device.fitted,
        };
        let Some(estimate) = estimate else {
            continue;
        };
        accountant
            .record(native_memory_charge(
                MemoryChargeOwner::Target,
                device,
                estimate,
                true,
            ))
            .map_err(accounting_error)?;
    }
    if let Some(speculative_devices) = speculative_devices {
        for device in speculative_devices {
            let Some(estimate) = device.initial else {
                continue;
            };
            accountant
                .record(native_memory_charge(
                    MemoryChargeOwner::SpeculativeDraft,
                    device,
                    estimate,
                    speculative_includes_model,
                ))
                .map_err(accounting_error)?;
        }
    }
    #[cfg(feature = "mtmd")]
    for projector in projectors {
        accountant
            .record(projector_memory_charge(projector)?)
            .map_err(accounting_error)?;
    }
    Ok(capacity_summary_from_accounting(accountant.finish()))
}

fn assessed_intent(
    requested: &ExecutionIntent,
    report: &FitReport,
    measurement: Measurement,
) -> ExecutionIntent {
    let configuration = match measurement {
        Measurement::Initial => report.requested,
        Measurement::Selected => report.fitted,
    };
    let mut plan = requested.clone();
    plan.physical_context_size = configuration.resolved_context_tokens;
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
            usable_capacity_bytes: summary.usable_capacity_bytes,
            headroom_bytes: summary
                .usable_capacity_bytes
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
/// the pinned native planner itself uses global backend registration.
///
/// # Errors
///
/// Returns [`NativePlanningError`] for invalid options or bridge/report failures.
pub fn assess_model_with_backend(
    backend: &LlamaBackend,
    request: &PlanningRequest,
) -> Result<FitReport, NativePlanningError> {
    Ok(resolve_native_plan(backend, request, None, false)?.report)
}

/// Estimate a speculative model/context linked to the exact target execution context.
pub fn assess_linked_model_with_backend(
    backend: &LlamaBackend,
    request: &PlanningRequest,
    target: &PlanningRequest,
) -> Result<FitReport, NativePlanningError> {
    let target = resolve_native_plan(backend, target, None, false)?;
    Ok(resolve_native_plan(backend, request, Some(&target.native), false)?.report)
}

pub struct NativeParameterPlan {
    model: PathBuf,
    model_params: std::pin::Pin<Box<LlamaModelParams>>,
    context_params: LlamaContextParams,
    threads: NonZeroU32,
    threads_batch: NonZeroU32,
}

/// Exact native parameter objects used only for speculative capability preflight. Construction lives
/// beside ordinary load planning so speculative discovery cannot drift from runtime defaults.
pub struct SpeculativePreflightParameters {
    pub target_model_params: std::pin::Pin<Box<LlamaModelParams>>,
    pub draft_model_params: std::pin::Pin<Box<LlamaModelParams>>,
    pub target_context: LlamaContextParams,
    pub draft_context: LlamaContextParams,
}

struct NativePlanAssessment {
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

pub fn speculative_preflight_parameters(
    intent: &ExecutionIntent,
) -> Result<SpeculativePreflightParameters, AssessmentError> {
    if !matches!(
        intent.speculative,
        SpeculativeDecodingConfig::Enabled { .. }
    ) {
        return Err(AssessmentError::MissingMeasurements);
    }
    let target = native_parameter_plan(&planning_request(intent, false)?)?;
    let draft = native_parameter_plan(&planning_request(intent, true)?)?;
    Ok(SpeculativePreflightParameters {
        target_model_params: target.model_params,
        draft_model_params: draft.model_params,
        target_context: target.context_params,
        draft_context: draft.context_params,
    })
}

/// Build native parameters from intent without placement selection. Intended for native parity
/// tooling; model serving uses [`plan_load_with_backend`] so selected placement is retained.
pub fn native_parameters_for_intent(
    intent: &ExecutionIntent,
) -> Result<NativeParameterPlan, AssessmentError> {
    Ok(native_parameter_plan(&planning_request(intent, false)?)?)
}

fn native_parameter_plan(
    request: &PlanningRequest,
) -> Result<NativeParameterPlan, NativePlanningError> {
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

fn resolve_native_plan(
    _backend: &LlamaBackend,
    request: &PlanningRequest,
    linked_target: Option<&NativeParameterPlan>,
    capture_decode_workload: bool,
) -> Result<NativePlanAssessment, NativePlanningError> {
    let mut native = native_parameter_plan(request)?;
    let model_path = path_c_string(&request.model)?;
    let max_devices = llama_cpp_2::max_devices();
    let mut margins = expand_margins(&request.options.margins_bytes, max_devices)?;

    let report = if let Some(target) = linked_target {
        let target_path = path_c_string(&target.model)?;
        let linked = llama_cpp_2::model::params::fit::LinkedFitTarget {
            model_path: &target_path,
            model_params: target.model_params.as_ref().get_ref(),
            context_params: &target.context_params,
        };
        native
            .model_params
            .as_mut()
            .fit_params_report_linked(
                &model_path,
                &mut native.context_params,
                linked,
                &mut margins,
                request.options.minimum_context_tokens,
            )
            .map_err(NativePlanningError::NativeBridge)?
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
            .map_err(NativePlanningError::NativeBridge)?
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
            .map_err(NativePlanningError::NativeBridge)?
    };

    Ok(NativePlanAssessment { native, report })
}

fn native_model_params(options: &PlanningOptions) -> Result<LlamaModelParams, NativePlanningError> {
    let params = LlamaModelParams::default()
        .with_gpu_layers(match options.gpu_layers {
            GpuLayers::Auto => llama_cpp_2::model::params::LlamaGpuLayers::Auto,
            GpuLayers::All => llama_cpp_2::model::params::LlamaGpuLayers::All,
            GpuLayers::Count(value) => llama_cpp_2::model::params::LlamaGpuLayers::Count(value),
        })
        .with_use_mmap(options.use_mmap)
        .with_use_mlock(options.use_mlock)
        .with_load_mtp(options.load_mtp)
        .with_split_mode(match options.split_mode {
            SplitMode::None => llama_cpp_2::model::params::LlamaSplitMode::None,
            SplitMode::Layer => llama_cpp_2::model::params::LlamaSplitMode::Layer,
            SplitMode::Row => llama_cpp_2::model::params::LlamaSplitMode::Row,
            SplitMode::Tensor => llama_cpp_2::model::params::LlamaSplitMode::Tensor,
        });
    match &options.tensor_split {
        Some(weights) => params
            .with_tensor_split(weights)
            .map_err(|error| NativePlanningError::InvalidOptions(error.to_string())),
        None => Ok(params),
    }
}

fn native_context_params(options: &PlanningOptions) -> LlamaContextParams {
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
            PlanningContextType::Target => LlamaContextType::Default,
            PlanningContextType::Mtp => LlamaContextType::Mtp,
            PlanningContextType::SpeculativeDraft => LlamaContextType::Default,
        })
        .with_n_rs_seq(options.recurrent_snapshots)
        .with_n_outputs_max(options.maximum_outputs)
        .with_n_threads(threads.get().min(i32::MAX as u32) as i32)
        .with_n_threads_batch(threads_batch.get().min(i32::MAX as u32) as i32)
}

fn native_thread_counts(options: &PlanningOptions) -> (NonZeroU32, NonZeroU32) {
    let threads = options.threads.unwrap_or_else(|| {
        NonZeroU32::new(llama_cpp_2::model::params::fit::default_math_threads())
            .expect("native math-thread default is positive")
    });
    let threads_batch = options.threads_batch.unwrap_or(threads);
    (threads, threads_batch)
}

fn validate(request: &PlanningRequest) -> Result<(), NativePlanningError> {
    if !request.model.is_file() {
        return Err(NativePlanningError::InvalidModel(request.model.clone()));
    }
    let options = &request.options;
    if options.minimum_context_tokens == 0 {
        return Err(NativePlanningError::InvalidOptions(
            "minimum context must be greater than zero".to_owned(),
        ));
    }
    if options.batch_tokens == 0 || options.micro_batch_tokens == 0 {
        return Err(NativePlanningError::InvalidOptions(
            "batch and micro-batch sizes must be greater than zero".to_owned(),
        ));
    }
    if options.micro_batch_tokens > options.batch_tokens {
        return Err(NativePlanningError::InvalidOptions(
            "micro-batch size must not exceed batch size".to_owned(),
        ));
    }
    if options.sequence_count == 0 {
        return Err(NativePlanningError::InvalidOptions(
            "sequence count must be greater than zero".to_owned(),
        ));
    }
    if options
        .maximum_outputs
        .is_some_and(|outputs| outputs.get() > options.batch_tokens)
    {
        return Err(NativePlanningError::InvalidOptions(
            "maximum outputs must not exceed batch size".to_owned(),
        ));
    }
    if options.margins_bytes.is_empty() {
        return Err(NativePlanningError::InvalidOptions(
            "at least one memory margin is required".to_owned(),
        ));
    }
    if options.tensor_split.as_ref().is_some_and(|weights| {
        weights.is_empty()
            || weights
                .iter()
                .any(|weight| !weight.is_finite() || *weight < 0.0)
    }) {
        return Err(NativePlanningError::InvalidOptions(
            "tensor_split must contain finite, non-negative weights".to_owned(),
        ));
    }
    Ok(())
}

fn expand_margins(values: &[u64], count: usize) -> Result<Vec<usize>, NativePlanningError> {
    if values.len() != 1 && values.len() != count {
        return Err(NativePlanningError::InvalidOptions(format!(
            "provide one memory margin to broadcast or exactly {count}; received {}",
            values.len()
        )));
    }
    let convert = |value: u64| {
        usize::try_from(value).map_err(|_| {
            NativePlanningError::InvalidOptions(format!(
                "memory margin {value} does not fit this target's usize"
            ))
        })
    };
    if values.len() == 1 {
        return Ok(vec![convert(values[0])?; count]);
    }
    values.iter().copied().map(convert).collect()
}

fn path_c_string(path: &Path) -> Result<CString, NativePlanningError> {
    Ok(CString::new(path.to_string_lossy().as_bytes())?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use llama_cpp_2::model::params::fit::{FitAllocations, FitDeviceKind, FitMemoryEstimate};

    #[test]
    fn system_memory_policy_uses_fractional_reserves_with_absolute_floors() {
        let gib = 1024 * 1024 * 1024;
        assert_eq!(
            system_memory_thresholds(16 * gib),
            SystemMemoryThresholds {
                assess_reserve_bytes: 2 * gib,
                abort_reserve_bytes: gib,
            },
        );
        assert_eq!(
            system_memory_thresholds(64 * gib),
            SystemMemoryThresholds {
                assess_reserve_bytes: 64 * gib / 10,
                abort_reserve_bytes: 64 * gib / 20,
            },
        );
    }

    #[test]
    fn system_reserve_does_not_reduce_dedicated_device_capacity() {
        let policy = CapacityPolicy {
            reserve_bytes_per_domain: 2,
            system_reserve_bytes: Some(10),
        };
        assert_eq!(policy.reserve_for_domain(&MemoryDomainId::system()), 10);
        assert_eq!(policy.reserve_for_domain(&MemoryDomainId::new("device")), 2);
    }

    #[test]
    fn native_model_params_preserve_mtp_layer_loading_policy() {
        let mut options = PlanningOptions::default();
        assert!(!native_model_params(&options).unwrap().load_mtp());
        options.load_mtp = true;
        assert!(native_model_params(&options).unwrap().load_mtp());
    }

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

    fn test_topology(
        devices: Vec<DiscoveredDevice>,
        system_memory_bytes: u64,
        policy: CapacityPolicy,
        platform: &str,
        architecture: &str,
    ) -> MemoryTopology {
        let snapshot = hardware_snapshot_from_devices(
            devices,
            policy,
            HardwareEnvironment {
                native_build: "test".to_owned(),
                enabled_backends: Vec::new(),
                platform: platform.to_owned(),
                architecture: architecture.to_owned(),
                system_product_name: None,
                logical_cores: 8,
                system_memory: HardwareSystemMemory {
                    physical_capacity_bytes: system_memory_bytes,
                    physical_available_bytes: system_memory_bytes,
                    allocation_capacity_bytes: system_memory_bytes,
                    allocation_headroom_bytes: system_memory_bytes,
                    assess_reserve_bytes: 0,
                    abort_reserve_bytes: 0,
                },
            },
        );
        MemoryTopology::from_snapshot(&snapshot).expect("valid test memory topology")
    }

    #[test]
    fn product_identity_normalization_preserves_real_names_and_rejects_placeholders() {
        assert_eq!(
            normalize_product_name(b"  NVIDIA DGX Spark\0\n"),
            Some("NVIDIA DGX Spark".to_owned())
        );
        assert_eq!(
            normalize_product_name(b"NVIDIA_DGX_Spark"),
            Some("NVIDIA DGX Spark".to_owned())
        );
        assert_eq!(
            normalize_product_name(b"MacBook   Pro"),
            Some("MacBook Pro".to_owned())
        );
        assert_eq!(normalize_product_name(b"System Product Name\n"), None);
        assert_eq!(normalize_product_name(b"To Be Filled By O.E.M.\0"), None);
    }

    #[test]
    fn hardware_snapshot_deduplicates_backend_aliases_without_merging_duplicate_cards() {
        let gpu = |backend: &str, name: &str, native_index: usize, card: usize| {
            let mut device = discovered_device(
                backend,
                name,
                "Example GPU",
                HardwareDeviceKind::Gpu,
                16_000,
                12_000,
            );
            device.native_index = native_index;
            device.physical_id = Some(format!("0000:01:0{card}.0"));
            device
        };
        let mut cpu = discovered_device(
            "CPU",
            "CPU",
            "Example CPU",
            HardwareDeviceKind::Cpu,
            64_000,
            32_000,
        );
        cpu.native_index = 4;
        let snapshot = hardware_snapshot_from_devices(
            vec![
                cpu,
                gpu("CUDA", "CUDA0", 0, 0),
                gpu("CUDA", "CUDA1", 1, 1),
                gpu("Vulkan", "Vulkan0", 2, 0),
                gpu("Vulkan", "Vulkan1", 3, 1),
            ],
            CapacityPolicy {
                reserve_bytes_per_domain: 1_000,
                system_reserve_bytes: None,
            },
            HardwareEnvironment {
                native_build: "build".to_owned(),
                enabled_backends: vec!["vulkan".to_owned(), "cuda".to_owned()],
                platform: "linux".to_owned(),
                architecture: "x86_64".to_owned(),
                system_product_name: None,
                logical_cores: 8,
                system_memory: HardwareSystemMemory {
                    physical_capacity_bytes: 64_000,
                    physical_available_bytes: 32_000,
                    allocation_capacity_bytes: 64_000,
                    allocation_headroom_bytes: 32_000,
                    assess_reserve_bytes: 0,
                    abort_reserve_bytes: 0,
                },
            },
        );
        assert_eq!(snapshot.memory_domains.len(), 3);
        assert_eq!(snapshot.memory_domains[0].id, MemoryDomainId::system());
        for domain in &snapshot.memory_domains[1..] {
            assert_eq!(domain.total_capacity_bytes, 16_000);
            assert_eq!(domain.stable_capacity_bytes, 15_000);
            assert_eq!(domain.devices.len(), 2);
        }
        assert_eq!(snapshot.enabled_backends, vec!["cuda", "vulkan"]);
        assert_eq!(snapshot.system_product_name, None);
        let topology = MemoryTopology::from_snapshot(&snapshot).expect("valid topology");
        assert_eq!(topology.system_domain(), &MemoryDomainId::system(),);
        assert_eq!(
            topology
                .resolve(&MemoryLocation::NativeDevice(NativeDeviceLocator::exact(
                    "CUDA",
                    Some("0000:01:00.0"),
                    0,
                )))
                .expect("CUDA device")
                .memory_domain,
            &snapshot.memory_domains[1].id,
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
                system_reserve_bytes: None,
            },
            HardwareEnvironment {
                native_build: "build".to_owned(),
                enabled_backends: vec!["cpu".to_owned(), "cuda".to_owned(), "vulkan".to_owned()],
                platform: "linux".to_owned(),
                architecture: "x86_64".to_owned(),
                system_product_name: None,
                logical_cores: 8,
                system_memory: HardwareSystemMemory {
                    physical_capacity_bytes: 64_000,
                    physical_available_bytes: 32_000,
                    allocation_capacity_bytes: 64_000,
                    allocation_headroom_bytes: 32_000,
                    assess_reserve_bytes: 0,
                    abort_reserve_bytes: 0,
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
        let blas_domain = snapshot
            .memory_domains
            .iter()
            .find(|domain| domain.devices.iter().any(|device| device.backend == "BLAS"))
            .expect("BLAS is represented as a host-memory-backed device");
        assert_eq!(blas_domain.id, MemoryDomainId::system());
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
                system_product_name: Some("MacBook Pro".to_owned()),
                logical_cores: 8,
                system_memory: HardwareSystemMemory {
                    physical_capacity_bytes: 64_000,
                    physical_available_bytes: 40_000,
                    allocation_capacity_bytes: 64_000,
                    allocation_headroom_bytes: 40_000,
                    assess_reserve_bytes: 0,
                    abort_reserve_bytes: 0,
                },
            },
        );
        let mut changed = devices.into_iter().rev().collect::<Vec<_>>();
        changed[0].free_bytes = Some(1);
        changed[1].free_bytes = Some(2);
        changed[0].name = "renamed device".to_owned();
        changed[0].description = "changed presentation description".to_owned();
        let second = hardware_snapshot_from_devices(
            changed,
            CapacityPolicy::default(),
            HardwareEnvironment {
                native_build: "build".to_owned(),
                enabled_backends: vec!["cpu".to_owned(), "metal".to_owned()],
                platform: "macos".to_owned(),
                architecture: "aarch64".to_owned(),
                system_product_name: Some("MacBook Pro".to_owned()),
                logical_cores: 8,
                system_memory: HardwareSystemMemory {
                    physical_capacity_bytes: 64_000,
                    physical_available_bytes: 1,
                    allocation_capacity_bytes: 64_000,
                    allocation_headroom_bytes: 1,
                    assess_reserve_bytes: 0,
                    abort_reserve_bytes: 0,
                },
            },
        );
        assert_eq!(first.topology_fingerprint, second.topology_fingerprint);
        assert_eq!(first.system_product_name.as_deref(), Some("MacBook Pro"));
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
        assert_eq!(first.system_memory.physical_capacity_bytes, 64_000);
        assert_ne!(
            first.memory_domains[0].current_free_bytes,
            second.memory_domains[0].current_free_bytes
        );
    }

    #[test]
    fn capacity_policy_selects_one_stable_fingerprinted_snapshot() {
        let snapshot = hardware_snapshot_from_devices(
            vec![
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
            ],
            CapacityPolicy {
                reserve_bytes_per_domain: 1_000,
                system_reserve_bytes: None,
            },
            HardwareEnvironment {
                native_build: "build".to_owned(),
                enabled_backends: vec!["cpu".to_owned(), "metal".to_owned()],
                platform: "macos".to_owned(),
                architecture: "aarch64".to_owned(),
                system_product_name: Some("MacBook Pro".to_owned()),
                logical_cores: 8,
                system_memory: HardwareSystemMemory {
                    physical_capacity_bytes: 64_000,
                    physical_available_bytes: 40_000,
                    allocation_capacity_bytes: 64_000,
                    allocation_headroom_bytes: 40_000,
                    assess_reserve_bytes: 0,
                    abort_reserve_bytes: 0,
                },
            },
        );
        let original_fingerprint = snapshot.topology_fingerprint.clone();
        let selected = with_capacity_policy(
            snapshot,
            CapacityPolicy {
                reserve_bytes_per_domain: 2_000,
                system_reserve_bytes: Some(10_000),
            },
        );

        assert_eq!(selected.memory_domains[0].total_capacity_bytes, 64_000);
        assert_eq!(selected.memory_domains[0].stable_capacity_bytes, 54_000);
        assert_eq!(selected.system_memory.physical_available_bytes, 40_000);
        let metal = selected.memory_domains[0]
            .devices
            .iter()
            .find(|device| device.backend == "Metal")
            .expect("Metal device");
        assert_eq!(
            metal.memory_limit,
            Some(HardwareDeviceMemoryLimit {
                kind: HardwareDeviceMemoryLimitKind::RecommendedWorkingSet,
                total_bytes: 48_000,
                stable_bytes: 38_000,
                current_free_bytes: Some(20_000),
            })
        );
        assert_ne!(selected.topology_fingerprint, original_fingerprint);

        let repeated = with_capacity_policy(
            selected.clone(),
            CapacityPolicy {
                reserve_bytes_per_domain: 2_000,
                system_reserve_bytes: Some(10_000),
            },
        );
        assert_eq!(repeated, selected);
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
                system_product_name: None,
                logical_cores: 8,
                system_memory: HardwareSystemMemory {
                    physical_capacity_bytes: 64_000,
                    physical_available_bytes: 40_000,
                    allocation_capacity_bytes: 64_000,
                    allocation_headroom_bytes: 40_000,
                    assess_reserve_bytes: 0,
                    abort_reserve_bytes: 0,
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
        let policy = CapacityPolicy {
            reserve_bytes_per_domain: 100,
            system_reserve_bytes: None,
        };
        let topology = test_topology(Vec::new(), 1_000, policy, "linux", "x86_64");
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
        let summary =
            capacity_summary(&topology, &[device], Measurement::Initial, None, false, &[]).unwrap();
        assert!(summary.fits);
        assert_eq!(summary.usable_capacity_bytes, 900);
        assert_eq!(summary.required_bytes, 400);
        assert_eq!(summary.domains[0].memory_domain, MemoryDomainId::system());
    }

    #[test]
    fn assessment_capacity_uses_exact_physical_identity_across_backend_views() {
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
        let policy = CapacityPolicy {
            reserve_bytes_per_domain: 1_000,
            system_reserve_bytes: None,
        };
        let mut cuda = discovered_device(
            "CUDA",
            "CUDA0",
            "Example GPU",
            HardwareDeviceKind::Gpu,
            16_000,
            12_000,
        );
        cuda.physical_id = Some("0000:01:00.0".to_owned());
        let mut vulkan = discovered_device(
            "Vulkan",
            "Vulkan0",
            "Example GPU",
            HardwareDeviceKind::Gpu,
            16_000,
            12_000,
        );
        vulkan.native_index = 1;
        vulkan.physical_id = Some("0000:01:00.0".to_owned());
        let mut cpu = discovered_device("CPU", "CPU", "Example CPU", HardwareDeviceKind::Cpu, 0, 0);
        cpu.native_index = 2;
        let topology = test_topology(vec![cuda, vulkan, cpu], 0, policy, "linux", "x86_64");
        let summary = capacity_summary(
            &topology,
            &[device(0, "CUDA", 4_000), device(1, "Vulkan", 3_000)],
            Measurement::Initial,
            None,
            false,
            &[],
        )
        .expect("capacity summary");

        assert_eq!(summary.usable_capacity_bytes, 15_000);
        assert_eq!(summary.required_bytes, 7_000);
        assert_eq!(summary.domains.len(), 2);
    }

    #[test]
    fn dedicated_assessment_keeps_an_explicit_zero_byte_system_domain() {
        let device =
            |index, kind, name: &str, device_id, total_bytes, required_bytes| FitDeviceEstimate {
                index,
                kind,
                backend_type: 0,
                backend: if kind == FitDeviceKind::Host {
                    "CPU".to_owned()
                } else {
                    "CUDA".to_owned()
                },
                device_id,
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
            };
        let policy = CapacityPolicy {
            reserve_bytes_per_domain: 1_000,
            system_reserve_bytes: Some(2_000),
        };
        let mut cuda = discovered_device(
            "CUDA",
            "CUDA0",
            "CUDA0",
            HardwareDeviceKind::Gpu,
            16_000,
            16_000,
        );
        cuda.physical_id = Some("0000:01:00.0".to_owned());
        let mut cpu =
            discovered_device("CPU", "CPU", "CPU", HardwareDeviceKind::Cpu, 64_000, 64_000);
        cpu.native_index = 1;
        let topology = test_topology(vec![cuda, cpu], 64_000, policy, "linux", "x86_64");
        let summary = capacity_summary(
            &topology,
            &[
                device(
                    0,
                    FitDeviceKind::Accelerator,
                    "CUDA0",
                    Some("0000:01:00.0".to_owned()),
                    16_000,
                    8_000,
                ),
                device(1, FitDeviceKind::Host, "CPU", None, 64_000, 0),
            ],
            Measurement::Initial,
            None,
            false,
            &[],
        )
        .expect("capacity summary");

        let system = summary
            .domains
            .iter()
            .find(|domain| domain.memory_domain.is_system())
            .expect("explicit system-memory domain");
        assert_eq!(system.required_bytes, 0);
        assert_eq!(system.usable_capacity_bytes, 62_000);
        assert_eq!(summary.domains.len(), 2);
    }

    #[test]
    fn integrated_gpu_and_host_share_one_assessment_capacity_domain() {
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
        let policy = CapacityPolicy {
            reserve_bytes_per_domain: 2_000,
            system_reserve_bytes: None,
        };
        let mut cpu =
            discovered_device("CPU", "CPU", "CPU", HardwareDeviceKind::Cpu, 32_000, 32_000);
        cpu.native_index = 1;
        let topology = test_topology(
            vec![
                discovered_device(
                    "Vulkan",
                    "iGPU",
                    "iGPU",
                    HardwareDeviceKind::IntegratedGpu,
                    8_000,
                    8_000,
                ),
                cpu,
            ],
            32_000,
            policy,
            "linux",
            "x86_64",
        );
        let mut mtp = device(0, FitDeviceKind::Accelerator, 2, "iGPU", 8_000, 0);
        mtp.initial
            .as_mut()
            .expect("MTP estimate")
            .allocations
            .context_bytes = 1_000;
        mtp.initial
            .as_mut()
            .expect("MTP estimate")
            .allocations
            .total_bytes = 1_000;
        let summary = capacity_summary(
            &topology,
            &[
                device(0, FitDeviceKind::Accelerator, 2, "iGPU", 8_000, 4_000),
                device(1, FitDeviceKind::Host, 0, "CPU", 32_000, 5_000),
            ],
            Measurement::Initial,
            Some(&[mtp]),
            false,
            &[],
        )
        .expect("capacity summary");

        assert_eq!(summary.usable_capacity_bytes, 30_000);
        assert_eq!(summary.required_bytes, 10_000);
        assert_eq!(summary.domains.len(), 1);
        assert_eq!(summary.domains[0].memory_domain, MemoryDomainId::system());
    }

    #[test]
    fn host_accelerator_and_cpu_share_one_assessment_capacity_domain() {
        let device_estimate = |index, kind, backend: &str, required_bytes| FitDeviceEstimate {
            index,
            kind,
            backend_type: 0,
            backend: backend.to_owned(),
            device_id: None,
            name: backend.to_owned(),
            description: backend.to_owned(),
            initial: Some(FitMemoryEstimate {
                total_bytes: 32_000,
                free_bytes: 32_000,
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
        let mut cpu =
            discovered_device("CPU", "CPU", "CPU", HardwareDeviceKind::Cpu, 32_000, 32_000);
        cpu.native_index = 1;
        let topology = test_topology(
            vec![
                discovered_device(
                    "BLAS",
                    "BLAS",
                    "host accelerator",
                    HardwareDeviceKind::Accelerator,
                    0,
                    0,
                ),
                cpu,
            ],
            32_000,
            CapacityPolicy {
                reserve_bytes_per_domain: 2_000,
                system_reserve_bytes: None,
            },
            "linux",
            "x86_64",
        );
        let summary = capacity_summary(
            &topology,
            &[
                device_estimate(0, FitDeviceKind::Accelerator, "BLAS", 4_000),
                device_estimate(1, FitDeviceKind::Host, "CPU", 5_000),
            ],
            Measurement::Initial,
            None,
            false,
            &[],
        )
        .expect("capacity summary");

        assert_eq!(summary.required_bytes, 9_000);
        assert_eq!(summary.domains.len(), 1);
        assert_eq!(summary.domains[0].memory_domain, MemoryDomainId::system());
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
        let policy = CapacityPolicy {
            reserve_bytes_per_domain: 4_000,
            system_reserve_bytes: None,
        };
        let mut cpu = discovered_device(
            "CPU",
            "CPU",
            "Apple CPU",
            HardwareDeviceKind::Cpu,
            64_000,
            50_000,
        );
        cpu.native_index = 1;
        let topology = test_topology(
            vec![
                discovered_device(
                    "MTL",
                    "MTL0",
                    "Apple GPU",
                    HardwareDeviceKind::Gpu,
                    48_000,
                    40_000,
                ),
                cpu,
            ],
            64_000,
            policy,
            "macos",
            "aarch64",
        );
        let summary = capacity_summary(
            &topology,
            &[
                device(0, FitDeviceKind::Accelerator, "MTL0", 48_000, 20_000),
                device(1, FitDeviceKind::Host, "CPU", 64_000, 5_000),
            ],
            Measurement::Initial,
            None,
            false,
            &[],
        )
        .unwrap();
        assert_eq!(summary.usable_capacity_bytes, 60_000);
        assert_eq!(summary.required_bytes, 25_000);
        assert_eq!(summary.domains[0].memory_domain, MemoryDomainId::system());
        assert_eq!(summary.device_constraints.len(), 1);
        assert_eq!(
            summary.device_constraints[0].device_id,
            native_device_id(&NativeDeviceIdentity::new("MTL", None::<String>, 0))
        );
        assert_eq!(summary.device_constraints[0].usable_capacity_bytes, 44_000);
        assert_eq!(summary.device_constraints[0].required_bytes, 20_000);

        let mut validation_cpu = discovered_device(
            "CPU",
            "CPU",
            "Apple CPU",
            HardwareDeviceKind::Cpu,
            64_000,
            50_000,
        );
        validation_cpu.native_index = 1;
        let snapshot = hardware_snapshot_from_devices(
            vec![
                validation_cpu,
                discovered_device(
                    "MTL",
                    "MTL0",
                    "Apple GPU",
                    HardwareDeviceKind::Gpu,
                    48_000,
                    40_000,
                ),
            ],
            CapacityPolicy {
                reserve_bytes_per_domain: 4_000,
                system_reserve_bytes: None,
            },
            HardwareEnvironment {
                native_build: "build".to_owned(),
                enabled_backends: vec!["metal".to_owned()],
                platform: "macos".to_owned(),
                architecture: "aarch64".to_owned(),
                system_product_name: Some("Mac".to_owned()),
                logical_cores: 8,
                system_memory: HardwareSystemMemory {
                    physical_capacity_bytes: 64_000,
                    physical_available_bytes: 50_000,
                    allocation_capacity_bytes: 64_000,
                    allocation_headroom_bytes: 50_000,
                    assess_reserve_bytes: 0,
                    abort_reserve_bytes: 0,
                },
            },
        );
        let topology =
            icn_contracts::MemoryTopology::from_snapshot(&snapshot).expect("valid topology");
        let assessment = HardwareAssessment::Fits {
            profile: HardwareProfile {
                context_length: 8_192,
                acceleration: "gpu".to_owned(),
                device: "system".to_owned(),
            },
            memory: HardwareMemory {
                required_bytes: summary.required_bytes,
                usable_capacity_bytes: summary.usable_capacity_bytes,
                headroom_bytes: summary
                    .usable_capacity_bytes
                    .saturating_sub(summary.required_bytes),
                domains: summary.domains,
                device_constraints: summary.device_constraints,
            },
            recommendation: HardwareRecommendation::Recommended,
        };
        assert!(topology.validates_hardware_assessment(&assessment));
    }

    #[test]
    fn native_device_identity_normalizes_backend_aliases() {
        assert_eq!(
            native_device_id(&NativeDeviceIdentity::new("MTL", Some("gpu-0"), 1)),
            native_device_id(&NativeDeviceIdentity::new("Metal", Some("gpu-0"), 1))
        );
        assert_ne!(
            native_device_id(&NativeDeviceIdentity::new("Metal", Some("gpu-0"), 1)),
            native_device_id(&NativeDeviceIdentity::new("Metal", Some("gpu-0"), 2))
        );
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
        let policy = CapacityPolicy {
            reserve_bytes_per_domain: 2_000,
            system_reserve_bytes: None,
        };
        let mut cpu = discovered_device(
            "CPU",
            "CPU",
            "Apple M4 Max",
            HardwareDeviceKind::Cpu,
            64_000,
            64_000,
        );
        cpu.native_index = 1;
        let topology = test_topology(
            vec![
                discovered_device(
                    "MTL",
                    "MTL0",
                    "Apple M4 Max",
                    HardwareDeviceKind::Gpu,
                    48_000,
                    48_000,
                ),
                cpu,
            ],
            64_000,
            policy,
            "macos",
            "aarch64",
        );
        let summary = capacity_summary(
            &topology,
            &[
                device(0, FitDeviceKind::Accelerator, "MTL", 48_000, 47_000),
                device(1, FitDeviceKind::Host, "CPU", 64_000, 1_000),
            ],
            Measurement::Initial,
            None,
            false,
            &[],
        )
        .expect("capacity summary");

        assert_eq!(summary.required_bytes, 48_000);
        assert_eq!(summary.usable_capacity_bytes, 62_000);
        assert!(!summary.fits);
        assert_eq!(summary.deficit_bytes, 1_000);
        assert_eq!(summary.limiting_resource, "MTL0");
        assert_eq!(summary.device_constraints[0].margin_bytes, -1_000);
    }

    #[test]
    fn built_in_mtp_adds_context_and_compute_but_not_duplicate_model_bytes() {
        let topology = test_topology(
            Vec::new(),
            2_000,
            CapacityPolicy {
                reserve_bytes_per_domain: 0,
                system_reserve_bytes: None,
            },
            "linux",
            "x86_64",
        );
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
            &topology,
            std::slice::from_ref(&target),
            Measurement::Initial,
            Some(std::slice::from_ref(&mtp)),
            false,
            &[],
        )
        .unwrap();
        assert_eq!(summary.required_bytes, 700);
        assert_eq!(summary.domains.len(), 1);
        assert_eq!(summary.domains[0].memory_domain, MemoryDomainId::system());
    }

    #[test]
    fn separate_mtp_model_storage_is_charged_once() {
        let topology = test_topology(
            Vec::new(),
            3_000,
            CapacityPolicy {
                reserve_bytes_per_domain: 0,
                system_reserve_bytes: None,
            },
            "linux",
            "x86_64",
        );
        let device = |model_bytes, context_bytes| FitDeviceEstimate {
            index: 0,
            kind: FitDeviceKind::Host,
            backend_type: 0,
            backend: "CPU".to_owned(),
            device_id: None,
            name: "CPU".to_owned(),
            description: "CPU".to_owned(),
            initial: Some(FitMemoryEstimate {
                total_bytes: 3_000,
                free_bytes: 3_000,
                allocations: FitAllocations {
                    model_bytes,
                    context_bytes,
                    compute_bytes: 0,
                    total_bytes: model_bytes + context_bytes,
                },
                target: None,
            }),
            fitted: None,
            margin_bytes: None,
        };
        let summary = capacity_summary(
            &topology,
            &[device(500, 100)],
            Measurement::Initial,
            Some(&[device(300, 50)]),
            true,
            &[],
        )
        .expect("capacity summary");

        assert_eq!(summary.required_bytes, 950);
        assert_eq!(summary.domains[0].model_bytes, 800);
    }

    #[cfg(feature = "mtmd")]
    #[test]
    fn projector_auxiliary_memory_resolves_through_topology() {
        let mut cpu =
            discovered_device("CPU", "CPU", "CPU", HardwareDeviceKind::Cpu, 32_000, 32_000);
        cpu.native_index = 1;
        let topology = test_topology(
            vec![
                discovered_device(
                    "Vulkan",
                    "iGPU",
                    "iGPU",
                    HardwareDeviceKind::IntegratedGpu,
                    8_000,
                    8_000,
                ),
                cpu,
            ],
            32_000,
            CapacityPolicy {
                reserve_bytes_per_domain: 0,
                system_reserve_bytes: None,
            },
            "linux",
            "x86_64",
        );
        let summary = capacity_summary(
            &topology,
            &[],
            Measurement::Initial,
            None,
            false,
            &[ProjectorMemory {
                device_index: Some(0),
                backend_type: 2,
                device_name: "iGPU".to_owned(),
                device_description: "iGPU".to_owned(),
                bytes: 700,
            }],
        )
        .expect("projector capacity summary");

        assert_eq!(summary.domains.len(), 1);
        assert_eq!(summary.domains[0].memory_domain, MemoryDomainId::system());
        assert_eq!(summary.domains[0].auxiliary_bytes, 700);
    }

    #[test]
    fn dedicated_accelerator_mtp_joins_its_target_memory_domain() {
        let device = |model_bytes, context_bytes, compute_bytes| FitDeviceEstimate {
            index: 0,
            kind: FitDeviceKind::Accelerator,
            backend_type: 1,
            backend: "CUDA".to_owned(),
            device_id: Some("gpu-0".to_owned()),
            name: "CUDA0".to_owned(),
            description: "dedicated gpu".to_owned(),
            initial: Some(FitMemoryEstimate {
                total_bytes: 4_000,
                free_bytes: 3_000,
                allocations: FitAllocations {
                    model_bytes,
                    context_bytes,
                    compute_bytes,
                    total_bytes: model_bytes + context_bytes + compute_bytes,
                },
                target: None,
            }),
            fitted: None,
            margin_bytes: Some(0),
        };
        let policy = CapacityPolicy {
            reserve_bytes_per_domain: 0,
            system_reserve_bytes: None,
        };
        let mut gpu = discovered_device(
            "CUDA",
            "CUDA0",
            "dedicated gpu",
            HardwareDeviceKind::Gpu,
            4_000,
            3_000,
        );
        gpu.physical_id = Some("gpu-0".to_owned());
        let mut cpu = discovered_device("CPU", "CPU", "CPU", HardwareDeviceKind::Cpu, 0, 0);
        cpu.native_index = 1;
        let topology = test_topology(vec![gpu, cpu], 0, policy, "linux", "x86_64");
        let summary = capacity_summary(
            &topology,
            &[device(500, 100, 50)],
            Measurement::Initial,
            Some(&[device(500, 40, 10)]),
            false,
            &[],
        )
        .expect("capacity summary");

        assert_eq!(summary.required_bytes, 700);
        assert_eq!(summary.domains.len(), 2);
        assert!(
            summary
                .domains
                .iter()
                .any(|domain| !domain.memory_domain.is_system() && domain.required_bytes == 700)
        );
    }

    #[test]
    fn dedicated_accelerator_mtp_requires_the_same_physical_device() {
        let device = |index, device_id: &str| FitDeviceEstimate {
            index,
            kind: FitDeviceKind::Accelerator,
            backend_type: 1,
            backend: "CUDA".to_owned(),
            device_id: Some(device_id.to_owned()),
            name: format!("CUDA{index}"),
            description: "dedicated gpu".to_owned(),
            initial: Some(FitMemoryEstimate {
                total_bytes: 4_000,
                free_bytes: 3_000,
                allocations: FitAllocations {
                    model_bytes: 500,
                    context_bytes: 100,
                    compute_bytes: 50,
                    total_bytes: 650,
                },
                target: None,
            }),
            fitted: None,
            margin_bytes: Some(0),
        };
        let policy = CapacityPolicy {
            reserve_bytes_per_domain: 0,
            system_reserve_bytes: None,
        };
        let mut gpu = discovered_device(
            "CUDA",
            "CUDA0",
            "dedicated gpu",
            HardwareDeviceKind::Gpu,
            4_000,
            3_000,
        );
        gpu.physical_id = Some("gpu-0".to_owned());
        let mut cpu = discovered_device("CPU", "CPU", "CPU", HardwareDeviceKind::Cpu, 0, 0);
        cpu.native_index = 2;
        let topology = test_topology(vec![gpu, cpu], 0, policy, "linux", "x86_64");
        let error = capacity_summary(
            &topology,
            &[device(0, "gpu-0")],
            Measurement::Initial,
            Some(&[device(1, "gpu-1")]),
            false,
            &[],
        )
        .expect_err("an unrelated accelerator must not be merged");

        assert!(matches!(
            error,
            AssessmentError::TopologyMismatch {
                owner: MemoryChargeOwner::SpeculativeDraft,
                backend: Some(ref backend),
                physical_id: Some(ref physical_id),
                native_index: 1,
            } if backend == "cuda" && physical_id == "gpu-1"
        ));
    }

    fn calibration_metric(
        backend: &str,
        device_id: &str,
        tensor_type: i32,
        routed: bool,
        bytes_per_second: f64,
    ) -> HardwareCalibrationMetric {
        HardwareCalibrationMetric {
            backend_type: 2,
            backend: backend.to_owned(),
            device_id: Some(device_id.to_owned()),
            tensor_type,
            routed,
            bytes_per_second,
            launch_microseconds: 0.0,
            relative_spread: 0.0,
            sample_count: 1,
            measured_microseconds: 1,
            stable: true,
        }
    }

    fn calibration(metrics: Vec<HardwareCalibrationMetric>) -> HardwareCalibration {
        HardwareCalibration {
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
            baseline_executed: true,
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
        use llama_cpp_2::model::params::fit::{FitAttentionRowWorkload, FitAttentionWorkload};
        llama_cpp_2::model::params::fit::FitKvLayerWorkload {
            layer,
            backend_type: 2,
            backend: backend.to_owned(),
            device_id: Some(device_id.to_owned()),
            attention: if row_bytes == 0 {
                FitAttentionWorkload::None
            } else {
                FitAttentionWorkload::Conventional {
                    key: FitAttentionRowWorkload {
                        tensor_type: 1,
                        bytes_per_token: row_bytes,
                    },
                    value: FitAttentionRowWorkload {
                        tensor_type: 1,
                        bytes_per_token: row_bytes,
                    },
                }
            },
            attention_head_size: row_bytes as u32,
            attention_state_type: 1,
            sliding_window_tokens,
            compression_ratio: 0,
            sparse_index: false,
            indexer_bytes_per_token: 0,
            recurrent,
            recurrent_type: 1,
            recurrent_conv_bytes: if recurrent { row_bytes } else { 0 },
            recurrent_state_bytes: if recurrent { row_bytes } else { 0 },
        }
    }

    fn workload(
        tensors: Vec<llama_cpp_2::model::params::fit::FitTensorWorkload>,
        kv_layers: Vec<llama_cpp_2::model::params::fit::FitKvLayerWorkload>,
        expert_count: u32,
        expert_used_count: u32,
    ) -> DecodeWorkload {
        DecodeWorkload {
            method: llama_cpp_2::model::params::fit::FIT_DECODE_WORKLOAD_METHOD.to_owned(),
            architecture: "test".to_owned(),
            expert_count,
            expert_used_count,
            nextn_layer_count: 0,
            kv_lora_rank: 0,
            indexer_head_count: 0,
            indexer_head_size: 0,
            indexer_top_k: 0,
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
        let mut cpu =
            discovered_device("CPU", "CPU", "CPU", HardwareDeviceKind::Cpu, 64_000, 64_000);
        cpu.physical_id = Some("CPU".to_owned());
        let mut metal = discovered_device(
            "Metal",
            "MTL0",
            "MTL0",
            HardwareDeviceKind::Gpu,
            48_000,
            48_000,
        );
        metal.native_index = 1;
        metal.physical_id = Some("MTL0".to_owned());
        let topology = test_topology(
            vec![cpu, metal],
            64_000,
            CapacityPolicy::default(),
            "macos",
            "aarch64",
        );

        assert!(!workload_crosses_memory_domains(&workload, &devices, &topology).unwrap());
    }

    #[test]
    fn linux_cpu_and_integrated_gpu_share_one_performance_memory_domain() {
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
            vec![kv_layer(0, 10, 0, false, "Vulkan", "iGPU")],
            0,
            0,
        );
        let devices = [
            performance_device(0, FitDeviceKind::Host, 2, "CPU", "CPU"),
            performance_device(1, FitDeviceKind::Accelerator, 2, "Vulkan", "iGPU"),
        ];
        let mut igpu = discovered_device(
            "Vulkan",
            "iGPU",
            "iGPU",
            HardwareDeviceKind::IntegratedGpu,
            16_000,
            16_000,
        );
        igpu.native_index = 1;
        igpu.physical_id = Some("iGPU".to_owned());
        let topology = test_topology(
            vec![
                discovered_device("CPU", "CPU", "CPU", HardwareDeviceKind::Cpu, 64_000, 64_000),
                igpu,
            ],
            64_000,
            CapacityPolicy::default(),
            "linux",
            "x86_64",
        );

        assert!(!workload_crosses_memory_domains(&workload, &devices, &topology).unwrap());
    }

    #[test]
    fn backend_views_of_one_gpu_share_one_performance_memory_domain() {
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
            vec![kv_layer(0, 10, 0, false, "Vulkan", "GPU0")],
            0,
            0,
        );
        workload.tensors[0].backend_type = 1;
        let devices = [
            performance_device(0, FitDeviceKind::Accelerator, 1, "CUDA", "GPU0"),
            performance_device(1, FitDeviceKind::Accelerator, 2, "Vulkan", "GPU0"),
        ];
        let mut cuda = discovered_device(
            "CUDA",
            "CUDA0",
            "GPU",
            HardwareDeviceKind::Gpu,
            16_000,
            16_000,
        );
        cuda.physical_id = Some("GPU0".to_owned());
        let mut vulkan = discovered_device(
            "Vulkan",
            "Vulkan0",
            "GPU",
            HardwareDeviceKind::Gpu,
            16_000,
            16_000,
        );
        vulkan.native_index = 1;
        vulkan.physical_id = Some("GPU0".to_owned());
        let mut cpu =
            discovered_device("CPU", "CPU", "CPU", HardwareDeviceKind::Cpu, 64_000, 64_000);
        cpu.native_index = 2;
        let topology = test_topology(
            vec![cuda, vulkan, cpu],
            64_000,
            CapacityPolicy::default(),
            "linux",
            "x86_64",
        );

        assert!(!workload_crosses_memory_domains(&workload, &devices, &topology).unwrap());
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
        let mut gpu0 = discovered_device(
            "CUDA",
            "GPU0",
            "GPU0",
            HardwareDeviceKind::Gpu,
            16_000,
            16_000,
        );
        gpu0.physical_id = Some("GPU0".to_owned());
        let mut gpu1 = discovered_device(
            "CUDA",
            "GPU1",
            "GPU1",
            HardwareDeviceKind::Gpu,
            16_000,
            16_000,
        );
        gpu1.native_index = 1;
        gpu1.physical_id = Some("GPU1".to_owned());
        let mut cpu =
            discovered_device("CPU", "CPU", "CPU", HardwareDeviceKind::Cpu, 64_000, 64_000);
        cpu.native_index = 2;
        let topology = test_topology(
            vec![gpu0, gpu1, cpu],
            64_000,
            CapacityPolicy::default(),
            "linux",
            "x86_64",
        );

        assert!(workload_crosses_memory_domains(&workload, &devices, &topology).unwrap());
    }

    fn estimated_parts(
        assessment: GenerationPerformanceAssessment,
    ) -> (
        GenerationPerformanceConfidence,
        u64,
        u64,
        bool,
        GenerationPerformanceAssessment,
    ) {
        (
            assessment.confidence,
            assessment.always_active_weight_bytes,
            assessment.routed_expert_weight_bytes,
            assessment.cross_memory_domain_placement,
            assessment,
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
        let short = estimate_generation_performance(&workload, &calibration, 10, false).unwrap();
        let (confidence, always_bytes, routed_bytes, hybrid, long) = estimated_parts(
            estimate_generation_performance(&workload, &calibration, 20, false).unwrap(),
        );
        assert_eq!(confidence, GenerationPerformanceConfidence::High);
        assert_eq!(always_bytes, 800);
        assert_eq!(routed_bytes, 0);
        assert!(!hybrid);
        assert_eq!(short.context_tokens, 10);
        assert_eq!(short.kv_bytes_read_per_token, 200);
        assert_eq!(long.context_tokens, 20);
        assert_eq!(long.kv_bytes_read_per_token, 400);
        assert!((short.expected_tokens_per_second - 0.82).abs() < 1e-12);
        assert!((long.expected_tokens_per_second - (0.82 / 1.2)).abs() < 1e-12);
        assert!(long.lower_tokens_per_second <= long.expected_tokens_per_second);
        assert!(long.expected_tokens_per_second <= long.upper_tokens_per_second);
    }

    #[test]
    fn estimator_returns_the_exact_assessed_context_above_200k() {
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

        let assessment =
            estimate_generation_performance(&workload, &calibration, 262_144, false).unwrap();

        assert_eq!(assessment.context_tokens, 262_144);
        assert_eq!(assessment.kv_bytes_read_per_token, 5_242_880);
    }

    #[test]
    fn unstable_calibration_lowers_generation_confidence() {
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
        let mut metric = calibration_metric("Metal", "MTL0", 1, false, 1_000.0);
        metric.stable = false;
        metric.relative_spread = 0.2;
        let calibration = calibration(vec![metric]);

        let (confidence, ..) = estimated_parts(
            estimate_generation_performance(&workload, &calibration, 20, false).unwrap(),
        );

        assert_eq!(confidence, GenerationPerformanceConfidence::Low);
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
            estimate_generation_performance(&workload, &calibration, 10, false).unwrap(),
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
        let (confidence, always_bytes, routed_bytes, _, estimate) = estimated_parts(
            estimate_generation_performance(&workload, &calibration, 10, false).unwrap(),
        );
        assert_eq!(confidence, GenerationPerformanceConfidence::Moderate);
        assert_eq!(always_bytes, 400);
        assert_eq!(routed_bytes, 800);
        // 400 always-active + 200 selected expert + 100 KV bytes at 1,000 B/s.
        assert!((estimate.expected_tokens_per_second - (0.75 / 0.7)).abs() < 1e-12);
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
        let short = estimate_generation_performance(&workload, &calibration, 10, false).unwrap();
        let long = estimate_generation_performance(&workload, &calibration, 20, false).unwrap();
        assert_eq!(short.kv_bytes_read_per_token, 200);
        assert_eq!(long.kv_bytes_read_per_token, 300);
        assert!(long.expected_tokens_per_second < short.expected_tokens_per_second);
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
        let (confidence, _, _, hybrid, estimate) = estimated_parts(
            estimate_generation_performance(&workload, &calibration, 10, true).unwrap(),
        );
        assert_eq!(confidence, GenerationPerformanceConfidence::Low);
        assert!(hybrid);
        // Unknown Metal type 7 uses the slower same-operation fallback (type 2 at 500 B/s).
        assert!((estimate.expected_tokens_per_second - (0.82 * 0.88 / 0.4)).abs() < 1e-12);
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
        let rate = |workload: &DecodeWorkload| {
            estimated_parts(
                estimate_generation_performance(workload, &calibration, 10, false).unwrap(),
            )
            .4
            .expected_tokens_per_second
        };

        let baseline = rate(&make_workload(100, 1));
        assert!(rate(&make_workload(200, 1)) < baseline);
        assert!(rate(&make_workload(100, 4)) < baseline);
    }

    #[test]
    fn invalid_moe_fails_but_dense_calibration_can_bound_routed_work() {
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
        let error = estimate_generation_performance(&invalid_moe, &dense_only, 10, false)
            .expect_err("invalid expert metadata must fail");
        assert_eq!(error.code, "invalid_expert_metadata");

        let valid_moe = DecodeWorkload {
            expert_used_count: 2,
            ..invalid_moe
        };
        let (confidence, _, _, _, _) = estimated_parts(
            estimate_generation_performance(&valid_moe, &dense_only, 10, false)
                .expect("dense calibration should provide a conservative routed fallback"),
        );
        assert_eq!(confidence, GenerationPerformanceConfidence::Low);

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
        let error = estimate_generation_performance(&dense, &malformed, 10, false)
            .expect_err("every calibration metric must be validated before use");
        assert_eq!(error.code, "invalid_calibration");
    }

    #[test]
    fn baseline_decode_excludes_stored_nextn_tensors() {
        let base = workload(
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
        let mut with_nextn = base.clone();
        let mut nextn = tensor(
            "blk.32.nextn.eh_proj.weight",
            FitTensorWorkloadKind::AlwaysActive,
            10_000,
            10_000,
            1,
            "Metal",
            "MTL0",
        );
        nextn.baseline_executed = false;
        with_nextn.tensors.push(nextn);
        with_nextn.nextn_layer_count = 1;
        let calibration = calibration(vec![calibration_metric("Metal", "MTL0", 1, false, 1_000.0)]);

        let base = estimated_parts(
            estimate_generation_performance(&base, &calibration, 10, false).unwrap(),
        );
        let with_nextn = estimated_parts(
            estimate_generation_performance(&with_nextn, &calibration, 10, false).unwrap(),
        );
        assert_eq!(base.1, with_nextn.1);
        assert_eq!(
            base.4.expected_tokens_per_second,
            with_nextn.4.expected_tokens_per_second
        );
    }

    #[test]
    fn recurrent_state_is_fixed_but_not_free() {
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
            vec![kv_layer(0, 0, 0, true, "Metal", "MTL0")],
            0,
            0,
        );
        recurrent.architecture = "qwen35".to_owned();
        recurrent.kv_layers[0].recurrent_conv_bytes = 10;
        recurrent.kv_layers[0].recurrent_state_bytes = 40;
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
            vec![kv_layer(0, 0, 0, false, "Metal", "MTL0")],
            0,
            0,
        );
        let calibration = calibration(vec![calibration_metric("Metal", "MTL0", 1, false, 1_000.0)]);
        let recurrent_short =
            estimate_generation_performance(&recurrent, &calibration, 10, false).unwrap();
        let recurrent = estimated_parts(
            estimate_generation_performance(&recurrent, &calibration, 20, false).unwrap(),
        );
        let dense = estimated_parts(
            estimate_generation_performance(&dense, &calibration, 20, false).unwrap(),
        );

        assert_eq!(
            recurrent_short.expected_tokens_per_second,
            recurrent.4.expected_tokens_per_second
        );
        assert!(recurrent.4.expected_tokens_per_second < dense.4.expected_tokens_per_second);
    }

    #[test]
    fn compressed_sparse_attention_scans_compressed_history_and_caps_gather() {
        let mut compressed = workload(
            vec![tensor(
                "output.weight",
                FitTensorWorkloadKind::AlwaysActive,
                100,
                100,
                1,
                "CUDA",
                "GPU0",
            )],
            vec![kv_layer(0, 4, 0, false, "CUDA", "GPU0")],
            0,
            0,
        );
        compressed.architecture = "deepseek4".to_owned();
        compressed.indexer_top_k = 512;
        compressed.kv_layers[0].compression_ratio = 4;
        compressed.kv_layers[0].sparse_index = true;
        compressed.kv_layers[0].indexer_bytes_per_token = 2;
        compressed.kv_layers[0].attention =
            llama_cpp_2::model::params::fit::FitAttentionWorkload::Mla {
                latent: llama_cpp_2::model::params::fit::FitAttentionRowWorkload {
                    tensor_type: 1,
                    bytes_per_token: 4,
                },
            };
        let calibration = calibration(vec![calibration_metric(
            "CUDA",
            "GPU0",
            1,
            false,
            1_000_000.0,
        )]);
        let short =
            estimate_generation_performance(&compressed, &calibration, 1_000, false).unwrap();
        let (confidence, _, _, _, long) = estimated_parts(
            estimate_generation_performance(&compressed, &calibration, 4_000, false).unwrap(),
        );

        assert_eq!(confidence, GenerationPerformanceConfidence::Moderate);
        assert_eq!(short.kv_bytes_read_per_token, 1_500);
        assert_eq!(long.kv_bytes_read_per_token, 4_048);
        assert!(long.expected_tokens_per_second < short.expected_tokens_per_second);
    }

    #[test]
    fn deepseek4_compressor_state_is_fixed_but_not_free() {
        let mut generic = workload(
            vec![tensor(
                "output.weight",
                FitTensorWorkloadKind::AlwaysActive,
                100,
                100,
                1,
                "CUDA",
                "GPU0",
            )],
            vec![kv_layer(0, 4, 0, false, "CUDA", "GPU0")],
            0,
            0,
        );
        generic.kv_layers[0].compression_ratio = 4;
        generic.kv_layers[0].attention_head_size = 16;
        let mut deepseek = generic.clone();
        deepseek.architecture = "deepseek4".to_owned();
        let calibration = calibration(vec![calibration_metric(
            "CUDA",
            "GPU0",
            1,
            false,
            1_000_000.0,
        )]);
        let generic_short =
            estimate_generation_performance(&generic, &calibration, 1_000, false).unwrap();
        let generic = estimated_parts(
            estimate_generation_performance(&generic, &calibration, 4_000, false).unwrap(),
        );
        let deepseek_short =
            estimate_generation_performance(&deepseek, &calibration, 1_000, false).unwrap();
        let deepseek = estimated_parts(
            estimate_generation_performance(&deepseek, &calibration, 4_000, false).unwrap(),
        );

        assert_eq!(
            generic_short.kv_bytes_read_per_token,
            deepseek_short.kv_bytes_read_per_token
        );
        assert_eq!(
            generic.4.kv_bytes_read_per_token,
            deepseek.4.kv_bytes_read_per_token
        );
        assert!(
            deepseek_short.expected_tokens_per_second < generic_short.expected_tokens_per_second
        );
        let short_penalty = 1.0 / deepseek_short.expected_tokens_per_second
            - 1.0 / generic_short.expected_tokens_per_second;
        let long_penalty = 1.0 / deepseek.4.expected_tokens_per_second
            - 1.0 / generic.4.expected_tokens_per_second;
        assert!((short_penalty - long_penalty).abs() < f64::EPSILON * 16.0);
    }

    #[test]
    fn dsa_index_scan_grows_after_sparse_attention_gather_is_capped() {
        let mut sparse = workload(
            vec![tensor(
                "output.weight",
                FitTensorWorkloadKind::AlwaysActive,
                100,
                100,
                1,
                "CUDA",
                "GPU0",
            )],
            vec![kv_layer(0, 4, 0, false, "CUDA", "GPU0")],
            0,
            0,
        );
        sparse.architecture = "glm-dsa".to_owned();
        sparse.indexer_top_k = 512;
        sparse.kv_layers[0].sparse_index = true;
        sparse.kv_layers[0].indexer_bytes_per_token = 2;
        let calibration = calibration(vec![calibration_metric(
            "CUDA",
            "GPU0",
            1,
            false,
            1_000_000.0,
        )]);
        let short = estimate_generation_performance(&sparse, &calibration, 1_000, false).unwrap();
        let long = estimate_generation_performance(&sparse, &calibration, 2_000, false).unwrap();

        assert_eq!(short.kv_bytes_read_per_token, 6_096);
        assert_eq!(long.kv_bytes_read_per_token, 8_096);
    }

    #[test]
    fn recurrent_and_attention_layers_charge_independent_context_traffic() {
        let hybrid = workload(
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
                kv_layer(0, 0, 0, true, "Metal", "MTL0"),
                kv_layer(1, 5, 0, false, "Metal", "MTL0"),
            ],
            0,
            0,
        );
        let calibration = calibration(vec![calibration_metric("Metal", "MTL0", 1, false, 1_000.0)]);
        let short = estimate_generation_performance(&hybrid, &calibration, 10, false).unwrap();
        let long = estimate_generation_performance(&hybrid, &calibration, 20, false).unwrap();
        assert_eq!(short.kv_bytes_read_per_token, 100);
        assert_eq!(long.kv_bytes_read_per_token, 200);
    }

    #[test]
    fn recurrent_state_rows_do_not_scale_with_context_depth() {
        let recurrent = workload(
            vec![tensor(
                "output.weight",
                FitTensorWorkloadKind::AlwaysActive,
                100,
                100,
                1,
                "Metal",
                "MTL0",
            )],
            vec![kv_layer(0, 5, 0, true, "Metal", "MTL0")],
            0,
            0,
        );
        let calibration = calibration(vec![calibration_metric("Metal", "MTL0", 1, false, 1_000.0)]);
        let short = estimate_generation_performance(&recurrent, &calibration, 10, false).unwrap();
        let (confidence, _, _, _, long) = estimated_parts(
            estimate_generation_performance(&recurrent, &calibration, 20, false).unwrap(),
        );
        assert_eq!(confidence, GenerationPerformanceConfidence::Moderate);
        assert_eq!(short.kv_bytes_read_per_token, 0);
        assert_eq!(long.kv_bytes_read_per_token, 0);
        assert_eq!(
            short.expected_tokens_per_second,
            long.expected_tokens_per_second
        );
    }

    #[test]
    fn layers_without_attention_kv_rows_do_not_add_context_traffic() {
        let no_attention = workload(
            vec![tensor(
                "output.weight",
                FitTensorWorkloadKind::AlwaysActive,
                100,
                100,
                1,
                "Metal",
                "MTL0",
            )],
            vec![kv_layer(0, 0, 0, false, "Metal", "MTL0")],
            0,
            0,
        );
        let calibration = calibration(vec![calibration_metric("Metal", "MTL0", 1, false, 1_000.0)]);
        let short =
            estimate_generation_performance(&no_attention, &calibration, 10, false).unwrap();
        let long = estimate_generation_performance(&no_attention, &calibration, 20, false).unwrap();
        assert_eq!(short.kv_bytes_read_per_token, 0);
        assert_eq!(long.kv_bytes_read_per_token, 0);
    }

    #[test]
    fn duplicate_native_kv_layer_identity_is_rejected() {
        let duplicate = workload(
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
                kv_layer(0, 1, 0, false, "Metal", "MTL0"),
                kv_layer(0, 1, 0, false, "Metal", "MTL0"),
            ],
            0,
            0,
        );
        let calibration = calibration(vec![calibration_metric("Metal", "MTL0", 1, false, 1_000.0)]);
        let error = estimate_generation_performance(&duplicate, &calibration, 10, false)
            .expect_err("duplicate KV layer identities must fail");
        assert_eq!(error.code, "invalid_native_workload");
    }

    #[test]
    fn kv_traffic_overflow_remains_typed() {
        let mut dense = workload(
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
        let calibration = calibration(vec![calibration_metric("Metal", "MTL0", 1, false, 1_000.0)]);

        let llama_cpp_2::model::params::fit::FitAttentionWorkload::Conventional { key, .. } =
            &mut dense.kv_layers[0].attention
        else {
            panic!("fixture must use conventional attention");
        };
        key.bytes_per_token = u64::MAX;
        let error = estimate_generation_performance(&dense, &calibration, 2, false)
            .expect_err("KV byte overflow must fail");
        assert_eq!(error.code, "workload_overflow");
    }
}
