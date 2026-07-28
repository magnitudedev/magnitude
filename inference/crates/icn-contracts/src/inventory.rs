//! Backend-neutral model inventory contracts.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use futures_util::future::BoxFuture;
use futures_util::stream::BoxStream;
use serde::{Deserialize, Serialize};

/// Stable identity of one physical memory pool.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct MemoryDomainId(String);

impl MemoryDomainId {
    const SYSTEM: &'static str = "system";

    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    #[must_use]
    pub fn system() -> Self {
        Self(Self::SYSTEM.to_owned())
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    #[must_use]
    pub fn is_system(&self) -> bool {
        self.0 == Self::SYSTEM
    }
}

impl std::fmt::Display for MemoryDomainId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

/// Stable identity of one native device view and its device-specific limits.
#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct HardwareDeviceId(String);

impl HardwareDeviceId {
    #[must_use]
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for HardwareDeviceId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

/// Source-scoped identity of one runnable model at one local location.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ModelId(pub String);

/// Content-derived identity shared by equivalent models at different locations.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ContentId(pub String);

impl ModelId {
    pub fn parse(value: impl Into<String>) -> Result<Self, InventoryError> {
        let value = value.into();
        validate_prefixed_digest(&value, "mdl_")?;
        Ok(Self(value))
    }
}

impl ContentId {
    pub fn parse(value: impl Into<String>) -> Result<Self, InventoryError> {
        let value = value.into();
        validate_prefixed_digest(&value, "content_")?;
        Ok(Self(value))
    }
}

fn validate_prefixed_digest(value: &str, prefix: &str) -> Result<(), InventoryError> {
    let Some(digest) = value.strip_prefix(prefix) else {
        return Err(InventoryError::InvalidId(value.to_owned()));
    };
    if digest.len() != 64
        || !digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(InventoryError::InvalidId(value.to_owned()));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComponentRole {
    Weights,
    Shard,
    Projector,
    Auxiliary,
    Draft,
    Mtp,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ContentIdentity {
    Sha256 { value: String },
    GitOid { value: String },
    Xet { value: String },
    FileIdentity { value: String },
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelComponent {
    pub path: PathBuf,
    pub role: ComponentRole,
    pub size_bytes: u64,
    pub content: ContentIdentity,
    pub shard_index: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relationship: Option<ComponentRelationship>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ComponentRelationship {
    ProjectorFor {
        projector: PathBuf,
        model: PathBuf,
    },
    DraftFor {
        draft: PathBuf,
        model: PathBuf,
        method: crate::models::SpeculativeMethod,
    },
    MtpFor {
        mtp: PathBuf,
        model: PathBuf,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Integrity {
    Verified { method: String },
    Unverified { reason: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ModelSource {
    HuggingFace {
        repository: String,
        requested_revision: String,
        commit: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metadata: Option<Box<HubMetadata>>,
    },
    Local {
        declared_by: LocalDeclaration,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalDeclaration {
    Configuration,
    Discovery,
    ActiveProcess,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct HubMetadata {
    pub access: Option<String>,
    pub author: Option<String>,
    pub license: Option<String>,
    pub pipeline_tag: Option<String>,
    pub library_name: Option<String>,
    pub tags: Vec<String>,
    pub downloads: Option<u64>,
    pub likes: Option<u64>,
    pub last_modified: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ModelLocation {
    MagnitudeCache {
        components: Vec<ModelComponent>,
        total_bytes: u64,
        integrity: Integrity,
    },
    HuggingFaceCache {
        cache_root: PathBuf,
        repository: String,
        commit: String,
        components: Vec<ModelComponent>,
        total_bytes: u64,
        integrity: Integrity,
    },
    Directory {
        source_id: String,
        root: PathBuf,
        components: Vec<ModelComponent>,
        total_bytes: u64,
        integrity: Integrity,
    },
    File {
        path: PathBuf,
        component: ModelComponent,
        integrity: Integrity,
    },
}

impl ModelLocation {
    pub fn components(&self) -> &[ModelComponent] {
        match self {
            Self::MagnitudeCache { components, .. }
            | Self::HuggingFaceCache { components, .. }
            | Self::Directory { components, .. } => components,
            Self::File { component, .. } => std::slice::from_ref(component),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ModelAvailability {
    Downloading {
        operation_id: String,
        stage: DownloadStage,
        completed_bytes: u64,
        total_bytes: u64,
        current_component: Option<PathBuf>,
        started_at: u64,
        updated_at: u64,
    },
    Interrupted {
        completed_bytes: u64,
        total_bytes: u64,
        resumable: bool,
        reason: Option<String>,
        last_error: String,
        updated_at: u64,
    },
    Available {
        ready_at: u64,
    },
    InvalidArtifact {
        detected_at: u64,
        code: String,
        message: String,
    },
    IncompatibleArtifact {
        detected_at: u64,
        code: String,
        message: String,
    },
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DownloadStage {
    Queued,
    Resolving,
    CheckingSpace,
    Downloading,
    Verifying,
    Publishing,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum CapabilitySupport {
    Supported { parallel: Option<bool> },
    Unsupported,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ReasoningCapability {
    Unsupported {
        evidence: CapabilityEvidence,
    },
    Supported {
        control: ReasoningControlDomain,
        visibility: ReasoningVisibility,
        delimiters: ReasoningDelimiters,
        evidence: CapabilityEvidence,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ReasoningControlDomain {
    Toggle {
        default: bool,
    },
    Effort {
        levels: Vec<String>,
        default: Option<String>,
    },
    Budget {
        min_tokens: u32,
        max_tokens: u32,
        default_tokens: u32,
    },
    EffortAndBudget {
        levels: Vec<String>,
        default_effort: Option<String>,
        min_tokens: u32,
        max_tokens: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningVisibility {
    Hidden,
    Preserved,
    Configurable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ReasoningDelimiters {
    Unavailable,
    Known { start: String, end: String },
}

/// Product-normalized reasoning selection. Native template spellings are kept in the compiled
/// mapping rather than exposed through this identifier.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct NormalizedReasoningEffort(pub String);

impl NormalizedReasoningEffort {
    pub fn parse(value: &str) -> Option<Self> {
        let normalized = match value {
            "off" | "no_think" | "disabled" => "none",
            "extra_high" | "extra-high" | "very_high" => "xhigh",
            "none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max" | "adaptive" => value,
            _ => return None,
        };
        Some(Self(normalized.to_owned()))
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AutomaticReasoningBudget {
    Disabled,
    FixedTokens { tokens: u32 },
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct NativeReasoningControls {
    /// `None` omits the native backend's dedicated control and preserves the authored template default.
    pub enable_thinking: Option<bool>,
    pub template_args: BTreeMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReasoningEffortMapping {
    pub effort: NormalizedReasoningEffort,
    pub controls: NativeReasoningControls,
    pub automatic_budget: AutomaticReasoningBudget,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReasoningProfile {
    pub default_effort: NormalizedReasoningEffort,
    pub mappings: Vec<ReasoningEffortMapping>,
    pub template_fingerprint: String,
}

impl ReasoningProfile {
    pub fn mapping(&self, effort: &NormalizedReasoningEffort) -> Option<&ReasoningEffortMapping> {
        self.mappings
            .iter()
            .find(|mapping| &mapping.effort == effort)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum CapabilityEvidence {
    NativeTemplate { fingerprint: String },
    BoundedTemplateProbe { fingerprint: String },
    DeclaredMetadata { source: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemplateAssessment {
    pub capabilities: crate::TemplateCapabilities,
    pub reasoning: ReasoningCapability,
    pub fingerprint: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectiveTemplateInputs {
    pub model_path: std::path::PathBuf,
}

/// Native chat-template assessment injected into model discovery.
pub trait TemplateAssessor: Send + Sync + 'static {
    /// Stable identity for every implementation and native-policy input that can change an
    /// assessment. This is cache evidence, not a persisted schema version.
    fn cache_identity(&self) -> &str;

    fn assess(&self, inputs: &EffectiveTemplateInputs) -> Result<TemplateAssessment, String>;
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
// This is a wire contract: introducing a nested payload solely to equalize in-memory variant
// sizes would make the serialized shape less direct for every API consumer.
#[allow(clippy::large_enum_variant)]
pub enum InventoryProperties {
    Pending,
    Unavailable {
        reason: String,
    },
    Inspected {
        architecture: Option<String>,
        quantization: Option<String>,
        quantization_name: Option<String>,
        parameter_count: Option<u64>,
        active_parameter_count: Option<u64>,
        training_context_length: Option<u32>,
        tokenizer: Option<String>,
        modalities: Vec<String>,
        base_models: Vec<String>,
        tools: CapabilitySupport,
        structured_output: CapabilitySupport,
        reasoning: ReasoningCapability,
        evidence_fingerprint: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum HardwareAssessment {
    NotAssessed {
        reason: String,
    },
    Fits {
        profile: HardwareProfile,
        memory: HardwareMemory,
        recommendation: HardwareRecommendation,
    },
    DoesNotFit {
        profile: HardwareProfile,
        memory: HardwareDeficit,
        limiting_resource: String,
        alternative: Option<HardwareProfile>,
    },
    InvalidArtifact {
        code: String,
        message: String,
    },
    IncompatibleArtifact {
        code: String,
        message: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerationPerformanceConfidence {
    High,
    Moderate,
    Low,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenerationSpeedPoint {
    pub context_tokens: u32,
    pub kv_bytes_read_per_token: u64,
    pub lower_tokens_per_second: f64,
    pub expected_tokens_per_second: f64,
    pub upper_tokens_per_second: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum GenerationPerformanceAssessment {
    Estimated {
        method: String,
        confidence: GenerationPerformanceConfidence,
        workload: String,
        always_active_weight_bytes: u64,
        routed_expert_weight_bytes: u64,
        expert_count: u32,
        expert_used_count: u32,
        cross_memory_domain_placement: bool,
        points: Vec<GenerationSpeedPoint>,
    },
    Unavailable {
        method: String,
        code: String,
        message: String,
    },
}

impl GenerationPerformanceAssessment {
    #[must_use]
    pub fn not_requested() -> Self {
        Self::Unavailable {
            method: "not_requested".to_owned(),
            code: "not_requested".to_owned(),
            message: "generation performance was not requested".to_owned(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelExecutionAssessment {
    pub hardware: HardwareAssessment,
    pub performance: GenerationPerformanceAssessment,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareProfile {
    pub context_length: u32,
    pub acceleration: String,
    pub device: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareMemory {
    pub required_bytes: u64,
    pub usable_capacity_bytes: u64,
    pub headroom_bytes: u64,
    pub domains: Vec<HardwareMemoryDomainAssessment>,
    #[serde(default)]
    pub device_constraints: Vec<HardwareDeviceMemoryAssessment>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareDeficit {
    pub required_bytes: u64,
    pub usable_capacity_bytes: u64,
    pub deficit_bytes: u64,
    pub domains: Vec<HardwareMemoryDomainAssessment>,
    #[serde(default)]
    pub device_constraints: Vec<HardwareDeviceMemoryAssessment>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareMemoryDomainAssessment {
    pub memory_domain: MemoryDomainId,
    pub model_bytes: u64,
    pub context_bytes: u64,
    pub compute_bytes: u64,
    pub auxiliary_bytes: u64,
    pub required_bytes: u64,
    pub usable_capacity_bytes: u64,
    pub margin_bytes: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareDeviceMemoryAssessment {
    pub device_id: HardwareDeviceId,
    pub device: String,
    pub kind: HardwareDeviceMemoryLimitKind,
    pub model_bytes: u64,
    pub context_bytes: u64,
    pub compute_bytes: u64,
    pub auxiliary_bytes: u64,
    pub required_bytes: u64,
    pub usable_capacity_bytes: u64,
    pub margin_bytes: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HardwareRecommendation {
    Recommended,
    Constrained,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HardwareMemoryDomainKind {
    System,
    PhysicalDevice,
    UnifiedMemory,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HardwareDeviceKind {
    Cpu,
    Gpu,
    IntegratedGpu,
    Accelerator,
    Unknown,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HardwareDeviceMemoryLimitKind {
    RecommendedWorkingSet,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareDeviceMemoryLimit {
    pub kind: HardwareDeviceMemoryLimitKind,
    pub total_bytes: u64,
    pub stable_bytes: u64,
    pub current_free_bytes: Option<u64>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareDevice {
    pub id: HardwareDeviceId,
    pub native_index: usize,
    pub backend: String,
    pub physical_id: Option<String>,
    pub name: String,
    pub description: String,
    pub kind: HardwareDeviceKind,
    #[serde(default)]
    pub memory_limit: Option<HardwareDeviceMemoryLimit>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareSystemMemory {
    pub total_bytes: u64,
    pub current_available_bytes: u64,
    pub warning_reserve_bytes: u64,
    pub assess_reserve_bytes: u64,
    pub abort_reserve_bytes: u64,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareMemoryDomain {
    pub id: MemoryDomainId,
    pub kind: HardwareMemoryDomainKind,
    pub total_capacity_bytes: u64,
    pub stable_capacity_bytes: u64,
    pub current_free_bytes: Option<u64>,
    pub shares_system_memory: bool,
    pub devices: Vec<HardwareDevice>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HardwareSnapshot {
    pub captured_at: u64,
    pub platform: String,
    pub architecture: String,
    #[serde(default)]
    pub system_product_name: Option<String>,
    pub cpu_model: Option<String>,
    pub logical_cores: usize,
    pub system_memory: HardwareSystemMemory,
    pub native_build: String,
    pub enabled_backends: Vec<String>,
    pub topology_fingerprint: String,
    pub memory_domains: Vec<HardwareMemoryDomain>,
}

/// Canonical physical memory pools used to interpret persisted assessments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryTopology {
    capacities: BTreeMap<MemoryDomainId, MemoryDomainCapacity>,
    device_limits: BTreeMap<HardwareDeviceId, HardwareDeviceMemoryLimit>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct MemoryDomainCapacity {
    total_bytes: u64,
    stable_bytes: u64,
}

impl MemoryTopology {
    #[must_use]
    pub fn from_snapshot(snapshot: &HardwareSnapshot) -> Option<Self> {
        Self::from_domains(&snapshot.memory_domains)
    }

    #[must_use]
    pub fn from_domains(domains: &[HardwareMemoryDomain]) -> Option<Self> {
        let mut capacities = BTreeMap::new();
        let mut device_limits = BTreeMap::new();
        let mut device_ids = BTreeSet::new();
        for domain in domains {
            if domain.id.as_str().is_empty()
                || domain.id.is_system() != domain.shares_system_memory
                || domain.stable_capacity_bytes > domain.total_capacity_bytes
            {
                return None;
            }
            if capacities
                .insert(
                    domain.id.clone(),
                    MemoryDomainCapacity {
                        total_bytes: domain.total_capacity_bytes,
                        stable_bytes: domain.stable_capacity_bytes,
                    },
                )
                .is_some()
            {
                return None;
            }
            for device in &domain.devices {
                if device.id.as_str().is_empty() || !device_ids.insert(device.id.clone()) {
                    return None;
                }
                let Some(limit) = &device.memory_limit else {
                    continue;
                };
                if limit.stable_bytes > limit.total_bytes
                    || device_limits
                        .insert(device.id.clone(), limit.clone())
                        .is_some()
                {
                    return None;
                }
            }
        }
        capacities
            .contains_key(&MemoryDomainId::system())
            .then_some(Self {
                capacities,
                device_limits,
            })
    }

    #[must_use]
    pub fn capacity(&self, domain: &MemoryDomainId) -> Option<u64> {
        self.capacities
            .get(domain)
            .map(|capacity| capacity.total_bytes)
    }

    #[must_use]
    pub fn validates_hardware_assessment(&self, assessment: &HardwareAssessment) -> bool {
        match assessment {
            HardwareAssessment::Fits { memory, .. } => {
                self.validates_hardware_domains(&memory.domains)
                    && self.validates_device_constraints(&memory.device_constraints)
                    && memory.required_bytes == sum_required_bytes(&memory.domains)
                    && memory.usable_capacity_bytes == sum_usable_capacity_bytes(&memory.domains)
                    && memory.headroom_bytes
                        == memory
                            .usable_capacity_bytes
                            .saturating_sub(memory.required_bytes)
            }
            HardwareAssessment::DoesNotFit { memory, .. } => {
                self.validates_hardware_domains(&memory.domains)
                    && self.validates_device_constraints(&memory.device_constraints)
                    && memory.required_bytes == sum_required_bytes(&memory.domains)
                    && memory.usable_capacity_bytes == sum_usable_capacity_bytes(&memory.domains)
                    && memory.deficit_bytes
                        == memory
                            .domains
                            .iter()
                            .map(|domain| {
                                domain
                                    .required_bytes
                                    .saturating_sub(domain.usable_capacity_bytes)
                            })
                            .chain(memory.device_constraints.iter().map(|constraint| {
                                constraint
                                    .required_bytes
                                    .saturating_sub(constraint.usable_capacity_bytes)
                            }))
                            .max()
                            .unwrap_or(0)
            }
            HardwareAssessment::InvalidArtifact { .. }
            | HardwareAssessment::IncompatibleArtifact { .. } => true,
            HardwareAssessment::NotAssessed { .. } => false,
        }
    }

    fn validates_hardware_domains(&self, domains: &[HardwareMemoryDomainAssessment]) -> bool {
        let mut seen = BTreeMap::new();
        for domain in domains {
            let Some(capacity) = self.capacities.get(&domain.memory_domain) else {
                return false;
            };
            let expected_required = domain
                .model_bytes
                .saturating_add(domain.context_bytes)
                .saturating_add(domain.compute_bytes)
                .saturating_add(domain.auxiliary_bytes);
            let expected_margin =
                (i128::from(domain.usable_capacity_bytes) - i128::from(domain.required_bytes))
                    .clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64;
            if seen.insert(domain.memory_domain.clone(), ()).is_some()
                || domain.usable_capacity_bytes != capacity.stable_bytes
                || domain.required_bytes != expected_required
                || domain.margin_bytes != expected_margin
            {
                return false;
            }
        }
        seen.contains_key(&MemoryDomainId::system())
    }

    fn validates_device_constraints(&self, constraints: &[HardwareDeviceMemoryAssessment]) -> bool {
        let mut seen = BTreeMap::new();
        constraints.iter().all(|constraint| {
            let Some(limit) = self.device_limits.get(&constraint.device_id) else {
                return false;
            };
            seen.insert(constraint.device_id.clone(), ()).is_none()
                && constraint.kind == limit.kind
                && constraint.usable_capacity_bytes == limit.stable_bytes
                && constraint.required_bytes
                    == constraint
                        .model_bytes
                        .saturating_add(constraint.context_bytes)
                        .saturating_add(constraint.compute_bytes)
                        .saturating_add(constraint.auxiliary_bytes)
                && constraint.margin_bytes
                    == (i128::from(constraint.usable_capacity_bytes)
                        - i128::from(constraint.required_bytes))
                    .clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64
        })
    }
}

fn sum_required_bytes(domains: &[HardwareMemoryDomainAssessment]) -> u64 {
    domains.iter().fold(0, |total, domain| {
        total.saturating_add(domain.required_bytes)
    })
}

fn sum_usable_capacity_bytes(domains: &[HardwareMemoryDomainAssessment]) -> u64 {
    domains.iter().fold(0, |total, domain| {
        total.saturating_add(domain.usable_capacity_bytes)
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelPreviewSource {
    pub repository: String,
    pub revision: String,
    pub primary_gguf: PathBuf,
    pub additional_components: Vec<ModelPreviewComponentSource>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelPreviewComponentSource {
    pub path: PathBuf,
    pub role: ModelPreviewComponentRole,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ModelPreviewComponentRole {
    Projector,
    Draft {
        method: crate::models::SpeculativeMethod,
    },
    Mtp,
}

impl ModelPreviewComponentRole {
    #[must_use]
    pub fn component_role(&self) -> ComponentRole {
        match self {
            Self::Projector => ComponentRole::Projector,
            Self::Draft { .. } => ComponentRole::Draft,
            Self::Mtp => ComponentRole::Mtp,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelPreviewProfile {
    pub id: String,
    pub context_length: u32,
    pub parallel_sequences: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelPreviewRequest {
    pub source: ModelPreviewSource,
    pub profiles: Vec<ModelPreviewProfile>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelPreviewAssessment {
    pub profile_id: String,
    pub artifact_fingerprint: String,
    pub hardware_topology: String,
    pub assessment: HardwareAssessment,
    pub performance: GenerationPerformanceAssessment,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelPreview {
    pub repository: String,
    pub commit: String,
    pub components: Vec<ModelComponent>,
    pub properties: InventoryProperties,
    pub assessments: Vec<ModelPreviewAssessment>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HuggingFaceModelSearchRequest {
    pub query: String,
    pub limit: u32,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HuggingFaceModelSearchResult {
    pub repository: String,
    pub commit: String,
    pub last_modified: Option<String>,
    pub downloads: Option<u64>,
    pub likes: Option<u64>,
    pub gated: bool,
    pub private: bool,
    pub tags: Vec<String>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HuggingFaceModelSearchResults {
    pub models: Vec<HuggingFaceModelSearchResult>,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HuggingFaceRepositoryRequest {
    pub repository: String,
    pub revision: String,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HuggingFaceRepositoryFile {
    #[cfg_attr(feature = "openapi", schema(value_type = String))]
    pub path: PathBuf,
    pub size_bytes: u64,
    pub content: ContentIdentity,
}

#[cfg_attr(feature = "openapi", derive(utoipa::ToSchema))]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HuggingFaceRepositorySnapshot {
    pub repository: String,
    pub commit: String,
    pub last_modified: Option<String>,
    pub downloads: Option<u64>,
    pub likes: Option<u64>,
    pub gated: bool,
    pub private: bool,
    pub license: Option<String>,
    pub license_url: Option<String>,
    pub base_models: Vec<String>,
    pub tags: Vec<String>,
    pub gguf_files: Vec<HuggingFaceRepositoryFile>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InventoryModel {
    pub id: ModelId,
    pub content_id: ContentId,
    pub created: u64,
    pub name: String,
    pub supported_parameters: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub serving_configuration: Option<ServingConfiguration>,
    pub availability: ModelAvailability,
    pub source: ModelSource,
    pub location: ModelLocation,
    pub properties: InventoryProperties,
    pub hardware: HardwareAssessment,
    pub operations: Vec<ModelOperation>,
    pub updated_at: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServingProfile {
    pub context_length: u32,
    pub parallel_sequences: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServingConfiguration {
    pub profile: ServingProfile,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedModel {
    pub model: InventoryModel,
    pub components: Vec<ResolvedComponent>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedComponent {
    pub path: PathBuf,
    pub role: ComponentRole,
    pub shard_index: Option<u32>,
    pub relationship: Option<ComponentRelationship>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelOperation {
    Load,
    Unload,
    Delete,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DownloadModelRequest {
    pub source: HuggingFaceDownloadSource,
    pub components: Vec<DownloadComponent>,
    #[serde(default)]
    pub relationships: Vec<ComponentRelationship>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum HuggingFaceDownloadSource {
    HuggingFace {
        repository: String,
        revision: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DownloadComponent {
    pub path: PathBuf,
    pub role: ComponentRole,
    pub shard_index: Option<u32>,
    pub expected_sha256: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ModelDownloadEvent {
    Resolving {
        operation_id: String,
        repository: String,
        revision: String,
    },
    CheckingSpace {
        operation_id: String,
        model_id: ModelId,
        required_bytes: u64,
        available_bytes: u64,
        completed_bytes: u64,
        total_bytes: u64,
    },
    Progress {
        operation_id: String,
        model_id: ModelId,
        stage: DownloadStage,
        completed_bytes: u64,
        total_bytes: u64,
        file: DownloadFileProgress,
        bytes_per_second: Option<f64>,
        resumed_from_bytes: u64,
    },
    Ready {
        operation_id: String,
        model: Box<InventoryModel>,
    },
    Failed {
        operation_id: String,
        model_id: Option<ModelId>,
        error: DownloadFailure,
        completed_bytes: u64,
        total_bytes: u64,
        resumable: bool,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DownloadFileProgress {
    pub path: PathBuf,
    pub completed_bytes: u64,
    pub total_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DownloadFailure {
    pub code: String,
    pub message: String,
    pub retryable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeletePlan {
    pub model_id: ModelId,
    pub supported: bool,
    pub reason: Option<String>,
    pub reclaimable_bytes: u64,
    pub retained_shared_bytes: u64,
    pub paths: Vec<PathBuf>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeletedModel {
    pub id: ModelId,
    pub deleted: bool,
    pub freed_bytes: u64,
    pub retained_shared_bytes: u64,
    pub plan: DeletePlan,
}

pub type DownloadEventStream = BoxStream<'static, ModelDownloadEvent>;

/// Model inventory boundary consumed by the HTTP API and server composition root.
pub trait ModelInventory: Send + Sync + 'static {
    fn list(&self) -> BoxFuture<'_, Result<Vec<InventoryModel>, InventoryError>>;
    fn get(&self, id: &ModelId) -> BoxFuture<'_, Result<InventoryModel, InventoryError>>;
    fn download(
        &self,
        request: DownloadModelRequest,
    ) -> BoxFuture<'_, Result<DownloadEventStream, InventoryError>>;
    fn plan_delete(&self, id: &ModelId) -> BoxFuture<'_, Result<DeletePlan, InventoryError>>;
    fn delete(&self, id: &ModelId) -> BoxFuture<'_, Result<DeletedModel, InventoryError>>;
    fn resolve_ready(&self, id: &ModelId) -> BoxFuture<'_, Result<ResolvedModel, InventoryError>>;
    fn configure_serving(
        &self,
        id: &ModelId,
        profile: ServingProfile,
    ) -> BoxFuture<'_, Result<InventoryModel, InventoryError>>;
}

/// Canonical inventory assessment implemented by the server composition root.
///
/// Keeping this boundary in contracts lets `icn-models` own reconciliation without depending on
/// the native planner or `icn-hardware`. The cache key covers the canonical execution policy, native build,
/// backend, and stable hardware topology. Assessment failures are operation failures, never model
/// properties.
pub trait InventoryHardwareAssessor: HardwareProvider {
    fn cache_key(&self, snapshot: &HardwareSnapshot) -> Result<String, InventoryError>;
    fn assess(
        &self,
        model: ResolvedModel,
    ) -> BoxFuture<'_, Result<HardwareAssessment, InventoryError>>;

    fn assess_serving(
        &self,
        model: ResolvedModel,
        _profile: ServingProfile,
    ) -> BoxFuture<'_, Result<HardwareAssessment, InventoryError>> {
        self.assess(model)
    }
}

pub trait HardwareProvider: Send + Sync + 'static {
    fn snapshot(&self) -> BoxFuture<'_, Result<HardwareSnapshot, InventoryError>>;
}

/// Canonical profile-aware model assessment used by inventory and remote preview.
pub trait ModelHardwareAssessor: HardwareProvider {
    fn cache_key(
        &self,
        profile: Option<&ModelPreviewProfile>,
        snapshot: &HardwareSnapshot,
    ) -> Result<String, InventoryError>;
    fn assess_profile(
        &self,
        model: ResolvedModel,
        profile: Option<ModelPreviewProfile>,
    ) -> BoxFuture<'_, Result<HardwareAssessment, InventoryError>>;

    fn assess_profiles(
        &self,
        model: ResolvedModel,
        profiles: Vec<ModelPreviewProfile>,
    ) -> BoxFuture<'_, Result<Vec<HardwareAssessment>, InventoryError>> {
        Box::pin(async move {
            let mut assessments = Vec::with_capacity(profiles.len());
            for profile in profiles {
                assessments.push(self.assess_profile(model.clone(), Some(profile)).await?);
            }
            Ok(assessments)
        })
    }

    fn assess_execution_profiles(
        &self,
        model: ResolvedModel,
        profiles: Vec<ModelPreviewProfile>,
    ) -> BoxFuture<'_, Result<Vec<ModelExecutionAssessment>, InventoryError>> {
        Box::pin(async move {
            Ok(self
                .assess_profiles(model, profiles)
                .await?
                .into_iter()
                .map(|hardware| ModelExecutionAssessment {
                    hardware,
                    performance: GenerationPerformanceAssessment::not_requested(),
                })
                .collect())
        })
    }
}

pub trait ModelPreviewer: Send + Sync + 'static {
    fn preview(
        &self,
        request: ModelPreviewRequest,
    ) -> BoxFuture<'_, Result<ModelPreview, InventoryError>>;
}

/// Live Hugging Face discovery. Resolved commits are immutable snapshots for a
/// subsequent preview or download, not catalog pins.
pub trait HuggingFaceModelCatalog: Send + Sync + 'static {
    fn search(
        &self,
        request: HuggingFaceModelSearchRequest,
    ) -> BoxFuture<'_, Result<HuggingFaceModelSearchResults, InventoryError>>;

    fn resolve(
        &self,
        request: HuggingFaceRepositoryRequest,
    ) -> BoxFuture<'_, Result<HuggingFaceRepositorySnapshot, InventoryError>>;
}

#[derive(Debug, thiserror::Error)]
pub enum InventoryError {
    #[error("invalid model id: {0}")]
    InvalidId(String),
    #[error("invalid model request: {0}")]
    InvalidRequest(String),
    #[error("model not found: {0}")]
    NotFound(String),
    #[error("model is not ready: {0}")]
    NotReady(String),
    #[error("model is busy: {0}")]
    Busy(String),
    #[error("model is loaded: {0}")]
    Loaded(String),
    #[error("deletion is unsafe: {0}")]
    DeletionUnsafe(String),
    #[error("model source does not support this operation: {0}")]
    Unsupported(String),
    #[error("inventory I/O failed: {0}")]
    Io(String),
    #[error("upstream model service failed: {0}")]
    Upstream(String),
    #[error("model integrity check failed: {0}")]
    Integrity(String),
    #[error("model artifacts changed during inspection: {0}")]
    ConcurrentMutation(String),
    #[error("{message}")]
    ModelOperation {
        code: String,
        message: String,
        retryable: bool,
    },
    #[error("internal inventory failure: {0}")]
    Internal(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_memory_topology() -> MemoryTopology {
        MemoryTopology::from_domains(&[HardwareMemoryDomain {
            id: MemoryDomainId::system(),
            kind: HardwareMemoryDomainKind::UnifiedMemory,
            total_capacity_bytes: 64,
            stable_capacity_bytes: 60,
            current_free_bytes: Some(40),
            shares_system_memory: true,
            devices: vec![HardwareDevice {
                id: HardwareDeviceId::new("metal-0"),
                native_index: 0,
                backend: "MTL".to_owned(),
                physical_id: Some("metal-0".to_owned()),
                name: "MTL0".to_owned(),
                description: "Test GPU".to_owned(),
                kind: HardwareDeviceKind::Gpu,
                memory_limit: Some(HardwareDeviceMemoryLimit {
                    kind: HardwareDeviceMemoryLimitKind::RecommendedWorkingSet,
                    total_bytes: 54,
                    stable_bytes: 50,
                    current_free_bytes: Some(30),
                }),
            }],
        }])
        .expect("valid topology")
    }

    fn fitting_assessment() -> HardwareAssessment {
        HardwareAssessment::Fits {
            profile: HardwareProfile {
                context_length: 8_192,
                acceleration: "gpu".to_owned(),
                device: "system".to_owned(),
            },
            memory: HardwareMemory {
                required_bytes: 20,
                usable_capacity_bytes: 60,
                headroom_bytes: 40,
                domains: vec![HardwareMemoryDomainAssessment {
                    memory_domain: MemoryDomainId::system(),
                    model_bytes: 10,
                    context_bytes: 4,
                    compute_bytes: 6,
                    auxiliary_bytes: 0,
                    required_bytes: 20,
                    usable_capacity_bytes: 60,
                    margin_bytes: 40,
                }],
                device_constraints: vec![HardwareDeviceMemoryAssessment {
                    device_id: HardwareDeviceId::new("metal-0"),
                    device: "MTL0".to_owned(),
                    kind: HardwareDeviceMemoryLimitKind::RecommendedWorkingSet,
                    model_bytes: 10,
                    context_bytes: 4,
                    compute_bytes: 6,
                    auxiliary_bytes: 0,
                    required_bytes: 20,
                    usable_capacity_bytes: 50,
                    margin_bytes: 30,
                }],
            },
            recommendation: HardwareRecommendation::Recommended,
        }
    }

    #[test]
    fn ids_require_the_exact_versioned_prefix_and_lowercase_digest() {
        let digest = "a".repeat(64);
        assert!(ModelId::parse(format!("mdl_{digest}")).is_ok());
        assert!(ContentId::parse(format!("content_{digest}")).is_ok());
        assert!(ModelId::parse(format!("content_{digest}")).is_err());
        assert!(ModelId::parse(format!("mdl_{}", "A".repeat(64))).is_err());
        assert!(ModelId::parse("mdl_short").is_err());
    }

    #[test]
    fn generation_performance_contract_round_trips_tagged_evidence() {
        let assessment = GenerationPerformanceAssessment::Estimated {
            method: "native".to_owned(),
            confidence: GenerationPerformanceConfidence::Low,
            workload: "baseline_single_sequence_decode".to_owned(),
            always_active_weight_bytes: 10,
            routed_expert_weight_bytes: 80,
            expert_count: 8,
            expert_used_count: 2,
            cross_memory_domain_placement: true,
            points: vec![GenerationSpeedPoint {
                context_tokens: 100_000,
                kv_bytes_read_per_token: 4_096,
                lower_tokens_per_second: 10.0,
                expected_tokens_per_second: 12.0,
                upper_tokens_per_second: 14.0,
            }],
        };
        let encoded = serde_json::to_value(&assessment).expect("serialize performance evidence");
        assert_eq!(encoded["status"], "estimated");
        assert_eq!(encoded["confidence"], "low");
        assert_eq!(
            serde_json::from_value::<GenerationPerformanceAssessment>(encoded)
                .expect("deserialize performance evidence"),
            assessment
        );
    }

    #[test]
    fn default_profile_assessor_marks_performance_as_not_requested() {
        assert_eq!(
            GenerationPerformanceAssessment::not_requested(),
            GenerationPerformanceAssessment::Unavailable {
                method: "not_requested".to_owned(),
                code: "not_requested".to_owned(),
                message: "generation performance was not requested".to_owned(),
            }
        );
    }

    #[test]
    fn memory_topology_requires_exact_stable_domain_capacity() {
        let topology = test_memory_topology();
        let assessment = fitting_assessment();
        assert!(topology.validates_hardware_assessment(&assessment));

        let mut corrupted = assessment;
        let HardwareAssessment::Fits { memory, .. } = &mut corrupted else {
            unreachable!();
        };
        memory.domains[0].usable_capacity_bytes = 59;
        memory.domains[0].margin_bytes = 39;
        memory.usable_capacity_bytes = 59;
        memory.headroom_bytes = 39;
        assert!(!topology.validates_hardware_assessment(&corrupted));
    }

    #[test]
    fn memory_topology_requires_exact_canonical_device_limit() {
        let topology = test_memory_topology();
        let assessment = fitting_assessment();

        let mut wrong_limit = assessment.clone();
        let HardwareAssessment::Fits { memory, .. } = &mut wrong_limit else {
            unreachable!();
        };
        memory.device_constraints[0].usable_capacity_bytes = 49;
        memory.device_constraints[0].margin_bytes = 29;
        assert!(!topology.validates_hardware_assessment(&wrong_limit));

        let mut unknown_device = assessment.clone();
        let HardwareAssessment::Fits { memory, .. } = &mut unknown_device else {
            unreachable!();
        };
        memory.device_constraints[0].device_id = HardwareDeviceId::new("unknown");
        assert!(!topology.validates_hardware_assessment(&unknown_device));

        let mut duplicate_device = assessment;
        let HardwareAssessment::Fits { memory, .. } = &mut duplicate_device else {
            unreachable!();
        };
        memory
            .device_constraints
            .push(memory.device_constraints[0].clone());
        assert!(!topology.validates_hardware_assessment(&duplicate_device));
    }
}
