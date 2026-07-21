//! OpenAPI-only mirrors of backend-neutral inventory contracts.
//!
//! The runtime values remain owned by `icn-contracts`. Keeping Utoipa out of
//! that crate preserves its transport-neutral boundary while still emitting a
//! complete generated client contract.

#![allow(dead_code)]

use serde::Serialize;
use serde_json::Value as JsonValue;
use utoipa::ToSchema;

#[derive(Debug, Serialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ComponentRoleSchema {
    Weights,
    Shard,
    Projector,
    Auxiliary,
    Draft,
    Mtp,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum HardwareMemoryDomainKindSchema {
    System,
    PhysicalDevice,
    UnifiedMemory,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum HardwareDeviceKindSchema {
    Cpu,
    Gpu,
    IntegratedGpu,
    Accelerator,
    Unknown,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum HardwareDeviceMemoryLimitKindSchema {
    RecommendedWorkingSet,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct HardwareDeviceMemoryLimitSchema {
    kind: HardwareDeviceMemoryLimitKindSchema,
    total_bytes: u64,
    stable_bytes: u64,
    current_free_bytes: Option<u64>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct HardwareDeviceSchema {
    id: String,
    backend: String,
    name: String,
    description: String,
    kind: HardwareDeviceKindSchema,
    memory_limit: Option<HardwareDeviceMemoryLimitSchema>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct HardwareMemoryDomainSchema {
    id: String,
    kind: HardwareMemoryDomainKindSchema,
    total_capacity_bytes: u64,
    stable_capacity_bytes: u64,
    current_free_bytes: Option<u64>,
    shares_system_memory: bool,
    devices: Vec<HardwareDeviceSchema>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct HardwareSystemMemorySchema {
    total_bytes: u64,
    current_available_bytes: Option<u64>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct HardwareSnapshotSchema {
    captured_at: u64,
    platform: String,
    architecture: String,
    cpu_model: Option<String>,
    logical_cores: usize,
    system_memory: HardwareSystemMemorySchema,
    native_build: String,
    enabled_backends: Vec<String>,
    assessment_policy: String,
    capacity_policy: String,
    topology_fingerprint: String,
    memory_domains: Vec<HardwareMemoryDomainSchema>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ModelPreviewSourceSchema {
    repository: String,
    revision: String,
    primary_gguf: String,
    additional_components: Vec<ModelPreviewComponentSourceSchema>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ModelPreviewComponentSourceSchema {
    path: String,
    role: ComponentRoleSchema,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ModelPreviewProfileSchema {
    id: String,
    policy: String,
    context_length: u32,
    parallel_sequences: u32,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ModelPreviewRequestSchema {
    source: ModelPreviewSourceSchema,
    profiles: Vec<ModelPreviewProfileSchema>,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ModelPreviewAssessmentSchema {
    profile_id: String,
    artifact_fingerprint: String,
    execution_policy: String,
    hardware_topology: String,
    assessment: HardwareAssessmentSchema,
    performance: GenerationPerformanceAssessmentSchema,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum GenerationPerformanceConfidenceSchema {
    High,
    Moderate,
    Low,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct GenerationSpeedPointSchema {
    context_tokens: u32,
    kv_bytes_read_per_token: u64,
    lower_tokens_per_second: f64,
    expected_tokens_per_second: f64,
    upper_tokens_per_second: f64,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(tag = "status", rename_all = "snake_case", deny_unknown_fields)]
pub enum GenerationPerformanceAssessmentSchema {
    Estimated {
        method: String,
        confidence: GenerationPerformanceConfidenceSchema,
        workload: String,
        always_active_weight_bytes: u64,
        routed_expert_weight_bytes: u64,
        expert_count: u32,
        expert_used_count: u32,
        cross_memory_domain_placement: bool,
        points: Vec<GenerationSpeedPointSchema>,
    },
    Unavailable {
        method: String,
        code: String,
        message: String,
    },
}

#[derive(Debug, Serialize, ToSchema)]
pub struct ModelPreviewSchema {
    repository: String,
    commit: String,
    components: Vec<ModelComponentSchema>,
    properties: InventoryPropertiesSchema,
    assessments: Vec<ModelPreviewAssessmentSchema>,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ContentIdentitySchema {
    Sha256 { value: String },
    GitOid { value: String },
    Xet { value: String },
    FileIdentity { value: String },
    Unknown,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
// The suffix is part of the public relationship vocabulary and makes direction explicit on wire.
#[allow(clippy::enum_variant_names)]
pub enum ComponentRelationshipSchema {
    ProjectorFor { projector: String, model: String },
    DraftFor { draft: String, model: String },
    MtpFor { mtp: String, model: String },
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum IntegritySchema {
    Verified { method: String },
    Unverified { reason: String },
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct ModelComponentSchema {
    path: String,
    role: ComponentRoleSchema,
    size_bytes: u64,
    content: ContentIdentitySchema,
    shard_index: Option<u32>,
    #[schema(nullable = false)]
    relationship: Option<ComponentRelationshipSchema>,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum LocalDeclarationSchema {
    Configuration,
    Discovery,
    ActiveProcess,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct HubMetadataSchema {
    access: Option<String>,
    author: Option<String>,
    license: Option<String>,
    pipeline_tag: Option<String>,
    library_name: Option<String>,
    tags: Vec<String>,
    downloads: Option<u64>,
    likes: Option<u64>,
    last_modified: Option<String>,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ModelSourceSchema {
    HuggingFace {
        repository: String,
        requested_revision: String,
        commit: String,
        metadata: Option<Box<HubMetadataSchema>>,
    },
    Local {
        declared_by: LocalDeclarationSchema,
    },
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ModelLocationSchema {
    MagnitudeCache {
        components: Vec<ModelComponentSchema>,
        total_bytes: u64,
        integrity: IntegritySchema,
    },
    HuggingFaceCache {
        cache_root: String,
        repository: String,
        commit: String,
        components: Vec<ModelComponentSchema>,
        total_bytes: u64,
        integrity: IntegritySchema,
    },
    Directory {
        source_id: String,
        root: String,
        components: Vec<ModelComponentSchema>,
        total_bytes: u64,
        integrity: IntegritySchema,
    },
    File {
        path: String,
        component: ModelComponentSchema,
        integrity: IntegritySchema,
    },
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum DownloadStageSchema {
    Queued,
    Resolving,
    CheckingSpace,
    Downloading,
    Verifying,
    Publishing,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum LoadStageSchema {
    Opening,
    Mapping,
    Allocating,
    InitializingContext,
    Warming,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ModelStatusSchema {
    Downloading {
        operation_id: String,
        stage: DownloadStageSchema,
        completed_bytes: u64,
        total_bytes: u64,
        current_component: Option<String>,
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
    Loading {
        load_id: String,
        stage: LoadStageSchema,
        started_at: u64,
    },
    Loaded {
        loaded_at: u64,
        backend: String,
        context_length: u32,
        execution: JsonValue,
    },
    Unloading {
        load_id: String,
        started_at: u64,
    },
    LoadFailed {
        attempted_at: u64,
        stage: LoadStageSchema,
        code: String,
        retryable: bool,
    },
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum CapabilitySupportSchema {
    Supported { parallel: Option<bool> },
    Unsupported,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum CapabilityEvidenceSchema {
    NativeTemplate { fingerprint: String },
    BoundedTemplateProbe { fingerprint: String },
    DeclaredMetadata { source: String },
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningVisibilitySchema {
    Hidden,
    Preserved,
    Configurable,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ReasoningDelimitersSchema {
    Unavailable,
    Known { start: String, end: String },
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ReasoningControlDomainSchema {
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

#[derive(Debug, Serialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ReasoningCapabilitySchema {
    Unsupported {
        evidence: CapabilityEvidenceSchema,
    },
    Supported {
        control: ReasoningControlDomainSchema,
        visibility: ReasoningVisibilitySchema,
        delimiters: ReasoningDelimitersSchema,
        evidence: CapabilityEvidenceSchema,
    },
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
// Preserve the flat public response shape instead of nesting the inspected payload for an
// in-memory enum-size optimization.
#[allow(clippy::large_enum_variant)]
pub enum InventoryPropertiesSchema {
    Pending,
    Unavailable {
        reason: String,
    },
    Inspected {
        architecture: Option<String>,
        quantization: Option<String>,
        parameter_count: Option<u64>,
        active_parameter_count: Option<u64>,
        training_context_length: Option<u32>,
        tokenizer: Option<String>,
        modalities: Vec<String>,
        base_models: Vec<String>,
        tools: CapabilitySupportSchema,
        structured_output: CapabilitySupportSchema,
        reasoning: ReasoningCapabilitySchema,
        evidence_fingerprint: String,
    },
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct HardwareProfileSchema {
    context_length: u32,
    acceleration: String,
    device: String,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct HardwareMemorySchema {
    required_bytes: u64,
    available_bytes: u64,
    headroom_bytes: u64,
    domains: Vec<HardwareMemoryDomainAssessmentSchema>,
    device_constraints: Vec<HardwareDeviceMemoryAssessmentSchema>,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct HardwareDeficitSchema {
    required_bytes: u64,
    available_bytes: u64,
    deficit_bytes: u64,
    domains: Vec<HardwareMemoryDomainAssessmentSchema>,
    device_constraints: Vec<HardwareDeviceMemoryAssessmentSchema>,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(deny_unknown_fields)]
pub struct HardwareMemoryDomainAssessmentSchema {
    memory_domain: String,
    model_bytes: u64,
    context_bytes: u64,
    compute_bytes: u64,
    auxiliary_bytes: u64,
    required_bytes: u64,
    available_bytes: u64,
    margin_bytes: i64,
}

#[derive(Debug, Serialize, ToSchema)]
pub struct HardwareDeviceMemoryAssessmentSchema {
    device: String,
    kind: HardwareDeviceMemoryLimitKindSchema,
    model_bytes: u64,
    context_bytes: u64,
    compute_bytes: u64,
    auxiliary_bytes: u64,
    required_bytes: u64,
    available_bytes: u64,
    margin_bytes: i64,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum HardwareRecommendationSchema {
    Recommended,
    Constrained,
}

#[derive(Debug, Serialize, ToSchema)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum HardwareAssessmentSchema {
    NotAssessed {
        reason: String,
    },
    Fits {
        profile: HardwareProfileSchema,
        memory: HardwareMemorySchema,
        recommendation: HardwareRecommendationSchema,
    },
    DoesNotFit {
        profile: HardwareProfileSchema,
        memory: HardwareDeficitSchema,
        limiting_resource: String,
        alternative: Option<HardwareProfileSchema>,
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
