//! Family-neutral model execution contracts and implementations.

pub mod assessment;
mod completion;
mod device_resources;
mod domain;
mod error;
mod execution_path;
mod kernel_cache;
mod lanes;
pub mod memory;
mod native;
mod operation;
mod planning;
pub mod programs;
pub use programs::{
    CommitSpan, CompletedHeadWork, CompletedImportWork, CompletedStateWork, CompletedTargetWork,
    CompletedVisionWork, HeadProgram, ImportProgram, PreparedHeadGraphs, PreparedStateCopyGraphs,
    PreparedTargetGraphs, PreparedTargetReadoutGraphs, PreparedVisionGraphs, SealReport,
    StateProgram, TargetOutput, TargetProgram, VisionProgram,
};
pub mod platform;
mod residency;
mod resident_weights;
mod resources;

pub use completion::{Completed, Completion, CompletionWake};
pub use device_resources::{
    ConditioningRange, ConditioningRef, FeatureRef, ImageRef, LogitsRef, ResourceDomain,
    ResourceError,
};
pub use domain::{
    DomainCheckpoint, DomainError, DomainRequirements, DomainReservation, ExecutorDomain,
    HeadFlight, MemoryChargeReconciliation, NativeFamily, OpenRequirements, OpenReservation,
    PendingOperationOutcome, PhysicalDecision, ProgramFamily, ReservedResources, TargetFlight,
    TargetHostTiming, VisionFlight,
};
pub use error::{CapacityError, DeviceError, InvariantError, PlanError, ResourceKind, SubmitError};
pub use execution_path::ExecutionPath;
pub use kernel_cache::{KernelCache, KernelCacheError, TuningCacheKey, DEFAULT_KERNEL_CACHE_BYTES};
pub use lanes::ConditioningSlice;
pub use lanes::{
    HeadLaunchCore, HeadLaunchInputs, ImportLaunchCore, ImportLaunchInputs, ResidentWeightSlot,
    StateLaunchCore, StateLaunchInputs, StateWork, TargetLaunchCore, TargetLaunchInputs,
    TargetLaunchWorkspace, TargetTokens, ValidatedHeadLaunch, ValidatedImportLaunch,
    ValidatedStateLaunch, ValidatedTargetLaunch, ValidatedVisionLaunch, VisionLaunchCore,
    VisionLaunchInputs,
};
pub use magnitude_model_batching::{self as batching, Demand, LaunchClass};
/// Development-only tuning pin for bit-exact measurement runs
/// (`forward_bench`); see the module documentation.
#[cfg(feature = "pinned-tuning")]
pub use native::pinned_tuning;
/// Development-only tuning survey for replaying the search
/// (`forward_bench --tuning-survey`); see the module documentation.
#[cfg(feature = "tuning-survey")]
pub use native::tuning_survey;
pub use native::{
    attention_points, row_points, AttestedPrograms, CatalogError, CatalogFailure, PointShape,
    QualificationCase, QualificationReport, TunedEntry, TuningContext, TuningEvent, TuningLimits,
    TuningObserver, TuningOrigin, TuningWeightSource, UnreportedTuning, ZeroTuningWeights,
    ROTATION_LAYERS, TUNING_CONTEXTS, TUNING_ROWS,
};
pub use operation::{
    CommittedClass, ExecutableKind, FeatureReader, FeatureRows, FeatureSpan, GroupKey, Operation,
    OperationError, Outcome, ProgramIdentity, RequestId, ResourceDomainId, RowResult, Sampling,
    SelectSpec, Selected, Shaping, TokenId, WorkKind,
};
pub use planning::{
    resident_element, resident_layout, source_element, ArtifactComponent, ArtifactComponentKind,
    AssessmentBindingEvidence, AssessmentFit, AssessmentFitVerdict, AssessmentGraphResourceBounds,
    AssessmentHeaderBounds, AssessmentMemoryBounds, AssessmentMemoryTerms, AttentionBinding,
    AttentionShape, CapabilityPlan, ComponentPlan, ComponentSelection, DenseBinding,
    EmbeddingBinding, ExecutionPlan, ExecutionPlanDraft, ExecutionPlanner, FeaturesBinding,
    FeedForwardProgramSlot, HeadBinding, HeadProgramPlan, ImportProgramSlot, MixerProgramSlot,
    ModelLoadPlan, NativeGraphCharge, PlannedDevice, PlannedMethod, ProgramPlan, ReadoutBinding,
    RecurrentBinding, ResolvedPolicy, ResourceBytes, ResourceCapacity, ResourceLimits,
    ResourcePlan, ResourcePlanner, RetentionCapacityPlan, RoutedBinding, StateCapacityPlan,
    StateProgramPlan, StateResourcePlan, StateStorePlan, StreamingCost, TargetBlockProgramSlot,
    TargetProgramPlan, VisionBlockBinding, VisionMergerBinding, VisionPatchBinding,
    VisionProgramPlan, WeightPlan, WeightStorageIdentity, MAX_DRAFT_PROPOSALS,
};
pub use residency::{
    ComponentLoader, ImportArtifactTensor, ResidentWeight, Stored, StoredTensor, WeightImportError,
};
pub use residency::{MappedImportReport, ResidencyStore};
pub use resident_weights::{
    ResidencyError, ResidentAttentionWeights, ResidentBlockWeights,
    ResidentDenseFeedForwardWeights, ResidentFeedForwardWeights, ResidentFusedQkvWeights,
    ResidentHead, ResidentHeadBlock, ResidentLayerNormWeights, ResidentMixerWeights,
    ResidentRecurrentWeights, ResidentRoutedFeedForwardWeights, ResidentTarget, ResidentVision,
    ResidentVisionAttentionWeights, ResidentVisionBlockWeights, ResidentVisionFeedForwardWeights,
    ResidentVisionMergerWeights,
};
pub use resources::{
    AllocatedResources, AllocationError, GraphOutputOwner, GraphOutputTensor, ImportWorkspaceLease,
    NativeGraphOutputLease, NativeGraphPool, NativeGraphWorkspaceLease, PoolClass,
    ResourceAllocator, TargetGraphOutputLease, TargetGraphPool, TargetGraphWorkspaceLease,
};
pub use seismic::PressureLevel;
