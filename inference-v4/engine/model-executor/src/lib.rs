//! Family-neutral model execution contracts and implementations.

mod completion;
mod device_resources;
mod domain;
mod error;
mod execution_path;
mod kernel_cache;
mod lanes;
mod native;
mod operation;
mod planning;
pub mod programs;
pub use programs::{
    CommitSpan, CompletedHeadWork, CompletedImportWork, CompletedProjectWork, CompletedStateWork,
    CompletedTargetWork, CompletedVisionWork, HeadProgram, ImportProgram, PreparedHeadGraphs, SealReport,
    PreparedStateCopyGraphs, PreparedTargetGraphs, PreparedTargetReadoutGraphs,
    PreparedVisionGraphs, StateProgram, TargetOutput, TargetProgram, VisionProgram,
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
    HeadFlight, NativeFamily, OpenRequirements, OpenReservation, PendingOperationOutcome,
    PhysicalDecision, PhysicalResolution, ProgramFamily, ProjectFlight, ReservedRepair,
    ReservedResources, StateFlight, TargetFlight, TargetHostTiming, VisionFlight,
};
pub use error::{CapacityError, DeviceError, InvariantError, PlanError, ResourceKind, SubmitError};
pub use execution_path::ExecutionPath;
pub use kernel_cache::{KernelCache, KernelCacheError, TuningCacheKey, DEFAULT_KERNEL_CACHE_BYTES};
pub use lanes::ConditioningSlice;
pub use lanes::{
    HeadLaunchCore, HeadLaunchInputs, ImportLaunchCore, ImportLaunchInputs, ProjectLaunchCore,
    ProjectionLaunchInputs, ProjectionRequest, ResidentWeightSlot, StateLaunchCore,
    StateLaunchInputs, StateWork, TargetLaunchCore, TargetLaunchInputs, TargetLaunchWorkspace,
    ValidatedHeadLaunch,
    ValidatedImportLaunch, ValidatedProjectionLaunch, ValidatedStateLaunch, ValidatedTargetLaunch,
    ValidatedVisionLaunch, VisionLaunchCore, VisionLaunchInputs,
};
pub use magnitude_model_batching::{self as batching, Demand, LaunchClass};
pub use native::{
    attention_points, row_points, AttestedPrograms, CatalogError, CatalogFailure,
    MissingImplementation, ZeroTuningWeights, PointShape, QualificationCase, QualificationReport,
    TunedEntry, TuningContext, TuningEvent, TuningLimits, TuningObserver, TuningOrigin,
    TuningWeightSource, UnreportedTuning,
    ROTATION_LAYERS, TUNING_CONTEXTS, TUNING_ROWS,
};
/// Development-only tuning pin for bit-exact measurement runs
/// (`forward_bench`); see the module documentation.
#[cfg(feature = "pinned-tuning")]
pub use native::pinned_tuning;
/// Development-only tuning survey for replaying the search
/// (`forward_bench --tuning-survey`); see the module documentation.
#[cfg(feature = "tuning-survey")]
pub use native::tuning_survey;
pub use operation::{
    CommittedClass, ExecutableKind, FeatureRetainer, FeatureSpan, GroupKey, Operation,
    OperationError, Outcome, ProgramIdentity, RequestId, ResourceDomainId, RetainedFeatureSpan,
    RowResult, Sampling, SelectSpec, Selected, Shaping, TokenId, WorkKind,
};
pub use planning::{
    resident_element, resident_layout, source_element, ArtifactComponent, ArtifactComponentKind,
    AttentionBinding, AttentionShape,
    CapabilityPlan, ComponentPlan, ComponentSelection, DenseBinding, EmbeddingBinding,
    ExecutionPlan, ExecutionPlanDraft, ExecutionPlanner, FeaturesBinding, FeedForwardProgramSlot,
    HeadBinding, HeadProgramPlan, ImportProgramSlot, MixerProgramSlot, ModelLoadPlan,
    PlannedDevice, PlannedMethod, ProgramPlan, ReadoutBinding,
    RecurrentBinding, ResolvedPolicy, ResourceBudget, ResourceBytes, ResourceLimits, ResourcePlan,
    ResourcePlanner, RetentionCapacityPlan, RoutedBinding, StateCapacityPlan,
    StateProgramPlan, StateResourcePlan, StateStorePlan, TargetBlockProgramSlot, NativeGraphCharge,
    TargetProgramPlan, VisionBlockBinding, VisionMergerBinding,
    VisionPatchBinding, VisionProgramPlan, WeightPlan, WeightStorageIdentity,
};
pub use residency::ResidencyStore;
pub use residency::{
    ComponentLoader, ImportArtifactTensor, ResidentWeight, Stored, StoredTensor, WeightImportError,
};
pub use resident_weights::{
    ResidencyError, ResidentAttentionWeights, ResidentBlockWeights,
    ResidentDenseFeedForwardWeights, ResidentFeedForwardWeights, ResidentFusedQkvWeights,
    ResidentHead, ResidentHeadBlock, ResidentLayerNormWeights, ResidentMixerWeights,
    ResidentRecurrentWeights, ResidentRoutedFeedForwardWeights, ResidentTarget, ResidentVision,
    ResidentVisionAttentionWeights, ResidentVisionBlockWeights, ResidentVisionFeedForwardWeights,
    ResidentVisionMergerWeights,
};
pub use resources::{
    AllocatedResources, AllocationError, GraphOutputOwner, GraphOutputTensor,
    ImportWorkspaceLease, NativeGraphOutputLease, NativeGraphPool,
    NativeGraphWorkspaceLease, PoolClass, ResourceAllocator, TargetGraphOutputLease,
    TargetGraphPool, TargetGraphWorkspaceLease,
};
