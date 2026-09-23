//! Family-neutral model execution contracts and implementations.

mod completion;
mod device_resources;
mod domain;
mod error;
mod execution_path;
mod lanes;
mod native;
mod operation;
mod planning;
pub mod programs;
pub use programs::{
    CompletedHeadWork, CompletedImportWork, CompletedProjectWork, CompletedStateWork,
    CompletedTargetWork, CompletedVisionWork, HeadProgram, ImportProgram, PreparedHeadGraphs,
    PreparedStateCopyGraphs, PreparedTargetGraphs, PreparedTargetReadoutGraphs,
    PreparedVisionGraphs, StateProgram, TargetProgram, VisionProgram,
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
    ReservedResources, StateFlight, TargetFlight, VisionFlight,
};
pub use error::{CapacityError, DeviceError, InvariantError, PlanError, ResourceKind, SubmitError};
pub use execution_path::ExecutionPath;
pub use lanes::ConditioningSlice;
pub use lanes::{
    HeadLaunchCore, HeadLaunchInputs, ImportLaunchCore, ImportLaunchInputs, ProjectLaunchCore,
    ProjectionLaunchInputs, ProjectionRequest, ResidentWeightSlot, StateLaunchCore,
    StateLaunchInputs, StateWork, TargetLaunchCore, TargetLaunchInputs, ValidatedHeadLaunch,
    ValidatedImportLaunch, ValidatedProjectionLaunch, ValidatedStateLaunch, ValidatedTargetLaunch,
    ValidatedVisionLaunch, VisionLaunchCore, VisionLaunchInputs,
};
pub use magnitude_model_batching::{self as batching, Demand, LaunchClass};
pub use native::{
    AttestedPrograms, CatalogError, QualificationCase, QualificationReport, StageCallError,
};
pub use operation::{
    CommittedClass, ExecutableKind, FeatureRetainer, FeatureSpan, GroupKey, Operation,
    OperationError, Outcome, ProgramIdentity, RequestId, ResourceDomainId, RetainedFeatureSpan,
    RowResult, Sampling, SelectSpec, Selected, Shaping, TokenId, WorkKind,
};
pub use planning::{
    resident_element, source_element, ArtifactComponent, ArtifactComponentKind, AttentionBinding,
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
