//! Family-neutral model execution contracts and implementations.

pub mod assessment;
mod completion;
mod device_resources;
mod domain;
mod error;
mod execution_path;
mod host_tables;
mod import_transforms;
mod kernel_cache;
mod lanes;
pub mod memory;
mod native;
mod operation;
mod operators;
mod planning;
pub mod programs;
pub use programs::{
    CommitSpan, CompletedHeadWork, CompletedImportWork, CompletedStateWork, CompletedTargetWork,
    CompletedVisionWork, HeadProgram, ImportProgram, PreparedDraftGraphs, PreparedDrafterGraphs,
    PreparedHeadGraphs, PreparedStateCopyGraphs, PreparedTargetGraphs, PreparedTargetReadoutGraphs,
    PreparedVisionGraphs, SealReport, StateProgram, TargetOutput, TargetProgram, VisionProgram,
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
    ClaimRefusal, DeviceHeap, DomainError, DomainRequirements, DomainReservation, ExecutorDomain,
    HeadFlight, MemoryChargeReconciliation, NativeFamily, OpenRequirements,
    PendingOperationOutcome, PhysicalDecision, ProgramFamily, ReservedResources, ResumeState,
    TargetFlight, TargetHostTiming, VisionFlight,
};
pub use error::{CapacityError, DeviceError, InvariantError, PlanError, ResourceKind, SubmitError};
pub use execution_path::ExecutionPath;
pub use kernel_cache::{KernelCache, KernelCacheError, TuningCacheKey, DEFAULT_KERNEL_CACHE_BYTES};
pub use lanes::ConditioningSlice;
pub use lanes::{
    HeadConditioning, HeadLaunchCore, HeadLaunchInputs, ImportLaunchCore, ImportLaunchInputs, ResidentWeightSlot,
    StateLaunchCore, StateLaunchInputs, StateWork, TargetLaunchCore, TargetLaunchInputs,
    TargetLaunchWorkspace, TargetTokens, ValidatedHeadLaunch, ValidatedImportLaunch,
    ValidatedStateLaunch, ValidatedTargetLaunch, ValidatedVisionLaunch, VisionLaunchCore,
    VisionLaunchInputs,
};
pub use magnitude_batching::{self as batching, Demand, LaunchClass};
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
    CommittedClass, DraftForm, ExecutableKind, FeatureReader, FeatureRows, FeatureSpan, GroupKey,
    HeadPhase, Operation, OperationError, Outcome, Priming, ProgramIdentity, RequestId,
    ResourceDomainId,
    RowResult, Sampling, SelectSpec, Selected, Shaping, TokenId, WorkKind,
};
pub use operators::routed::{
    Expansion, GeneralRoutedBinding, GeneralRoutedScales, GeneralRoutedShape,
};
pub use operators::short_conv::{ShortConvBinding, ShortConvShape};
pub use operators::state_space::{StateSpaceBinding, StateSpaceShape};
pub use operators::vision::{VisionEntry, VisionKernel};
pub use planning::{
    resident_element, resident_layout, source_element, ArtifactComponent, ArtifactComponentKind,
    AssessmentFit, AssessmentFitVerdict, AssessmentGraphResourceBounds, AssessmentHeaderBounds,
    AssessmentMemoryBounds, AssessmentMemoryCharge, AssessmentMemoryTerms, AttentionBinding,
    AttentionShape, CapabilityPlan, ComponentPlan, ComponentSelection, DenseBinding,
    DenseBranchBinding, DenseScales, Dflash2Binding, DraftBlockBinding, DraftProgramPlan,
    EmbeddingBinding, ExecutionPlan, ExecutionPlanDraft, ExecutionPlanner, FeaturesBinding,
    FeedForwardProgramSlot, GraphSlots, HeadBinding, HeadProgramPlan, HostTablePlan,
    ImportProgramSlot, MarkovBinding, MixerProgramSlot, ModelLoadPlan, NativeGraphCharge,
    ParallelBinding, PerLayerBinding, PerLayerEntryBinding, PlannedDevice, PlannedMethod,
    ProgramPlan, ReadoutBinding, RecurrentBinding, ResolvedPolicy, ResourceBytes, ResourceCapacity,
    ResourceLimits, ResourcePlan, ResourcePlanner, RoutedBinding, ScalableWeight, SelectorBinding,
    StartupSlots, StateCapacityPlan, StateProgramPlan, StateResourcePlan, StateStorePlan,
    StreamingCost, SublayerTail, TapProgramPlan, TargetBlockProgramSlot, TargetProgramPlan,
    VisionProgramPlan, WeightPlan, WeightScalePlan, WeightStorageIdentity, MAX_DRAFT_PROPOSALS,
};

/// Whether this executor runs `definition` with its draft head selected.
pub fn head_admitted(definition: &magnitude_family_contracts::ModelDefinition) -> bool {
    definition.head.is_some() && operators::admit(definition, true).is_ok()
}
pub use residency::{
    ComponentLoader, ImportArtifactTensor, ResidentWeight, Stored, StoredTensor, WeightImportError,
};
pub use residency::{MappedImportReport, ResidencyStore};
pub use resident_weights::{
    ResidencyError, ResidentHead, ResidentRoles, ResidentTarget, ResidentVision,
};
pub use resources::{
    AllocatedResources, AllocationError, GraphOutputOwner, GraphOutputTensor, ImportWorkspaceLease,
    NativeGraphOutputLease, NativeGraphPool, NativeGraphWorkspaceLease, PoolClass,
    ResourceAllocator, TargetGraphOutputLease, TargetGraphPool, TargetGraphWorkspaceLease,
};
