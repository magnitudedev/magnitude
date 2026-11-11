//! Family-neutral model execution contracts and implementations.

/// The executor build identity: the package version and a digest of the
/// executor's dependency closure (`build.rs`). What the executor alone decides
/// is stored under it.
pub const EXECUTOR_BUILD: &str = env!("MAGNITUDE_EXECUTOR_BUILD");

pub mod assessment;
mod completion;
mod device_resources;
mod device_table;
pub mod diagnostics;
pub mod watched;
pub mod workload;
mod domain;
#[cfg(any(test, feature = "fixtures"))]
pub use domain::fixtures;
mod error;
mod execution_path;
mod experts;
pub use experts::{
    ExpertCosts, GroupCost as PromptGroupCost, GroupLayer as PromptGroupLayer, LayerTimeline as PromptLayerTimeline, Span as PromptSpan, StepTimeline as PromptStepTimeline, LayerCost as ExpertLayerCost, LayerExperts,
    SITE_TERM_NAMES, SiteServed, StepCost as ExpertStepCost, STEP_TRACE as EXPERT_STEP_TRACE, LayerCurve, MissCurve, RoutedRead,
};
#[cfg(feature = "expert-routing")]
pub use experts::{Recency as EvictionRecency, replay as replay_routing};
mod host_tables;
mod import_transforms;
mod inventory;
mod kernel_cache;
mod lanes;
pub mod memory;
mod native;
#[cfg(test)]
mod ngram_steps_tests;
#[cfg(any(test, feature = "fixtures"))]
mod test_support;
mod operation;
mod operators;
mod planning;
pub mod programs;
mod progressive;
mod slot_directory;
mod storage_thread;
mod stored_table;
mod streamed_table;
pub use programs::{
    Chain, CommitSpan, CompletedHeadWork, CompletedImportWork, CompletedStateWork, CompletedTargetWork,
    CompletedVisionWork, GraphError, HeadProgram, ImportProgram, PlannedSealedGraphs,
    PreparedDraftGraphs, PreparedDrafterGraphs, PreparedHeadGraphs, PreparedStateCopyGraphs,
    PreparedTargetGraphs, PreparedTargetReadoutGraphs, PreparedVisionGraphs, SealReport,
    SealedGraphsHeap, StateProgram, TargetOutput, TargetProgram, VisionProgram,
};
pub mod platform;
mod residency;
pub mod settle;
mod resident_weights;
mod resources;

pub use completion::{Completed, Completion, CompletionWake};
pub use experts::derived::{import_slots, in_process_growth};
pub use stored_table::{Prefetched, TableWaits};
pub use device_resources::{
    ConditioningRange, ConditioningRef, FeatureRef, ImageRef, LogitsRef, ResourceDomain,
    ResourceError, WeakFeatureRef,
};
pub use domain::{
    MEMORY_TRACE, BankHolding, BankResize, BankResized, BankStep, Followed, TableResidency, DeviceReader, LoadRoom, MachineReading, DomainError, DomainRequirements, DomainReservation, ExecutorDomain,
    ExpertMeasures, LoadCosts, LoadMeasures, ExpertResidency, ExpertResizes, HeldBy, HostResidency, ThreadHeld, OptionalComponent, TierFill, DecodeStep,
    HeadFlight, LaunchDuration, MemoryChargeReconciliation, MemoryMoment, NativeFamily, OpenRequirements,
    PendingOperationOutcome, PhysicalDecision, PrimingCompletionFlight, ProgramFamily,
    ReservedResources, ResumeState, RunningBegun, RunningPressure, StateBindings, SlotRelease, StateShrink, StoreAllocation, SubmitFailure,
    TargetContinuation, TargetFlight, TargetHostTiming, TrafficHold, VisionFlight,
};
pub use error::{
    CapacityError, DeviceError, InvariantError, PlanError, ResourceKind, SubmitError,
    UnstreamedExperts, WeightReadError,
};
pub use execution_path::ExecutionPath;
pub use inventory::kernel_inventory;
pub use kernel_cache::{
    DEFAULT_KERNEL_CACHE_BYTES, KernelCache, KernelCacheError, StoredEvidence, TuningCacheKey,
};
pub use lanes::ConditioningSlice;
pub use lanes::{
    HeadConditioning, HeadLaunchCore, HeadLaunchInputs, ImportLaunchCore, ImportLaunchInputs,
    ResidentWeightSlot, StateLaunchCore, StateLaunchInputs, StateWork, TargetHistoryRead,
    TargetLaunchCore, TargetLaunchInputs, TargetLaunchWorkspace, TargetReads, TargetTokens,
    ValidatedHeadLaunch, ValidatedImportLaunch, ValidatedStateLaunch, ValidatedTargetLaunch,
    ValidatedVisionLaunch, VisionLaunchCore, VisionLaunchInputs,
};
pub use magnitude_batching::{self as batching, Demand, LaunchClass};
/// Development-only tuning pin for bit-exact measurement runs
/// (`magnitude-harness --tuning-record`); see the module documentation.
pub use native::pinned_tuning;
/// Development-only tuning survey for replaying the search
/// (`magnitude-harness --tuning-survey`); see the module documentation.
#[cfg(feature = "tuning-survey")]
pub use native::tuning_survey;
pub use native::{
    AttestedPrograms, CatalogError, CatalogFailure, MIN_SAMPLE_SECONDS, NumericalProfile,
    PinnedChoice, PointShape, QualificationCase, QualificationReport, ROTATION_LAYERS, Receipt,
    ResolvedNumericalProfile, SEARCH_SETTINGS, SEARCH_VERSION, STANDARD_TUNING_BUDGET,
    SearchProgress, TRACE_SCHEMA, TUNING_CONTEXTS, TUNING_ROWS, TracedWork, TunedEntry,
    TuningContext, TuningEvent, TuningLimits, TuningObserver, TuningOrigin, TuningRoom, RoomNow,
    TuningSettings, TuningTrace, TuningWeightSource, UnpreparedUnits, UnreportedTuning,
    ZeroTuningWeights, attention_points, row_points,
};
pub use operation::{
    CommittedClass, DraftForm, DraftShape, ExecutableKind, FeatureReader, FeatureRows, FeatureSpan, GroupKey,
    HeadPhase, Operation, OperationError, Outcome, Priming, ProgramIdentity, RequestId,
    ResourceDomainId, RowResult, Sampling, SelectSpec, Selected, Shaping, TokenId, WorkKind,
};
pub use operators::routed::{
    Expansion, ExpertScales, GeneralRoutedBinding, GeneralRoutedScales, GeneralRoutedShape,
};
pub use operators::short_conv::{ShortConvBinding, ShortConvShape};
pub use operators::state_space::{StateSpaceBinding, StateSpaceScales, StateSpaceShape};
pub use operators::vision::{VisionEntry, VisionKernel};
pub use planning::{
    growth_target,
    ArtifactComponent, ArtifactComponentKind, AssessmentFit, AssessmentFitVerdict,
    AssessmentGraphResourceBounds, AssessmentHeaderBounds, AssessmentMemoryBounds, TargetGraphBounds, UndraftedBounds, VisionGraphBounds,
    AssessmentMemoryCharge, AssessmentMemoryTerms, AttentionBinding, AttentionShape, AttentionValues,
    RoundingArgs,
    BackendPlan,
    BankAllocation, BankBinding, BankTier, ConveyorCarrier, ExpertBankPlan, Holding, StoredStack,
    CONVERSATION_CHECKPOINTS, CPU_IMAGE_CELLS, CapabilityPlan, ComponentPlan, ComponentSelection,
    ConvolutionBinding, ConvolvedLayerBinding, CorrectorBinding, CorrectorLayerBinding,
    DenseBinding, DenseBranchBinding, DenseScales, DeviceProcessBytes, DraftBlockBinding,
    DraftProgramPlan, DraftWidths,
    BankForm, ChargedDraft, EmbeddingBinding, ExecutionPlan, ExecutionPlanDraft, ExecutionPlanner, FeaturesBinding, FloorForms, HeldForm, HeldBound, HeldCharge, Placed, Unplaced, WeightBound,
    FeedForwardProgramSlot, FlightDepth, GraphSlots, HeadBinding, HeadProgramPlan, HostProcessBytes,
    CandidatePool, CandidatesBinding, HostTablePlan, ImportProgramSlot, IndexerBinding,
    LowRankQueryBinding, PickedBinding,
    PooledBinding, MAX_DRAFT_PROPOSALS, MAX_IMAGE_CELLS, MarkovBinding,
    MixerProgramSlot, ModelLoadPlan, NGramBinding, NGramGateForm, NGramTaps, NoBank, PlanRoom, SettleRoom, StableRoom, WholeFit, NativeGraphCharge, ParallelBinding,
    PassLaunches, PlanPrices, PlanSteps, PerLayerBinding, PerLayerEntryBinding, PlannedDevice, PlannedMethod, ProgramPlan,
    ReadoutBinding, ReadoutClassChoice, ReadoutHead, RecurrentBinding, ResolvedPolicy,
    ResourceBytes, ResourceCapacity, ResourceLimits, ResourcePlan, ResourcePlanner, RoutedBinding,
    GroupedOutputBinding, GroupedOutputShape, ScalableWeight, SealedGraphHostBytes,
    SelectionBinding, SelectionShape, SelectorBinding,
    SharedHeadProgramPlan, StartupSlots, StateCapacityPlan, StateProgramPlan, StateResourcePlan,
    StateStorePlan, SublayerTail, TapProgramPlan, TargetBlockProgramSlot, TargetProgramPlan,
    TensorOperations, VisionProgramPlan, WeightPlan, WeightScalePlan, WeightStorageIdentity,
    draft_ladder, held_storage, image_cell_limit, max_image_cells, resident_element, resident_elements, resident_layout, source_element,
    scaled_source_element, stored_weights,
};

/// Whether this executor runs `definition` with its draft head selected.
pub fn head_admitted(definition: &magnitude_family_contracts::ModelDefinition) -> bool {
    definition.head.is_some() && operators::admit(definition, true).is_ok()
}
pub use residency::{
    ComponentLoader, ImportArtifactTensor, ResidentWeight, Stored, StoredTensor, WeightImportError,
};
pub use residency::{ImportReport, ResidencyStore};
pub use storage_thread::Releases;
pub use resident_weights::{
    ResidencyError, ResidentHead, ResidentOutput, ResidentPlanes, ResidentRoles, ResidentTarget,
    ResidentVision,
};
pub use resources::{
    AllocatedResources, AllocationError, GraphOutputOwner, GraphOutputTensor, ImportWorkspaceLease,
    NativeGraphOutputLease, NativeGraphPool, NativeGraphWorkspaceLease, PoolClass,
    ResourceAllocator, SlotsReleased, TargetGraphOutputLease, TargetGraphPool, TargetGraphWorkspaceLease,
    VisionGraphPool,
};
