export { protocolLayer } from "./protocol"
export {
  MagnitudeBoundary,
  magnitudeImplementationsLayer,
  type MagnitudeImplementationError,
} from "./inference"
export {
  MAGNITUDE_SERVICE_ORIGIN,
  MAGNITUDE_INFERENCE_BASE_URL,
  MAGNITUDE_ANTHROPIC_BASE_URL,
} from "./inference-endpoint"
export {
  makeInferenceClient,
  InferenceModelSchema,
  InferenceModelsResponseSchema,
  ResponseObjectSchema,
  type InferenceClient,
  type InferenceModel,
  type InferenceModelsResponse,
  type ResponseObject,
} from "./inference-client"

export {
  projectInferenceAllocation,
  projectInferenceLoadPlan,
  projectInferenceResidency,
} from "./inference-projection"
export type { ModelInstancesSnapshot as InferenceInstancesSnapshot } from "@magnitudedev/icn-protocol/schemas"

export {
  AcnInstanceManager,
  AcnEnsureRequestSchema,
  AcnEnsureEventSchema,
  AcnReadyInstanceSchema,
  RemoteAcnEnsureMessageSchema,
  runAcnEnsure,
  type AcnEnsureRequest,
  type AcnEnsureEvent,
  type AcnInstance,
  type RemoteAcnEnsureMessage,
} from "./acn-jit/acn-instance-manager"
export {
  makeLocalAcnInstanceManager,
  type LocalAcnInstanceManagerOptions,
  type AcnLaunchOverride,
} from "./acn-jit/local-acn-instance-manager"
export {
  makeLocalAcnRequireRunningInstanceManager,
  makeLocalAcnStartingInstanceManager,
  type LocalAcnObservationOptions,
} from "./acn-jit/local-acn-require-running-manager"
export {
  ChildProcessSpawner,
  scopeAcnCandidate,
  type SpawnedAcnCandidate,
} from "./acn-jit/child-process"
export { BunDetachedChildProcessSpawner } from "./acn-jit/bun-spawn-process"
export {
  makeRemoteAcnInstanceManager,
  RemoteAcnErrorResponseSchema,
} from "./acn-jit/remote-acn-instance-manager"
export {
  makeAcnConnection,
  AcnRecoveryInactive,
  AcnRecovering,
  AcnRecovered,
  type AcnStartup,
  type AcnRecovery,
  type AcnRecoveryState,
  type AcnConnection,
} from "./acn-jit/acn-recovering-client"
export {
  AcnLifecycleStateSchema,
  AcnLifecycleObservationSchema,
  AcnStartingPhaseSchema,
  AcnFailureStageSchema,
  type AcnLifecycle,
  type AcnLifecycleState,
  type AcnStartingPhase,
  type AcnInstallationPhase,
  type AcnFailureStage,
  type AcnStartupProgress,
  formatAcnEnsuranceError,
} from "./acn-jit/lifecycle"

export { TracingLayer, makeTracingLayer, type MakeTracingLayerOptions } from "./tracing"

/**
 * The complete ACN boundary: its root group, domain groups, operation
 * declarations (query, mutation, subscription), schemas, and errors.
 */
export * from "@magnitudedev/acn-protocol"
export {
  DisplayState as DisplayStateSchema,
  DisplayTimeline as DisplayTimelineSchema,
  DisplayViewShape as DisplayViewShapeSchema,
  StreamEvent as StreamEventSchema,
} from "@magnitudedev/acn-protocol"

export { createRoles, isRoleId, ROLE_IDS, ROLE_TO_SLOT, DEFAULT_REASONING_EFFORT, SLOT_IDS, SLOT_DISPLAY_NAMES, SLOT_DESCRIPTIONS } from "@magnitudedev/roles"
export type { RoleId } from "@magnitudedev/roles"

export { acnInstallationPresent, resolveBinaryCommand, defaultBinaryPath, defaultDataDir, type BinaryAcquisitionEvent, type ResolveBinaryOptions, type ResolvedBinaryCommand } from "./binary"
export { ACN_EXECUTABLE_NAME } from "@magnitudedev/release/executables"
export { SDK_VERSION, SDK_REVISION, SDK_ACN_TARGET } from "./version"
export {
  AcnAdministrationFailed,
  AcnOwnerRecordReadUnavailable,
  AcnOwnerRecordInvalid,
  AcnProcessIdentityObservationTimedOut,
  AcnCandidateSpawnFailed,
  AcnCandidateIdentityUnavailable,
  AcnCandidateExitedBeforeAdmission,
  AcnCandidateExitedAfterAdmission,
  AcnCandidateAdmissionTimedOut,
  AcnCandidateParentChannelReleaseFailed,
  AcnCandidateOwnershipLost,
  AcnCandidateFailureSchema,
  type AcnCandidateFailure,
  AcnCandidateBootstrapProcessStopFailed,
  AcnCandidateBootstrapProcessExitUnproven,
  AcnDaemonTargetUnsupported,
  AcnLaunchOverrideTargetMismatch,
  AcnDaemonShutdownFailed,
  AcnDaemonShutdownControlFailureSchema,
  type AcnDaemonShutdownControlFailure,
  AcnDaemonStartupTimedOut,
  AcnEnsuranceConvergenceTimedOut,
  AcnEnsuranceFailed,
  BinaryNotFound,
  BinaryRevisionMismatch,
  BinaryVersionMismatch,
  DownloadFailed,
  ChecksumMismatch,
  AcnEnsuranceError,
  type StreamDisplayViewFailure,
  type WatchFileFailure,
} from "./errors"

export { isEnvFlagOn } from "@magnitudedev/utils"
export { normalizeReferencedPath } from "./path-utils"
export { isRpcOutcomeUnknown } from "./mutation-outcome"

// =============================================================================
// Provider client surface — the sole provider boundary for agent & ACN
// =============================================================================

export {
  ProviderClient,
  createCustomEndpointProvider,
  customEndpointProviderId,
  createProviderClient,
  type ProviderClientShape,
  type ProviderClientConfig,
  type IcnModelPreparation,
  type ProviderRegistryInfo,
  type ProviderRuntimeConfig,
  type ProviderCatalogOutcome,
  type ProviderRejection,
  type ProviderClientError,
  type BaseCallOptions,
  type ModelDiscoveryOperationId,
  type ModelPropertyDiscoveryError,
  type ModelPropertyDiscoveryRequest,
  type ModelPropertyName,
  type ReasoningEffort,
  type ProviderModelBindOptions,
  type ProviderModel,
  type ProviderModelAvailability,
  type ProviderModelDisabledReason,
  type ProviderId,
  type ProviderModelId,
  type ModelFamilyId,
  AVAILABLE_PROVIDER_MODEL,
  isProviderModelAvailable,
  ModelCatalogError,
  ProviderIdSchema,
  ProviderModelIdSchema,
  ModelFamilyIdSchema,
  ProviderModelAvailabilitySchema,
  ProviderModelSchema,
  ModelDiscoveryOperationIdSchema,
  ModelPropertyDiscoveryErrorSchema,
  ModelPropertyDiscoveryRequestSchema,
  ModelPropertyNameSchema,
  ReasoningEffortSchema,
  ReasoningProperty,
  VisionProperty,
  type MagnitudeModelInfo,
  type MagnitudeCallOptions,
  type MagnitudeAdditionalOptions,
  type WebSearchResult,
  type WebSearchError,
  type WebSearchSource,
  WebSearchSourceSchema,
  formatWebSearchError,
  type UsageQuery,
  type CloudUsageResponse,
  type FetchUsageOptions,
  type UsagePeriod,
  makeFileBackedModelCatalog,
  createMagnitudeCompatibleSpec,
  MagnitudeModelListResponseSchema,
  toMagnitudeModelInfo,
  classifyMagnitudeRejectedResponse,
  classifyModelFamilyFromEvidence,
  tryParseErrorBody,
  type ParsedMagnitudeApiError,
} from "./provider-client"

export {
  type ToolCallId,
  type ChatCompletionsStreamChunk,
  type StreamFailure,
  type StreamStartFailure,
  type Prompt,
  type ToolDefinition,
  StreamOperationalFailure,
  StreamProviderError,
  StreamProviderCorrectnessViolation,
  StreamClientCorrectnessViolation,
  StreamStartClientCorrectnessViolation,
  StreamStartProviderCorrectnessViolation,
  StreamStartProviderRejection,
  StreamStartOperationalFailure,
  acceptedHttpResponse,
  payloadSample,
  rejectedHttpResponse,
  streamStartFailureFromRejectedResponse,
  toCauseInfo,
  nativeChatCompletionsCodec,
} from "@magnitudedev/ai"
