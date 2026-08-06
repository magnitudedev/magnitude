/**
 * Browser-safe SDK entry.
 *
 * Renderer builds resolve `@magnitudedev/sdk` here. ACN ensurance is
 * delegated to the host over HTTP.
 */
export {
  AcnEnsurer,
  AcnEnsureEventSchema,
  AcnEnsureRequestSchema,
  ReadyAcnSchema,
  RemoteAcnEnsureMessageSchema,
} from "./acn-jit/acn-ensurer"
export type { AcnEnsureEvent, AcnEnsureRequest, ReadyAcn } from "./acn-jit/acn-ensurer"
export { makeRemoteAcnEnsurer } from "./acn-jit/remote-acn-ensurer"
export {
  makeAcnJitRuntime,
} from "./acn-jit/acn-recovering-client"
export type {
  AcnClientCloseReport,
  AcnClientCloseResult,
  AcnJitRuntime,
} from "./acn-jit/acn-recovering-client"
export {
  AcnEnsuranceFailed,
  BinaryNotFound,
  BinaryVersionMismatch,
  DownloadFailed,
  ChecksumMismatch,
  AcnEnsuranceError,
} from "./errors"

export {
  DisplayState as DisplayStateSchema,
  DisplayViewSnapshot,
  DisplayViewShape as DisplayViewShapeSchema,
  MagnitudeRpcs,
  StreamEvent as StreamEventSchema,
  canonicalExtensionForImageMediaType,
  filenameWithImageExtension,
  forkIdToKey,
  imageMediaTypeFromFilename,
  imageMediaTypeFromMime,
  isSupportedImageFilename,
  ProviderModelCatalogLifecycle,
  ProviderModelCatalogLoading,
  ProviderModelCatalogReady,
  ProviderModelCatalogRefreshing,
  ProviderModelCatalogDegraded,
  ProviderModelCatalogUnavailable,
  ModelSlotUnassigned,
  ModelSlotConfiguredRemote,
  ModelSlotConfiguredLocal,
  SlotIdSchema,
  LocalInferenceAcceleratorIdSchema,
  LocalInferenceMemoryDomainIdSchema,
  PercentageSchema,
  ModelCapabilitiesSchema,
  ModelFailureSchema,
  ProviderModelCatalogStateSchema,
  ProviderCatalogEntrySchema,
  ProviderModelCatalogEntrySchema,
  SlotSelectionSchema,
  ModelSlotDescriptorSchema,
  ModelSlotAvailabilitySchema,
  ModelLoadPlanSchema,
  ModelSlotInstanceLifecycleSchema,
  ModelSlotInstanceSchema,
  ModelSlotActionSchema,
  ModelSlotSchema,
  LocalInferenceAcceleratorSchema,
  LocalInferenceMemoryDomainSchema,
  ModelSlotsStateSchema,
  LocalInferenceHardwareSchema,
  PRIMARY_SLOT_ID,
  SECONDARY_SLOT_ID,
  MODEL_SLOT_IDS,
  ProviderModelCatalogMirror,
  ModelSlotsMirror,
  LocalInferenceHardwareMirror,
  LocalModelsMirror,
  OnboardingMirror,
  ModelOfferingTargetIdSchema,
  DownloadAttemptIdSchema,
  ModelServingConfigurationIdSchema,
  ModelInstanceIdSchema,
  RecommendationIdSchema,
  ModelAssessmentIdSchema,
  AssessmentEnvironmentIdSchema,
  LocalModelsStateSchema,
} from "@magnitudedev/acn-protocol"

export {
  ProviderIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
  ReasoningProperty,
  VisionProperty,
} from "@magnitudedev/ai/provider/model"

export type {
  DownloadAttemptId,
  ModelOfferingTargetId,
  RecommendationId,
  LocalModel,
  LocalModelRecommendation,
  LocalModelsState,
  AgentCommunicationMessage,
  RawClipboardImageAttachment,
  CreateSessionInitial,
  DisplayActor,
  DisplayRootStatus,
  DisplayWorkerStatus,
  DisplayActivity,
  DisplayAttachment,
  DisplayMessage,
  DisplayState,
  DisplayTasks,
  DisplayTimeline,
  DisplayTimelineWindowShape,
  DisplayViewShape,
  DirectoryCandidate,
  ErrorDisplayMessage,
  RawFileImageAttachment,
  RawImageAttachment,
  RawMentionOccurrence,
  ImageAttachment,
  ImageMediaType,
  InterruptedMessage,
  ListSessionsResult,
  MentionCandidate,
  MentionAttachment,
  MentionDirectoryAttachment,
  MentionFileAttachment,
  MentionFileRangeAttachment,
  MentionContentType,
  MentionLineRange,
  MessageAttachment,
  PendingInboundCommunication,
  SearchDirectoriesResult,
  SearchMentionsResult,
  SessionCwdSummary,
  SessionMetadata,
  SessionOptions,
  StreamEvent,
  TaskAssignee,
  TaskDisplayRow,
  TimelineActivity,
  ToolMessage,
} from "@magnitudedev/acn-protocol"
export type * from "@magnitudedev/acn-protocol"

export {
  isRoleId,
  ROLE_IDS,
  ROLE_TO_SLOT,
  DEFAULT_REASONING_EFFORT,
  SLOT_IDS,
  SLOT_DISPLAY_NAMES,
  SLOT_DESCRIPTIONS,
} from "@magnitudedev/roles/constants"
export type { RoleId } from "@magnitudedev/roles/constants"
export type {
  FetchUsageOptions,
  CloudUsageResponse,
} from "@magnitudedev/providers"
export type { UsageQuery } from "@magnitudedev/ai"
export type { UsagePeriod } from "@magnitudedev/acn-protocol"
