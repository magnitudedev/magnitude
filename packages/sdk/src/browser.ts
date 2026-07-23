/**
 * Browser-safe SDK entry.
 *
 * Renderer builds resolve `@magnitudedev/sdk` here. The only daemon lifecycle
 * dependency exposed to browser code is `DaemonSpawner`; concrete browser
 * behavior is provided by `makeRemoteDaemonSpawner`.
 */
export { DaemonSpawnerTag } from "./daemon-spawner"
export type { DaemonSpawner } from "./daemon-spawner"
export { makeRemoteDaemonSpawner } from "./remote-spawner"
export {
  makeAcnJitRuntime,
} from "./acn-jit/acn-recovering-client"
export type { AcnJitRuntime, AcnJitRuntimeOptions } from "./acn-jit/acn-recovering-client"
export {
  NoDaemon,
  DaemonSpawnFailed,
  BinaryNotFound,
  BinaryVersionMismatch,
  RegistrationFileInvalid,
  DownloadFailed,
  ChecksumMismatch,
  DaemonCrashed,
  DaemonError,
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
  ModelSlotLifecycle,
  ModelSlotUnassigned,
  ModelSlotUnloadedLocalModel,
  ModelSlotLoadingLocalModel,
  ModelSlotReady,
  ModelSlotUnloadingLocalModel,
  ModelSlotBlocked,
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
  ModelSlotBlockedReasonSchema,
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
  ModelOfferingTargetIdSchema,
  RecommendationIdSchema,
  LocalModelsStateSchema,
} from "@magnitudedev/protocol"

export {
  ProviderIdSchema,
  ProviderModelIdSchema,
  ReasoningEffortSchema,
  ReasoningProperty,
  VisionProperty,
} from "@magnitudedev/ai/provider/model"

export type {
  ModelOfferingTargetId,
  RecommendationId,
  LocalModel,
  LocalModelRecommendation,
  LocalModelsState,
  AgentCommunicationMessage,
  RawClipboardImageAttachment,
  CreateSessionInitial,
  DisplayActor,
  DisplayActorWork,
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
} from "@magnitudedev/protocol"
export type * from "@magnitudedev/protocol"

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
export type { UsagePeriod } from "@magnitudedev/protocol"
