export { connect, makeClientLayer, protocolLayer, protocolLayerWithRecovery, AcnClientTag } from "./protocol"
export type { AcnClient } from "./protocol"

export { DaemonSpawnerTag } from "./daemon-spawner"
export type { DaemonSpawner } from "./daemon-spawner"
export { makeLocalDaemonSpawner, type SpawnProcess, type LocalSpawnerOptions } from "./local-spawner"
export { makeRemoteDaemonSpawner } from "./remote-spawner"
export {
  makeRecoveringAcnClient,
  recoveringProtocolLayer,
  type RecoveringClientOptions,
} from "./recovering-client"

export { TracingLayer, makeTracingLayer, type MakeTracingLayerOptions } from "./tracing"

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
} from "@magnitudedev/protocol"

export type {
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
  DisplayTimelineEntry,
  DisplayTimelinePresentation,
  DisplayTimelinePresentationMode,
  DisplayTimelineStatusSlot,
  DisplayTimelineWindowShape,
  DisplayTimelineWindowInfo,
  DisplayViewShape,
  DisplayMessageTimelineEntry,
  DisplayToolSummaryTimelineEntry,
  DisplayToolStepTimelineEntry,
  DirectoryCandidate,
  ErrorDisplayMessage,
  RawFileImageAttachment,
  RawImageAttachment,
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
  ToolStepPresentation,
  ToolSummaryPresentation,
  ToolPhase,
  ToolTone,
  ToolIcon,
  ToolDiffHunk,
  ToolDiffSlot,
  ToolFileRef,
  ToolSummaryDetailItem,
  ShellPresentation,
  RawMessageAttachment,
  FileWritePresentation,
  FileEditPresentation,
  FileReadPresentation,
  FileSearchPresentation,
  FileTreePresentation,
  FileViewPresentation,
  WebSearchPresentation,
  WebFetchPresentation,
  SkillPresentation,
  CheckpointPresentation,
  SpawnWorkerPresentation,
  GenericToolPresentation,
  QueryImagePresentation,
} from "@magnitudedev/protocol"
export type * from "@magnitudedev/protocol"

export { createRoles, isRoleId, ROLE_IDS, ROLE_TO_SLOT, DEFAULT_REASONING_EFFORT, resolveReasoningEffort, SLOT_IDS, SLOT_DISPLAY_NAMES, SLOT_DESCRIPTIONS } from "@magnitudedev/roles"
export type { RoleId, SlotId } from "@magnitudedev/roles"

export { resolveBinaryCommand, defaultBinaryPath, defaultDataDir, type ResolveBinaryOptions, type ResolvedBinaryCommand } from "./binary"
export { SDK_VERSION } from "./version"
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
  type StreamDisplayViewFailure,
  type WatchFileFailure,
} from "./errors"

export { isEnvFlagOn } from "@magnitudedev/utils"
export { normalizeReferencedPath } from "./path-utils"

// =============================================================================
// Provider client surface — the sole provider boundary for agent & ACN
// =============================================================================

export {
  ProviderClient,
  createProviderClient,
  type ProviderClientShape,
  type ProviderClientConfig,
  type ProviderConnectionConfig,
  type ProviderRegistryInfo,
  type ProviderRuntimeConfig,
  type ProviderRejection,
  type ProviderClientError,
  type BaseCallOptions,
  type ProviderModelBindOptions,
  type ProviderModel,
  type MagnitudeModelInfo,
  type MagnitudeCallOptions,
  type MagnitudeAdditionalOptions,
  type WebSearchResult,
  type WebSearchError,
  type BalanceQuery,
  type BalanceResponse,
  type FetchBalanceOptions,
  type UsagePeriod,
  makeFileBackedModelCatalog,
  createMagnitudeCompatibleSpec,
  DEFAULT_LLAMACPP_ENDPOINT,
  SUPPORTED_PROVIDER_DEFINITIONS,
  type SupportedProviderDefinition,
  type ProviderAuthKind,
  classifyMagnitudeRejectedResponse,
  tryParseErrorBody,
  type ParsedMagnitudeApiError,
} from "./provider-client"
