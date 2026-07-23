/**
 * Magnitude Agent
 *
 * A minimal coding agent using event-core architecture.
 */

export type { MagnitudeStorageShape } from '@magnitudedev/storage'

// Agent
export {
  CodingAgent,
  createCodingAgentSession,
  createCodingAgentClient,
  type CodingAgentSession,
  type CodingAgentClient,
} from './coding-agent'
export { deriveSessionWorkStatus, type SessionWorkSnapshot, type SessionWorkStatus } from './session-work-status'
export type { CreateClientOptions } from './coding-agent'

// Errors
export {
  classifyUnknownError,
  present,
  type ErrorPresentation,
  type ErrorSurface,
  type ErrorSeverity,
  type ErrorCta,
  type ActionId,
} from './errors'

// Events
export { UserBashCommandId } from './events'
export type {
  AppEvent,
  SessionInitialized,
  SessionContext,
  GitContext,
  UserMessage,
  TurnStarted,
  TurnOutcomeEvent,
  ObservedResult,

  TurnOutcome,
  TurnYieldTarget,
  YieldTarget,
  TurnCompletion,
  TurnFeedback,
  TurnToolCall,
  StrategyId,
  ThinkingChunk,
  ToolResult,
  ToolDisplay,
  Interrupt,
  AutopilotOutcome,
  AutopilotToggled,
  GoalStarted,
  GoalFinished,
  ToolApproved,
  ToolRejected,

  SkillActivated,
  ChatTitleGenerated,

  ImageAttachment,
  MentionOccurrence,
} from './events'

// Agents
export type { RoleId } from './agents/role-validation'
export { isRoleId, isSpawnableRole, getSpawnableRoles, ROLE_IDS } from './agents/role-validation'
export type { AgentRoleDefinition } from './agents/registry'
export type { PolicyContext } from './agents/types'
export { getAgentDefinition, getForkInfo, registerAgentDefinition, clearAgentOverrides } from './agents/registry'

// Constants
export { PROSE_DELIM_OPEN, PROSE_DELIM_CLOSE, DEFAULT_CHAT_NAME } from './constants'

// Session Context Collection
export { collectSessionContext } from './util/collect-session-context'
export type { CollectSessionContextOptions } from './util/collect-session-context'

// Skills (loaded from @magnitudedev/skills)
export { loadSkills } from '@magnitudedev/skills'
export type { Skill } from '@magnitudedev/skills'

// Scratchpad
export * from './scratchpad'

// Projections
export { WindowProjection } from './window'
export type { WindowEntry, WindowEntrySource, ForkWindowState } from './window'
export type { QueuedEntry } from './window/inbox/types'

export { CompactionProjection } from './projections/compaction'
export type { CompactionState } from './projections/compaction'

export { HarnessStateProjection, getToolHandlesRecord } from './projections/harness-state'
export type { TurnState } from '@magnitudedev/harness'

export { TaskAssignmentProjection } from './projections/task-assignment'
export type {
  WorkerState,
  WorkerActivity,
  TaskAssignmentRow,
  TaskAssignmentState,
} from './projections/task-assignment'

export {
  DisplayTimelineProjection,
} from './display'
export type {
  DisplayState,
  DisplayActor,
  DisplayActorWork,
  DisplayTimeline,
  DisplayTimelineState,
  DisplayMessage,
  UserMessageDisplay,
  QueuedUserMessageDisplay,
  AssistantMessageDisplay,
  ThinkingMessage,
  ToolMessage,
  StatusIndicatorMessage,
  GoalStatusMessage,
  WorkerResumedMessage,
  WorkerFinishedMessage,
  WorkerKilledMessage,
  WorkerUserKilledMessage,
  InterruptedMessage,
  ErrorDisplayMessage,
  ForkResultMessage,
  ForkActivityMessage,
  ForkActivityToolCounts,
  PendingInboundCommunicationDisplay,
  StatusBarActivity,
  StatusBarDecorator,
  StatusToolKey,
} from './display'

export * from './display-view'

export { TurnProjection } from './projections/turn'
export type { ToolCall, TurnTrigger, PendingInboundCommunication, ForkTurnState } from './projections/turn'

export { DetachedProcessProjection } from './projections/detached-process'
export type { DetachedProcessState, TrackedProcess } from './projections/detached-process'

export { AgentRoutingProjection } from './projections/agent-routing'
export type {
  AgentRoutingState,
  RoutingEntry,
  AgentMessageSignal,
  AgentResponseSignal,
} from './projections/agent-routing'
export { AgentLifecycleProjection } from './projections/agent-lifecycle'
export type {
  AgentInfo,
  AgentLifecycleState,
  AgentLifecycleStatus,
  AgentCreatedSignal,
  AgentBecameIdleSignal,
  AgentBecameWorkingSignal,
} from './projections/agent-lifecycle'

export { OutboundMessagesProjection } from './projections/outbound-messages'
export type { OutboundMessagesState, OutboundMessageCompletedSignal } from './projections/outbound-messages'

export { SessionContextProjection } from './projections/session-context'
export type { SessionContextState } from './projections/session-context'

export { GoalProjection } from './projections/goal'
export type { GoalState, ActiveGoal, FinishedGoal } from './projections/goal'

export { ChatTitleProjection } from './projections/chat-title'
export type { ChatTitleState, ChatTitleGeneratedSignal } from './projections/chat-title'


export { TaskGraphProjection, canTransition, isTaskStatus } from './projections/task-graph'
export type { TaskGraphState, TaskRecord, TaskStatus, TaskWorkerInfo } from './projections/task-graph'

// Line-edit types
export type { EditDiff } from './util/line-edit'

// Execution
export { ExecutionManager } from './execution/types'
export type { ExecutionManagerService, ExecuteResult } from './execution/types'
// ExecutionManagerLive — xml-act paradigm, orphaned. Import directly from the file if needed.
export { PermissionRejection } from './execution/permission-rejection'

// Prompt Utilities
// TODO: Re-add tool docs generation when implemented
// export { generateToolDocs } from './tools/tool-docs'

// Tools
export { isToolKey, type ToolKey } from './tools/toolkits'
export type { ToolHandle } from './tools/tool-handle'
export type { ToolState } from './models'
export type {
  FileReadState,
  FileWriteState,
  FileEditState,
  FileSearchState,
  FileTreeState,
  FileViewState,
  ShellState,
  WebSearchState,
  WebFetchState,
  SpawnWorkerState,
  SkillActivationState,
  ReassignWorkerState,
  MessageAdvisorState,
  FinishGoalState,
} from './models'
export { globalTools } from './tools/globals'
export { shellTool } from './tools/shell'
export { readTool, writeTool, editTool, treeTool, grepTool, fsTools } from './tools/fs'
export { webSearchTool } from './tools/web-search'
export { webFetchTool } from './tools/web-fetch-tool'
export { messageAdvisorTool } from './tools/advisor'
export { finishGoalTool } from './tools/goal'

export type { AgentStateReader } from './tools/fork'

// Workers
export { TurnController } from './workers/turn-controller'

export { AgentLifecycle } from './workers/agent-lifecycle'
export { LifecycleCoordinator } from './workers/lifecycle-coordinator'
export { Autopilot } from './workers/autopilot'
export type { AutopilotState } from './projections/autopilot-state'

export { ChatTitleServiceTag, ChatTitleServiceLive } from './workers/chat-title-service'
export type { ChatTitleService } from './workers/chat-title-service'
export { ChatTitleWorker } from './workers/chat-title-worker'

// Persistence
export { ChatPersistence, PersistenceError } from './persistence/chat-persistence-service'
export type {
  ChatPersistenceService,
  SessionMetadata,
} from './persistence/chat-persistence-service'


// Serialization (for persistence)
export {
  serializeEvent,
  serializeEvents,
  deserializeEvent,
  deserializeEvents,
  validateEventOrder,
  testEventRoundTrip
} from './serialization'
export type { SerializedEvent } from './serialization'

// Introspection
export { AgentIntrospectionError } from './introspection/session'
export type {
  AddressedAtlasGroup,
  AddressedAtlasMetrics,
  AddressedAtlasNode,
  AddressedAtlasResident,
  AddressedAtlasSegment,
  AddressedPin,
  AgentIntrospection,
  ContextIntrospection,
  DisplayIntrospection,
  ProjectionIntrospection,
  ProjectionSummary,
  RuntimeIntrospection,
} from './introspection/session'

export { AtifProjection } from './projections/atif/projection'
export type {
  AtifTrajectory,
  AtifStep,
  AtifForkState,
  AtifProjectionState,
} from './projections/atif'
export { serializeAtif } from './projections/atif/serialize'

// Ambient config
export * from './ambient'

// Model resolution
export { AgentModelResolver } from './model/model-resolver'
export type { AgentBoundModel } from './model/model-resolver'

// Execution usage types
export { type AgentCallUsage, fromResponseUsage } from './execution/types'

// Tracing
export { initTraceSession, getTraceSessionId } from '@magnitudedev/tracing'
export type { TraceSessionMeta, AgentCallTrace } from '@magnitudedev/tracing'
export {
  ContextTextPartSchema,
  ContextImagePartSchema,
  ContextPartSchema,
  ContextImageResultSchema,
  textParts,
  textOf,
  hasImages,
  wrapTextParts,
  renderContextImageAnchor,
  renderContextParts,
  isContextImagePart,
} from './content'
export type {
  ContextTextPart,
  ContextImagePart,
  ContextPart,
  ContextImageResult,
  ContextRenderPolicy,
  ImageMediaType,
} from './content'
export {
  captureContextImageFromFile,
  captureContextImageInline,
} from './util/capture-context-image'
