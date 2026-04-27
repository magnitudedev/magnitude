/**
 * Magnitude Agent
 *
 * A minimal coding agent using event-core architecture.
 */

export type { StorageClient } from '@magnitudedev/storage'

// Model Slots
export { MAGNITUDE_SLOTS, isMagnitudeSlot } from './model-slots'
export type { MagnitudeSlot } from './model-slots'

// Agent
export { CodingAgent, createCodingAgentClient } from './coding-agent'
export type { CreateClientOptions } from './coding-agent'

// Events
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
  TurnCompletion,
  TurnFeedback,
  TurnToolCall,
  StrategyId,
  MessageStart,
  ThinkingChunk,
  ToolEvent,
  ToolResult,
  ToolDisplay,
  Interrupt,
  AutopilotMessageGenerated,
  AutopilotToggled,
  ToolApproved,
  ToolRejected,

  SkillActivated,

  Attachment,
  ImageAttachment,
} from './events'

// Agents
export type { AgentVariant } from './agents/variants'
export { isValidVariant, getSpawnableVariants } from './agents/variants'
export type { PolicyContext } from './agents/types'
export { getAgentDefinition, registerAgentDefinition, clearAgentOverrides } from './agents/registry'

// Constants
export { PROSE_DELIM_OPEN, PROSE_DELIM_CLOSE, DEFAULT_CHAT_NAME, USER_BLUR_DEBOUNCE_MS } from './constants'

// Session Context Collection
export { collectSessionContext } from './util/collect-session-context'
export type { CollectSessionContextOptions } from './util/collect-session-context'

// Skills (loaded from @magnitudedev/skills)
export { loadSkills } from '@magnitudedev/skills'
export type { Skill } from '@magnitudedev/skills'

// Workspace
export * from './workspace'

// Projections
export { MemoryProjection } from './projections/memory'
export { getView } from './projections/memory'
export type { Message, MessageSource, LLMMessage, Perspective, ForkMemoryState } from './projections/memory'
export type { QueuedEntry } from './inbox/types'

export { CompactionProjection } from './projections/compaction'
export type { CompactionState } from './projections/compaction'

export { ToolStateProjection } from './projections/tool-state'
export type { ToolStateProjectionState } from './projections/tool-state'

export { TaskWorkerProjection } from './projections/task-worker'
export type {
  WorkerState,
  WorkerActivity,
  TaskWorkerSnapshot,
  TaskWorkerState,
} from './projections/task-worker'

export {
  DisplayProjection,
} from './projections/display'
export type {
  DisplayState,
  DisplayMessage,
  UserMessageDisplay,
  QueuedUserMessageDisplay,
  AssistantMessageDisplay,
  ThinkBlockMessage,
  ThinkBlockStep,
  CommunicationStep,
  SubagentStartedStep,
  SubagentFinishedStep,
  InterruptedMessage,
  ErrorDisplayMessage,
  ForkResultMessage,
  ForkActivityMessage,
  ForkActivityToolCounts,
  ApprovalRequestMessage,
  PendingInboundCommunicationDisplay,
} from './projections/display'

export { TurnProjection } from './projections/turn'
export type { ToolCall, TurnTrigger, PendingInboundCommunication, ForkTurnState } from './projections/turn'

export { AgentRoutingProjection } from './projections/agent-routing'
export type {
  AgentRoutingState,
  RoutingEntry,
  AgentMessageSignal,
  AgentResponseSignal,
} from './projections/agent-routing'
export { AgentStatusProjection } from './projections/agent-status'
export type {
  AgentInfo,
  AgentStatusState,
  AgentStatus,
  AgentCreatedSignal,
  AgentBecameIdleSignal,
  AgentBecameWorkingSignal,
} from './projections/agent-status'

export { OutboundMessagesProjection } from './projections/outbound-messages'
export type { OutboundMessagesState, OutboundMessageCompletedSignal } from './projections/outbound-messages'

export { SessionContextProjection } from './projections/session-context'
export type { SessionContextState } from './projections/session-context'

export { ReplayProjection } from './projections/replay'
export { TaskGraphProjection, getPrimaryRootTask, getSessionTitleFromTaskGraph } from './projections/task-graph'
export type { TaskGraphState, TaskRecord, TaskStatus, TaskWorkerInfo } from './projections/task-graph'

// Line-edit types
export type { EditDiff } from './util/line-edit'

// Execution
export { ExecutionManager } from './execution/types'
export type { ExecutionManagerService, ExecuteOptions, ExecuteResult } from './execution/types'
export { ExecutionManagerLive } from './execution/execution-manager'
export { PermissionRejection } from './execution/permission-rejection'

// Prompt Utilities
// TODO: Re-add tool docs generation when implemented
// export { generateToolDocs } from './tools/tool-docs'
// TODO: Re-add protocol export when implemented
// export { getProtocol } from './prompts/protocol'

// Tool types (re-exported from xml-act and tools packages)
export type { TurnEngineEvent } from '@magnitudedev/xml-act'
export type { ToolDefinition } from '@magnitudedev/tools'

// Tools
export { catalog, isToolKey, type ToolKey } from './catalog'
export type { ToolHandle } from './tools/tool-handle'
export type { ToolState } from './models'
export type { FileEditState, FileWriteState } from './models'
export { globalTools } from './tools/globals'
export { shellTool, SHELL_TOOLS } from './tools/shell'
export { readTool, writeTool, editTool, treeTool, grepTool, fsTools } from './tools/fs'
// webSearchTool disabled — awaiting Exa reimplementation
export { webFetchTool } from './tools/web-fetch-tool'

export {
  clickTool, doubleClickTool, rightClickTool, typeTool, scrollTool, dragTool,
  navigateTool, goBackTool, switchTabTool, newTabTool, screenshotTool, evaluateTool,
  browserTools,
} from './tools/browser-tools'
export type { AgentStateReader } from './tools/fork'

// Workers
export { TurnController } from './workers/turn-controller'
export { Cortex } from './workers/cortex'
export { AgentLifecycle } from './workers/agent-lifecycle'
export { LifecycleCoordinator } from './workers/lifecycle-coordinator'
export { Autopilot } from './workers/autopilot'
export { ApprovalWorker } from './workers/approval-worker'

export { SessionTitleWorker } from './workers/session-title-worker'

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

// Debug Introspection
export { createDebugStream, getDebugSnapshot } from './projections/debug-introspection'
export type { DebugSnapshot, ProjectionSnapshot, ContextUsage } from './projections/debug-introspection'

// Ambient config
export * from './ambient'

// Providers (re-exported from @magnitudedev/providers)
export {
  PROVIDERS,
  getProvider,
  getProviderIds,

  detectDefaultProvider,
  buildClientRegistry,
  startAnthropicOAuth,
  exchangeAnthropicCode,
  refreshAnthropicToken,
  ANTHROPIC_OAUTH_BETA_HEADERS,
  startOpenAIBrowserOAuth,
  startOpenAIDeviceOAuth,
  refreshOpenAIToken,

  isBrowserCompatible,
  getBrowserCompatibleModels,
  BROWSER_COMPATIBLE_MODELS,
} from '@magnitudedev/providers'

export type {
  BamlProviderType,
  AuthFlowType,
  AuthMethodDef,
  ProviderDefinition,
  AuthInfo,
  ApiKeyAuth,
  OAuthAuth,
  AwsAuth,
  GcpAuth,
  ModelSelection,
  MagnitudeConfig,
  ProviderOptions,
  DetectedProvider,
  DetectedAuthMethod,
  ProviderAuthMethodStatus,
  AnthropicOAuthStart,
  OpenAIBrowserOAuthStart,
  OpenAIDeviceOAuthStart,

  ChatStream,
  CallUsage,
  SlotUsage,
  getModelCost,
} from '@magnitudedev/providers'

// Tracing
export { initTraceSession, writeTrace, getTraceSessionId } from '@magnitudedev/tracing'
export type { TraceSessionMeta, AgentTrace } from '@magnitudedev/tracing'
export { withTraceScope } from './tracing'
export type { TraceScope } from './tracing'
export type { ContentPart, ImageMediaType } from './content'
export { textParts, imagePart, textOf, hasImages, wrapTextParts, migrateContent } from './content'