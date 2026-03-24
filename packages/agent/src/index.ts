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
  TurnCompleted,
  ObservedResult,
  TurnUnexpectedError,
  TurnResult,
  TurnDecision,
  TurnToolCall,
  ResponsePart,
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
  ChatTitleGenerated,
  PhaseCriteriaVerdict,
  PhaseVerdict,
  PhaseVerdictEntry,
  SkillActivated,
  SkillStarted,

  Attachment,
  ImageAttachment,
} from './events'

// Agents
export type { AgentVariant } from './agents'
export type { PolicyContext } from './agents/types'
export { getAgentDefinition, registerAgentDefinition, clearAgentOverrides } from './agents'

// Constants
export { DEFAULT_CONTEXT_WINDOW, COMPACT_TRIGGER_RATIO, PROSE_DELIM_OPEN, PROSE_DELIM_CLOSE, DEFAULT_CHAT_NAME, USER_BLUR_DEBOUNCE_MS } from './constants'

// Session Context Collection
export { collectSessionContext } from './util/collect-session-context'
export type { CollectSessionContextOptions } from './util/collect-session-context'

// Skill Scanner
export { scanSkills } from './util/skill-scanner'
export type { SkillMetadata } from './util/skill-scanner'
// Frontmatter Utility
export { parseFrontmatter, serializeFrontmatter } from './util/frontmatter'
export type { FrontmatterResult } from './util/frontmatter'

// Workspace
export * from './workspace'

// Projections
export { WorkingStateProjection, shouldTrigger, isStable } from './projections/working-state'
export type { ForkWorkingState, PendingInboundCommunication } from './projections/working-state'

export { MemoryProjection } from './projections/memory'
export { getView } from './projections/memory'
export type { Message, MessageSource, LLMMessage, Perspective, QueuedMessage, QueuedCommsMessage, QueuedSystemMessage, ForkMemoryState } from './projections/memory'
export type { CommsAttachment, CommsEntry, SystemEntry, AgentActivityEntry } from './prompts/agents'
export { CompactionProjection } from './projections/compaction'
export type { ForkCompactionState } from './projections/compaction'

export {
  DisplayProjection,
  getInProgressFileStreams,

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
  UnexpectedErrorMessage,
  ForkResultMessage,
  ForkActivityMessage,
  ForkActivityToolCounts,
  ApprovalRequestMessage,
  PendingInboundCommunicationDisplay,
} from './projections/display'

export { TurnProjection } from './projections/turn'
export type { TurnState, ToolCall } from './projections/turn'

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

export { ChatTitleProjection } from './projections/chat-title'
export type { ChatTitleState, ChatTitleGeneratedSignal } from './projections/chat-title'

export { SessionContextProjection } from './projections/session-context'
export type { SessionContextState } from './projections/session-context'

export { ReplayProjection } from './projections/replay'
export { WorkflowProjection } from './projections/workflow'
export type { WorkflowCriteriaState } from './projections/workflow'

// Line-edit types
export type { EditDiff } from './util/line-edit'

// Execution
export { ExecutionManager, ExecutionManagerLive } from './execution/execution-manager'
export type { ExecutionManagerService, ExecuteOptions, ExecuteResult } from './execution/execution-manager'
export { PermissionRejection } from './execution/permission-rejection'

// Prompt Utilities
export { generateXmlActToolDocs } from './tools/xml-tool-docs'
export { getXmlActProtocol, buildAckTurn } from './prompts/protocol'

// Tool types (re-exported from xml-act and tools packages)
export type { ToolCallEvent } from '@magnitudedev/xml-act'
export type { Tool } from '@magnitudedev/tools'

// Tools
export type {
  ToolKey,
  ToolStateFor,
  ToolEventFor,
} from './tools/tool-definitions'
export type { ToolHandle, ToolState } from './tools/tool-handle'
export { globalTools } from './tools/globals'
export { shellTool, SHELL_TOOLS } from './tools/shell'
export { shellBgTool, SHELL_BG_TOOLS } from './tools/shell-bg'
export { readTool, writeTool, editTool, treeTool, searchTool, fsTools } from './tools/fs'
export { webSearchTool } from './tools/web-search-tool'
export { webFetchTool } from './tools/web-fetch-tool'

export { agentCreateTool, agentKillTool } from './tools/agent-tools'
export {
  clickTool, doubleClickTool, rightClickTool, typeTool, scrollTool, dragTool,
  navigateTool, goBackTool, switchTabTool, newTabTool, screenshotTool, evaluateTool,
  browserTools,
} from './tools/browser-tools'
export type { AgentStateReader } from './tools/fork'
export { skillTool } from './tools/skill'
export type { SkillStateReader } from './tools/skill'
export { phaseSubmitTool } from './tools/phase-submit'
export { phaseVerdictTool, PhaseVerdictContextTag } from './tools/phase-verdict'


// Skills
export { resolveSkill, getUserSkills } from './skills'
export type { ResolvedSkill } from './skills'

// Workers
export { TurnController } from './workers/turn-controller'
export { Cortex } from './workers/cortex'
export { AgentLifecycle } from './workers/agent-lifecycle'
export { LifecycleCoordinator } from './workers/lifecycle-coordinator'
export { Autopilot } from './workers/autopilot'
export { ApprovalWorker } from './workers/approval-worker'
export { WorkflowWorker } from './workers/workflow-worker'

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
  startCopilotAuth,
  exchangeCopilotToken,
  COPILOT_HEADERS,

  detectBrowserModel,
  isBrowserCompatible,
  getBrowserCompatibleModels,
  BROWSER_COMPATIBLE_MODELS,
} from '@magnitudedev/providers'

export type {
  BamlProviderType,
  AuthFlowType,
  AuthMethodDef,
  ModelDefinition,
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
  CopilotOAuthStart,
  
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