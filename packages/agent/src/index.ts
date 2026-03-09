/**
 * Magnitude Agent
 *
 * A minimal coding agent using event-core architecture.
 */

// Agent
export { CodingAgent, createCodingAgentClient } from './coding-agent'

// Events
export type {
  AppEvent,
  SessionInitialized,
  SessionContext,
  GitContext,
  UserMessage,
  TurnStarted,
  TurnCompleted,
  InspectResult,
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
  ForkStarted,
  ForkCompleted,
  AutopilotMessageGenerated,
  AutopilotToggled,
  ToolApproved,
  ToolRejected,
  ChatTitleGenerated,
  WorkAgentType,
  Attachment,
  ImageAttachment,
} from './events'

// Agents
export type { AgentVariant } from './agents'
export type { PolicyContext } from './agents/types'
export { getAgentDefinition, registerAgentDefinition, clearAgentOverrides } from './agents'

// Constants
export { DEFAULT_CONTEXT_WINDOW, COMPACT_TRIGGER_RATIO, PROSE_DELIM_OPEN, PROSE_DELIM_CLOSE, getContextLimits, DEFAULT_CHAT_NAME, USER_BLUR_DEBOUNCE_MS } from './constants'

// Session Context Collection
export { collectSessionContext } from './util/collect-session-context'
export type { CollectSessionContextOptions } from './util/collect-session-context'

// Skill Scanner
export { scanSkills } from './util/skill-scanner'
export type { SkillMetadata } from './util/skill-scanner'
// Frontmatter Utility
export { parseFrontmatter, serializeFrontmatter } from './util/frontmatter'
export type { FrontmatterResult } from './util/frontmatter'


// Projections
export { WorkingStateProjection, shouldTrigger, isStable } from './projections/working-state'
export type { ForkWorkingState } from './projections/working-state'

export { MemoryProjection } from './projections/memory'
export { getView } from './projections/memory'
export type { Message, MessageSource, LLMMessage, Perspective, QueuedMessage, QueuedCommsMessage, QueuedSystemMessage, ForkMemoryState } from './projections/memory'
export type { CommsAttachment, CommsEntry, SystemEntry, AgentActivityEntry } from './prompts/agents'
export { CompactionProjection } from './projections/compaction'
export type { ForkCompactionState } from './projections/compaction'

export { DisplayProjection } from './projections/display'
export type {
  DisplayState,
  DisplayMessage,
  UserMessageDisplay,
  QueuedUserMessageDisplay,
  AssistantMessageDisplay,
  ThinkBlockMessage,
  ThinkBlockStep,
  InterruptedMessage,
  UnexpectedErrorMessage,
  ForkResultMessage,
  ForkActivityMessage,
  ForkActivityToolCounts,
  ApprovalRequestMessage,
} from './projections/display'

export { TurnProjection } from './projections/turn'
export type { TurnState, ToolCall } from './projections/turn'

export { ForkProjection } from './projections/fork'
export type { ForkState, ForkInstance, ForkCreated, ForkCompletedSignal } from './projections/fork'


export { ArtifactProjection } from './projections/artifact'
export type { ArtifactState, ArtifactItem } from './projections/artifact'
export { ArtifactAwarenessProjection } from './projections/artifact-awareness'
export type { ForkArtifactAwarenessState } from './projections/artifact-awareness'
export { OutboundMessagesProjection } from './projections/outbound-messages'
export type { OutboundMessagesState, OutboundMessageCompletedSignal } from './projections/outbound-messages'

export { AgentRegistryProjection } from './projections/agent-registry'
export type { AgentRegistryState } from './projections/agent-registry'

export { ChatTitleProjection } from './projections/chat-title'
export type { ChatTitleState, ChatTitleGeneratedSignal } from './projections/chat-title'

export { SessionContextProjection } from './projections/session-context'
export type { SessionContextState } from './projections/session-context'

export { ReplayProjection } from './projections/replay'

// Line-edit types
export type { EditDiff } from './util/line-edit'

// Visual Reducers
export {
  setVisualRegistry, getVisualRegistry,
  defineToolReducer, reducer, defineCluster,
  shellReducer,
  readReducer, writeReducer, editReducer, treeReducer, searchReducer,
  webSearchReducer, webFetchReducer,
  clickReducer, doubleClickReducer, rightClickReducer, typeReducer, scrollReducer, dragReducer,
  navigateReducer, goBackReducer, switchTabReducer, newTabReducer, screenshotReducer, evaluateReducer,
  artifactSyncReducer, artifactReadReducer, artifactWriteReducer, artifactUpdateReducer,
  agentCreateReducer, agentPauseReducer, agentDismissReducer, agentMessageReducer, parentMessageReducer,
  skillReducer,
  resolveEndPhase, isActive,
} from './visuals'
export type {
  ToolVisualReducer, VisualReducerRegistry, ToolReducerConfig, SimpleReducerConfig, ClusterFactory,
  ShellState,
  ReadState, WriteState, EditState, TreeState, TreeEntry, SearchState, SearchMatch,
  Phase, WebSearchState, WebFetchState, BrowserState,
  ArtifactVisualState, ArtifactSyncState, AgentCreateState, AgentIdState, AgentMessageState,
  ParentMessageState, SkillState,
} from './visuals'

// Execution
export { ExecutionManager, ExecutionManagerLive } from './execution/execution-manager'
export type { ExecutionManagerService, ExecuteOptions, ExecuteResult } from './execution/execution-manager'
export { PermissionRejection } from './execution/permission-rejection'

// Prompt Utilities
export { generateXmlActToolDocs } from './tools/xml-tool-docs'

// Tool types (re-exported from xml-act and tools packages)
export type { ToolCallEvent } from '@magnitudedev/xml-act'
export type { Tool } from '@magnitudedev/tools'

// Tools
export { thinkTool, globalTools } from './tools/globals'
export { shellTool, SHELL_TOOLS } from './tools/shell'
export { readTool, writeTool, editTool, treeTool, searchTool, fsTools } from './tools/fs'
export { webSearchTool } from './tools/web-search-tool'
export { webFetchTool } from './tools/web-fetch-tool'

export { agentCreateTool, agentPauseTool, agentDismissTool } from './tools/agent-tools'
export { artifactSyncTool, artifactReadTool, artifactWriteTool, artifactUpdateTool } from './tools/artifact-tools'
export {
  clickTool, doubleClickTool, rightClickTool, typeTool, scrollTool, dragTool,
  navigateTool, goBackTool, switchTabTool, newTabTool, screenshotTool, evaluateTool,
  browserTools,
} from './tools/browser-tools'
export type { ForkStateReader } from './tools/fork'
export { skillTool } from './tools/skill'
export type { SkillStateReader } from './tools/skill'

// Skills
export { resolveSkill, getActiveCoreSkills, getUserSkills, CORE_SKILL_NAMES } from './skills'
export type { CoreSkillEntry, ResolvedSkill, CoreSkillName } from './skills'

// Workers
export { TurnController } from './workers/turn-controller'
export { Cortex } from './workers/cortex'
export { ForkOrchestrator } from './workers/fork-orchestrator'
export { LifecycleCoordinator } from './workers/lifecycle-coordinator'
export { Autopilot } from './workers/autopilot'
export { ApprovalWorker } from './workers/approval-worker'
export { ArtifactSyncWorker } from './workers/artifact-sync-worker'

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
  populateModels,
  initializeModels,
  loadAuth,
  getAuth,
  setAuth,
  removeAuth,
  loadConfig,
  saveConfig,
  setPrimarySelection,
  detectProviders,
  detectDefaultProvider,
  detectProviderAuthMethods,
  buildClientRegistry,
  getPrimaryProviderId,
  getPrimaryModelId,
  getClientRegistry,
  setPrimaryModel,
  clearPrimaryModel,
  initializeProviderState,
  getProviderSummary,
  getPrimaryModelContextWindow,
  validateModelSwitch,
  setLocalProviderConfig,
  getLocalProviderConfig,
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
  isOpenAICodex,
  isCopilotCodex,
  ensureValidAuth,
  resolveModel,
  createModelProxy,
  primary,
  secondary,
  setModel,
  setSecondaryModel,
  clearSecondaryModel,
  setBrowserModel,
  clearBrowserModel,
  setBrowserSelection,
  getModelContextWindow,
  getSlotUsage,
  resetSlotUsage,
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
  ModelSlot,
  ResolvedModel,
  ChatStream,
  ModelProxy,
  CallUsage,
  SlotUsage,
  getModelCost,
} from '@magnitudedev/providers'

// Tracing
export { initTraceSession, writeTrace, getTraceSessionId } from '@magnitudedev/tracing'
export type { TraceSessionMeta, AgentTrace, AgentTraceMeta } from '@magnitudedev/tracing'
export type { ContentPart, ImageMediaType } from './content'
export { textParts, imagePart, textOf, hasImages, wrapTextParts, migrateContent } from './content'