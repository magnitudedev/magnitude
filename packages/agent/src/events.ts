/**
 * Coding Agent Event Definitions
 *
 * All events that flow through the agent's event bus.
 * Events have forkId: string | undefined - undefined means root agent.
 * forkId is REQUIRED (not optional) so TypeScript catches missing forkId at compile time.
 */


import type { ContentPart } from './content'
import type { ImageMediaType } from './content'
import type {
  TurnEngineEvent,
  TurnStructureDecodeFailure,
  ToolInputDecodeFailure,
} from '@magnitudedev/turn-engine'
import type { ToolKey } from './catalog'
import type { ObservationPart } from '@magnitudedev/roles'
import type { Skill } from '@magnitudedev/skills'
import type { TaskAssignee } from './tasks/types'


export type Attachment = ImageAttachment | MentionAttachment

export interface ImageAttachment {
  readonly type: 'image'
  readonly base64: string
  readonly mediaType: ImageMediaType
  readonly width: number
  readonly height: number
  readonly filename: string
}

export type MentionAttachment = {
  readonly type: 'mention'
  readonly path: string
  readonly contentType: 'text' | 'image' | 'directory'
}

export type ResolvedMention = {
  readonly path: string
  readonly contentType: 'text' | 'image' | 'directory'
  readonly content?: string
  readonly error?: string
  readonly truncated?: boolean
  readonly originalBytes?: number
}
// =============================================================================
// Strategy & Response Types (defined here to avoid circular imports)
// =============================================================================

export type StrategyId = 'xml-act' | 'native'

// =============================================================================
// Work Agent Types
// =============================================================================



// =============================================================================
// Session Events
// =============================================================================

export interface GitContext {
  readonly branch: string
  readonly status: string  // Condensed status output (max 20 lines)
  readonly recentCommits: string  // Last 5 commits, one-line format
}

export interface SessionContext {
  readonly cwd: string
  readonly workspacePath: string
  readonly platform: 'macos' | 'linux' | 'windows'
  readonly shell: string
  readonly timezone: string
  readonly username: string
  readonly fullName: string | null  // User's full name, null if not available
  readonly git: GitContext | null  // null if not a git repo
  readonly folderStructure: string  // Truncated tree output
  readonly agentsFile: { readonly filename: string; readonly content: string } | null  // Agent instruction file if present
  readonly skills: readonly { readonly name: string; readonly description: string; readonly path: string }[] | null  // Available agent skills
  readonly oneshot?: {
    readonly prompt: string
  }
}

export interface SessionInitialized {
  readonly type: 'session_initialized'
  readonly forkId: null  // Global event, applies to root
  readonly context: SessionContext
}

// =============================================================================
// User Events
// =============================================================================

export interface UserMessage {
  readonly type: 'user_message'
  readonly messageId: string
  readonly forkId: string | null
  readonly timestamp: number
  readonly content: ContentPart[]
  readonly attachments: Attachment[]
  readonly mode: 'text' | 'audio'
  readonly synthetic: boolean  // true when sent by autopilot
  readonly taskMode: boolean
}

export interface ObservationsCaptured {
  readonly type: 'observations_captured'
  readonly forkId: string | null
  readonly turnId: string
  readonly parts: readonly ObservationPart[]
}

export interface UserBashCommand {
  readonly type: 'user_bash_command'
  readonly forkId: null
  readonly timestamp: number
  readonly command: string
  readonly cwd: string
  readonly exitCode: number
  readonly stdout: string
  readonly stderr: string
}

export interface UserMessageReady {
  readonly type: 'user_message_ready'
  readonly messageId: string
  readonly forkId: string | null
  readonly resolvedMentions: readonly ResolvedMention[]
}

// =============================================================================
// Turn Lifecycle
// =============================================================================

export interface TurnStarted {
  readonly type: 'turn_started'
  readonly forkId: string | null
  readonly turnId: string
  readonly chainId: string  // Groups turns within user→stable cycle
}

export interface TurnToolCall {
  readonly toolKey: ToolKey
  readonly group: string
  readonly toolName: string
  readonly result: ToolResult
}

// =============================================================================
// Response Representation
// =============================================================================

export interface ObservedResult {
  readonly toolCallId: string
  readonly toolName: string
  readonly query: string
  readonly content: ContentPart[]
}

export interface TurnOutcomeEvent {
  readonly type: 'turn_outcome'
  readonly forkId: string | null
  readonly turnId: string
  readonly chainId: string
  readonly strategyId: StrategyId
  readonly outcome: TurnOutcome
  /** Actual input token count from LLM provider (via BAML Collector). Null when unavailable (e.g. Codex path, interrupted turns). */
  readonly inputTokens: number | null
  /** Output token count from LLM provider. Null when unavailable. */
  readonly outputTokens: number | null
  /** Cache read token count (Anthropic prompt caching). Null when unavailable. */
  readonly cacheReadTokens: number | null
  /** Cache write token count (Anthropic prompt caching). Null when unavailable. */
  readonly cacheWriteTokens: number | null
  /** Provider ID of the model used for this turn. Null when unavailable. */
  readonly providerId: string | null
  /** Model ID used for this turn. Null when unavailable. */
  readonly modelId: string | null
}

/** @deprecated xml-act paradigm only — kept for orphaned xml-act code. Use toolCallsCount on native path. */
export type TurnYieldTarget = 'user' | 'invoke' | 'worker' | 'parent'

export type TurnFinishReason = 'stop' | 'tool_calls' | 'length' | 'content_filter' | 'other'

export interface TurnCompletion {
  /**
   * Number of tool calls dispatched this turn.
   * Implicit turn control: chain-continue iff toolCallsCount > 0.
   */
  readonly toolCallsCount: number
  /**
   * The model's finish reason from the wire protocol.
   */
  readonly finishReason: TurnFinishReason
  readonly feedback: readonly TurnFeedback[]
}

export type TurnFeedback =
  | { readonly _tag: 'InvalidMessageDestination'; readonly destination: string; readonly message: string }
  | { readonly _tag: 'OneshotLivenessRetriggered' }
  | { readonly _tag: 'YieldWorkerRetriggered' }

export type ParseFailureEvent = TurnStructureDecodeFailure | ToolInputDecodeFailure

export type MagnitudeBillingReason =
  | { readonly _tag: 'SubscriptionRequired'; readonly message: string }
  | { readonly _tag: 'TrialExpired'; readonly message: string }
  | { readonly _tag: 'UsageLimitExceeded'; readonly message: string }

export type ProviderNotReadyDetail =
  | { readonly _tag: 'NotConfigured' }
  | { readonly _tag: 'ProviderDisconnected'; readonly providerId: string; readonly providerName: string }
  | { readonly _tag: 'AuthFailed'; readonly providerId: string; readonly providerName: string }
  | { readonly _tag: 'MagnitudeBilling'; readonly reason: MagnitudeBillingReason }

export type ConnectionFailureDetail =
  | { readonly _tag: 'ProviderError'; readonly httpStatus: number }
  | { readonly _tag: 'TransportError'; readonly httpStatus?: number }
  | { readonly _tag: 'StreamError' }

export type CancelledReason =
  | { readonly _tag: 'UserInterrupt' }
  | { readonly _tag: 'WorkerKilled' }
  | { readonly _tag: 'TurnSuperseded' }

export type SafetyStopReason =
  | { readonly _tag: 'IdenticalResponseCircuitBreaker'; readonly threshold: number }
  | { readonly _tag: 'Other'; readonly message: string }

export type UnexpectedErrorDetail =
  | { readonly _tag: 'EngineDefect' }
  | { readonly _tag: 'ExecutionManagerDefect' }
  | { readonly _tag: 'CortexDefect' }
  | { readonly _tag: 'ProviderDefect' }
  | { readonly _tag: 'ToolRuntimeDefect' }
  | { readonly _tag: 'Unknown' }

export type TurnOutcome =
  | { readonly _tag: 'Completed'; readonly completion: TurnCompletion }
  | { readonly _tag: 'ParseFailure'; readonly error: ParseFailureEvent }
  | { readonly _tag: 'ProviderNotReady'; readonly detail: ProviderNotReadyDetail }
  | { readonly _tag: 'ConnectionFailure'; readonly detail: ConnectionFailureDetail }
  | { readonly _tag: 'ContextWindowExceeded' }
  | { readonly _tag: 'OutputTruncated' }
  | { readonly _tag: 'SafetyStop'; readonly reason: SafetyStopReason }
  | { readonly _tag: 'Cancelled'; readonly reason: CancelledReason }
  | { readonly _tag: 'UnexpectedError'; readonly message: string; readonly detail?: UnexpectedErrorDetail }

/** Whether a turn outcome will chain-continue (start another turn automatically).
 *  This is the single source of truth — projections that need to distinguish
 *  "still working" from "actually idle" must use this instead of checking
 *  Completed+invoke independently. */
export function outcomeWillChainContinue(outcome: TurnOutcome): boolean {
  return (
    (outcome._tag === 'Completed' && outcome.completion.toolCallsCount > 0)
    || outcome._tag === 'ParseFailure'
    || outcome._tag === 'ConnectionFailure'
    || outcome._tag === 'ContextWindowExceeded'
  )
}

// =============================================================================
// Streaming Events
// =============================================================================

export type MessageDestination =
  | { readonly kind: 'user' }
  | { readonly kind: 'parent' }
  | { readonly kind: 'worker'; readonly taskId: string }

export interface MessageStart {
  readonly type: 'message_start'
  readonly forkId: string | null
  readonly turnId: string
  readonly id: string
  readonly destination: MessageDestination
}

export interface ThinkingStart {
  readonly type: 'thinking_start'
  readonly forkId: string | null
  readonly turnId: string
}

export interface ThinkingChunk {
  readonly type: 'thinking_chunk'
  readonly forkId: string | null
  readonly turnId: string
  readonly text: string
}

export interface ThinkingEnd {
  readonly type: 'thinking_end'
  readonly forkId: string | null
  readonly turnId: string
}

export interface MessageChunkEvent {
  readonly type: 'message_chunk'
  readonly forkId: string | null
  readonly turnId: string
  readonly id: string
  readonly text: string
}

export interface MessageEnd {
  readonly type: 'message_end'
  readonly forkId: string | null
  readonly turnId: string
  readonly id: string
}

/**
 * Raw provider output streamed during a turn. Intentionally redundant with parsed
 * streaming events (message_chunk, thinking_chunk, etc.) — the parsed events carry
 * the same content in structured form. This event exists because:
 *
 * 1. Dirty turns (parse failures, truncation) need the original raw text to show
 *    the model what it actually produced so it can correct mistakes.
 * 2. Raw chunks are persisted as they stream, so they survive process crashes.
 *    Without this, crash recovery would lose the raw response entirely since
 *    turn_completed (which arrives after streaming) may never be persisted.
 *
 * If the redundancy cost becomes unacceptable, these events can be filtered out
 * of persistence or stopped — but raw text from past sessions cannot be added
 * retroactively.
 */
export interface RawResponseChunk {
  readonly type: 'raw_response_chunk'
  readonly forkId: string | null
  readonly turnId: string
  readonly text: string
}

// =============================================================================
// Tool Events
// =============================================================================

/** Unified tool event — wraps every xml-act tool-scoped runtime event with agent metadata. */
export interface ToolEvent {
  readonly type: 'tool_event'
  readonly forkId: string | null
  readonly turnId: string
  readonly toolCallId: string
  readonly toolKey: ToolKey
  readonly event: TurnEngineEvent
}

export type ToolDisplay =
  | { readonly type: 'gather'; readonly targetCount: number; readonly paths: readonly string[] }
  | { readonly type: 'write_stats'; readonly path: string; readonly linesWritten: number }

export type ToolResult =
  | { readonly status: 'success'; readonly output: unknown; readonly display?: ToolDisplay }
  | { readonly status: 'error'; readonly message: string }
  | { readonly status: 'rejected'; readonly message: string; readonly reason?: string }
  | { readonly status: 'interrupted' }

export type ToolResultStatus = ToolResult['status']



// =============================================================================
// Autopilot Events
// =============================================================================

/** Autopilot generated a continuation message for the user to review */
export interface AutopilotMessageGenerated {
  readonly type: 'autopilot_message_generated'
  readonly forkId: string | null
  readonly content: string
}

/** Autopilot enabled/disabled by user */
export interface AutopilotToggled {
  readonly type: 'autopilot_toggled'
  readonly forkId: string | null
  readonly enabled: boolean
}

// =============================================================================
// Control
// =============================================================================

export interface Interrupt {
  readonly type: 'interrupt'
  readonly forkId: string | null
  readonly allKilled?: boolean
}

export interface SoftInterrupt {
  readonly type: 'soft_interrupt'
  readonly forkId: string
}

// =============================================================================
// Agent Events
// =============================================================================

/** Agent created and dispatched to a task */
export interface AgentCreated {
  readonly type: 'agent_created'
  readonly forkId: string
  readonly parentForkId: string | null
  readonly agentId: string
  readonly name: string
  readonly role: string
  readonly context: string
  readonly mode: 'clone' | 'spawn'
  readonly taskId: string
  readonly message: string
  readonly outputSchema?: unknown
}

/** Agent killed by its parent while active */
export interface AgentKilled {
  readonly type: 'agent_killed'
  readonly forkId: string
  readonly parentForkId: string | null
  readonly agentId: string
  readonly reason: string
}

/** Active subagent explicitly killed by user from subagent tab close confirmation */
export interface SubagentUserKilled {
  readonly type: 'subagent_user_killed'
  readonly forkId: string
  readonly parentForkId: string | null
  readonly agentId: string
  readonly source: 'tab_close_confirm'
}

/** Idle subagent tab explicitly closed by user from tab close (silent durable close) */
export interface SubagentIdleClosed {
  readonly type: 'subagent_idle_closed'
  readonly forkId: string
  readonly parentForkId: string | null
  readonly agentId: string
  readonly source: 'idle_tab_close'
}

// =============================================================================
// Task Events
// =============================================================================

export interface TaskCreated {
  readonly type: 'task_created'
  readonly forkId: string | null
  readonly taskId: string
  readonly title: string
  readonly parentId: string | null
  readonly after?: string
  readonly timestamp: number
}

export interface TaskUpdated {
  readonly type: 'task_updated'
  readonly forkId: string | null
  readonly taskId: string
  readonly patch: {
    readonly title?: string
    readonly parentId?: string | null
    readonly after?: string
    readonly status?: string
  }
  readonly timestamp: number
}

export interface TaskAssigned {
  readonly type: 'task_assigned'
  readonly forkId: string | null
  readonly taskId: string
  readonly assignee: TaskAssignee
  readonly workerRole?: string
  readonly message: string
  readonly workerInfo?: {
    readonly agentId: string
    readonly forkId: string
    readonly role: string
  }
  readonly replacedWorker?: {
    readonly agentId: string
    readonly forkId: string
  }
  readonly timestamp: number
}

export interface TaskCancelled {
  readonly type: 'task_cancelled'
  readonly forkId: string | null
  readonly taskId: string
  readonly cancelledSubtree: readonly string[]
  readonly killedWorkers: readonly {
    readonly agentId: string
    readonly forkId: string
  }[]
  readonly timestamp: number
}

// =============================================================================
// Compaction Events
// =============================================================================

/** Compaction started - token budget exceeded */
export interface CompactionStarted {
  readonly type: 'compaction_started'
  readonly forkId: string | null
  readonly compactedMessageCount: number  // Number of messages to compact (frozen at trigger time)
}

/** Compaction BAML summarization complete, ready to finalize */
export interface CompactionReady {
  readonly type: 'compaction_ready'
  readonly forkId: string | null
  readonly summary: string
  readonly compactedMessageCount: number
  readonly originalTokenEstimate: number  // Token estimate of compacted messages (for tokensSaved calc)
  readonly refreshedContext: SessionContext | null
}

/** Compaction completed - summary replaces old messages */
export interface CompactionCompleted {
  readonly type: 'compaction_completed'
  readonly forkId: string | null
  readonly summary: string
  readonly compactedMessageCount: number
  readonly tokensSaved: number
  readonly preservedVariables: readonly string[]
  readonly refreshedContext: SessionContext | null  // Fresh session context to replace stale original
}

/** Compaction failed */
export interface CompactionFailed {
  readonly type: 'compaction_failed'
  readonly forkId: string | null
  readonly error: string
}

/** Context limit hit — LLM returned a context-length error */
export interface ContextLimitHit {
  readonly type: 'context_limit_hit'
  readonly forkId: string | null
  readonly error: string  // Original error message for logging
}

// =============================================================================
// Approval Events (UI-initiated)
// =============================================================================

/** User approved a gated tool call */
export interface ToolApproved {
  readonly type: 'tool_approved'
  readonly forkId: string | null
  readonly toolCallId: string
}

/** User rejected a gated tool call */
export interface ToolRejected {
  readonly type: 'tool_rejected'
  readonly forkId: string | null
  readonly toolCallId: string
  readonly reason?: string
}

// =============================================================================
// Union Type
// =============================================================================
export interface Wake {
  readonly type: 'wake'
  readonly forkId: string | null
}

export interface WindowFocusChanged {
  readonly type: 'window_focus_changed'
  readonly forkId: null
  readonly focused: boolean
}

export interface UserReturnConfirmed {
  readonly type: 'user_return_confirmed'
  readonly forkId: null
}



export interface OneshotTask {
  readonly type: 'oneshot_task'
  readonly forkId: null
  readonly prompt: string
}


export type SkillActivated =
  | {
      readonly type: 'skill_activated'
      readonly forkId: string | null
      readonly skillName: string
      readonly skillPath: string
      readonly source: 'user'
      readonly message: string | null
    }
  | {
      readonly type: 'skill_activated'
      readonly forkId: string | null
      readonly skillName: string
      readonly skillPath: string
      readonly source: 'assistant'
    }

// Note: SkillStarted and SkillCompleted event types removed.
// Skills are now activated via the `skill` tool which returns content directly.

export type AppEvent =
  | SessionInitialized
  | OneshotTask
  | UserMessage
  | ObservationsCaptured
  | UserBashCommand
  | UserMessageReady
  | TurnStarted
  | TurnOutcomeEvent
  | MessageStart
  | MessageChunkEvent
  | ThinkingStart
  | ThinkingChunk
  | ThinkingEnd
  | MessageEnd
  | RawResponseChunk
  | ToolEvent
  | AutopilotMessageGenerated
  | AutopilotToggled
  | Wake
  | WindowFocusChanged
  | Interrupt
  | SoftInterrupt
  | CompactionStarted
  | CompactionReady
  | CompactionCompleted
  | CompactionFailed
  | ContextLimitHit
  | ToolApproved
  | ToolRejected
  // Task events
  | TaskCreated
  | TaskUpdated
  | TaskAssigned
  | TaskCancelled
  // Agent events
  | AgentCreated
  | AgentKilled
  | SubagentUserKilled
  | SubagentIdleClosed
  | UserReturnConfirmed
  | SkillActivated

