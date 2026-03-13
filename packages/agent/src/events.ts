/**
 * Coding Agent Event Definitions
 *
 * All events that flow through the agent's event bus.
 * Events have forkId: string | undefined - undefined means root agent.
 * forkId is REQUIRED (not optional) so TypeScript catches missing forkId at compile time.
 */

import type { EditDiff } from './util/line-edit'
import type { ContentPart } from './content'
import type { ImageMediaType } from './content'
import type { ToolCallEvent } from '@magnitudedev/xml-act'
import type { ObservationPart } from '@magnitudedev/agent-definition'

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
  readonly contentType: 'text' | 'image'
  readonly content: string
}
// =============================================================================
// Strategy & Response Types (defined here to avoid circular imports)
// =============================================================================

export type StrategyId = 'xml-act'

// =============================================================================
// Work Agent Types
// =============================================================================

export type WorkAgentType = 'explorer' | 'planner' | 'builder' | 'debugger' | 'reviewer' | 'browser'

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
  readonly platform: 'macos' | 'linux' | 'windows'
  readonly shell: string
  readonly timezone: string
  readonly username: string
  readonly fullName: string | null  // User's full name, null if not available
  readonly git: GitContext | null  // null if not a git repo
  readonly folderStructure: string  // Truncated tree output
  readonly agentsFile: { readonly filename: string; readonly content: string } | null  // Agent instruction file if present
  readonly skills: readonly { readonly name: string; readonly description: string; readonly trigger: string; readonly path: string }[] | null  // Available agent skills
  readonly userMemory?: string | null
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
  readonly forkId: string | null
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
  readonly toolKey: string
  readonly group: string
  readonly toolName: string
  readonly result: ToolResult
}

// =============================================================================
// Response Representation
// =============================================================================

/**
 * Durable record of what the model produced in a turn.
 * Used for memory reconstruction. Single text part containing the raw XML.
 */
export type ResponsePart =
  | { readonly type: 'text'; readonly content: string }
  | { readonly type: 'thinking'; readonly content: string }

export type InspectResult =
  | { readonly status: 'resolved'; readonly toolRef: string; readonly query?: string; readonly content: string }
  | { readonly status: 'invalid_ref'; readonly toolRef: string }

export interface TurnCompleted {
  readonly type: 'turn_completed'
  readonly forkId: string | null
  readonly turnId: string
  readonly chainId: string
  readonly strategyId: StrategyId
  readonly responseParts: readonly ResponsePart[]
  readonly toolCalls: readonly TurnToolCall[]
  readonly inspectResults: readonly InspectResult[]
  readonly result: TurnResult
  /** Actual input token count from LLM provider (via BAML Collector). Null when unavailable (e.g. Codex path, interrupted turns). */
  readonly inputTokens: number | null
  /** Output token count from LLM provider. Null when unavailable. */
  readonly outputTokens: number | null
  /** Cache read token count (Anthropic prompt caching). Null when unavailable. */
  readonly cacheReadTokens: number | null
  /** Cache write token count (Anthropic prompt caching). Null when unavailable. */
  readonly cacheWriteTokens: number | null
}

export type TurnDecision = 'continue' | 'yield' | 'finish'

export type TurnResult =
  | { readonly success: true; readonly turnDecision: TurnDecision; readonly reminder?: string }
  | { readonly success: false; readonly error: string; readonly cancelled: boolean }

// Turn unexpected error (irrecoverable - e.g. LLM connection failure after all retries)
export interface TurnUnexpectedError {
  readonly type: 'turn_unexpected_error'
  readonly forkId: string | null
  readonly turnId: string
  readonly message: string
}

// =============================================================================
// Streaming Events
// =============================================================================

export interface MessageStart {
  readonly type: 'message_start'
  readonly forkId: string | null
  readonly turnId: string
  readonly id: string
  readonly dest: string
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
  readonly about: string | null
}

export interface LensStart {
  readonly type: 'lens_start'
  readonly forkId: string | null
  readonly turnId: string
  readonly name: string
}

export interface LensChunk {
  readonly type: 'lens_chunk'
  readonly forkId: string | null
  readonly turnId: string
  readonly text: string
}

export interface LensEnd {
  readonly type: 'lens_end'
  readonly forkId: string | null
  readonly turnId: string
  readonly name: string
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

// =============================================================================
// Tool Events
// =============================================================================

/** Unified tool event — wraps every xml-act ToolCallEvent with agent metadata. */
export interface ToolEvent {
  readonly type: 'tool_event'
  readonly forkId: string | null
  readonly turnId: string
  readonly toolCallId: string
  readonly toolKey: string
  readonly event: ToolCallEvent
  /** Tool-emitted display data (diffs, etc.), only present on ToolExecutionEnded events */
  readonly display?: ToolDisplay
}

export type ToolDisplay =
  | { readonly type: 'edit_diff'; readonly path: string; readonly diffs: readonly EditDiff[] }
  | { readonly type: 'gather'; readonly targetCount: number; readonly paths: readonly string[] }
  | { readonly type: 'write_stats'; readonly path: string; readonly linesWritten: number }

export type ToolResult =
  | { readonly status: 'success'; readonly output: unknown; readonly display?: ToolDisplay }
  | { readonly status: 'error'; readonly message: string }
  | { readonly status: 'rejected'; readonly message: string; readonly reason?: string }
  | { readonly status: 'interrupted' }


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
// Fork Events
// =============================================================================

/** Fork created and started */
export interface ForkStarted {
  readonly type: 'fork_started'
  readonly forkId: string
  readonly parentForkId: string | null
  readonly name: string
  readonly agentId: string
  readonly context: string
  readonly outputSchema?: unknown  // JSON Schema for structured output
  readonly blocking?: boolean  // true for cloneSync/spawnSync - blocks parent execution
  readonly mode: 'clone' | 'spawn'  // clone = inherited context, spawn = fresh context
  readonly role: string  // agent type: WorkAgentType
  readonly taskId: string  // task this fork is working on (was workItemId)
}

/** Fork completed (triggered by dismiss) */
export interface ForkCompleted {
  readonly type: 'fork_completed'
  readonly forkId: string
  readonly parentForkId: string | null
  readonly result: unknown
}


/** Fork removed (after cleanup delay) */
export interface ForkRemoved {
  readonly type: 'fork_removed'
  readonly forkId: string
  readonly parentForkId: string | null
}

// =============================================================================
// Artifact Events
// =============================================================================

/** Artifact content changed (written or edited) */
export interface ArtifactChanged {
  readonly type: 'artifact_changed'
  readonly forkId: string | null
  readonly id: string
  readonly previousContent: string | null
  readonly content: string
}

/** Artifact synced to a file on disk */
export interface ArtifactSynced {
  readonly type: 'artifact_synced'
  readonly forkId: string | null
  readonly id: string
  readonly path: string
}

// =============================================================================
// Agent Events
// =============================================================================

/** Agent created and dispatched to a task */
export interface AgentCreated {
  readonly type: 'agent_created'
  readonly forkId: string | null
  readonly agentId: string
  readonly taskId: string
  readonly agentType: WorkAgentType
  readonly agentForkId: string
  readonly message: string
}

/** Agent paused (soft interrupt — current turn completes, then stops) */
export interface AgentPaused {
  readonly type: 'agent_paused'
  readonly forkId: null
  readonly agentId: string
  readonly agentForkId: string
}

/** Agent dismissed (removed permanently) */
export interface AgentDismissed {
  readonly type: 'agent_dismissed'
  readonly forkId: null
  readonly agentId: string
  readonly agentForkId: string
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
// Chat Title Events
// =============================================================================

/** Chat title auto-generated from conversation */
export interface ChatTitleGenerated {
  readonly type: 'chat_title_generated'
  readonly forkId: null  // Always root
  readonly title: string
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



export type AppEvent =
  | SessionInitialized
  | UserMessage
  | ObservationsCaptured
  | TurnStarted
  | TurnCompleted
  | TurnUnexpectedError
  | MessageStart
  | MessageChunkEvent
  | ThinkingChunk
  | ThinkingEnd
  | LensStart
  | LensChunk
  | LensEnd
  | MessageEnd
  | ToolEvent
  | AutopilotMessageGenerated
  | AutopilotToggled
  | Wake
  | WindowFocusChanged
  | Interrupt
  | SoftInterrupt
  | ForkStarted
  | ForkCompleted
  | ForkRemoved
  | CompactionStarted
  | CompactionReady
  | CompactionCompleted
  | CompactionFailed
  | ContextLimitHit
  | ToolApproved
  | ToolRejected
  | ChatTitleGenerated
  // Artifact events
  | ArtifactChanged
  | ArtifactSynced
  // Agent events
  | AgentCreated
  | AgentPaused
  | AgentDismissed
  | UserReturnConfirmed
