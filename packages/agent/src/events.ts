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
  readonly contentType: 'text' | 'image' | 'directory'
  readonly content?: string  // optional for backward compat with old sessions
}

export type ResolvedMention = {
  path: string
  contentType: 'text' | 'image' | 'directory'
  content?: string
  error?: string
  truncated?: boolean
  originalBytes?: number
}
// =============================================================================
// Strategy & Response Types (defined here to avoid circular imports)
// =============================================================================

export type StrategyId = 'xml-act'

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

export type FileMentionResolved = {
  type: 'file_mention_resolved'
  forkId: string | null
  sourceMessageTimestamp: number
  mentions: readonly ResolvedMention[]
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

export interface ObservedResult {
  readonly toolCallId: string
  readonly tagName: string
  readonly query: string
  readonly content: ContentPart[]
}

export interface TurnCompleted {
  readonly type: 'turn_completed'
  readonly forkId: string | null
  readonly turnId: string
  readonly chainId: string
  readonly strategyId: StrategyId
  readonly responseParts: readonly ResponsePart[]
  readonly toolCalls: readonly TurnToolCall[]
  readonly observedResults: readonly ObservedResult[]
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
  | { readonly success: true; readonly turnDecision: 'continue' | 'yield'; readonly reminder?: string }
  | { readonly success: true; readonly turnDecision: 'finish'; readonly reminder?: string; readonly evidence: string }
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

/** Unified tool event — wraps every xml-act tool-scoped runtime event with agent metadata. */
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

export interface BackgroundProcessRegistered {
  readonly type: 'background_process_registered'
  readonly forkId: string | null
  readonly pid: number
  readonly command: string
  readonly sourceTurnId: string
  readonly startedAt: number
  readonly initialStdout: string
  readonly initialStderr: string
}

export type BackgroundProcessOutput =
  | {
      readonly type: 'background_process_output'
      readonly forkId: string | null
      readonly pid: number
      readonly mode: 'inline'
      readonly stdoutChunk: string
      readonly stderrChunk: string
    }
  | {
      readonly type: 'background_process_output'
      readonly forkId: string | null
      readonly pid: number
      readonly mode: 'tail'
      readonly stdoutChunk: string
      readonly stderrChunk: string
      readonly stdoutLines: number
      readonly stderrLines: number
    }

export interface BackgroundProcessDemoted {
  readonly type: 'background_process_demoted'
  readonly forkId: string | null
  readonly pid: number
  readonly stdoutFilePath: string
  readonly stderrFilePath: string
}

export interface BackgroundProcessExited {
  readonly type: 'background_process_exited'
  readonly forkId: string | null
  readonly pid: number
  readonly exitCode: number | null
  readonly signal: string | null
  readonly status: 'exited' | 'killed'
}


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

/** Agent paused (soft interrupt — current turn completes, then stops) */
export interface AgentPaused {
  readonly type: 'agent_paused'
  readonly forkId: string
  readonly agentId: string
}

/** Agent dismissed (removed permanently) */
export interface AgentDismissed {
  readonly type: 'agent_dismissed'
  readonly forkId: string
  readonly parentForkId: string | null
  readonly agentId: string
  readonly result: unknown
  readonly reason: 'dismissed' | 'interrupted' | 'completed'
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



export interface OneshotTask {
  readonly type: 'oneshot_task'
  readonly forkId: null
  readonly prompt: string
}

export type AppEvent =
  | SessionInitialized
  | OneshotTask
  | UserMessage
  | ObservationsCaptured
  | FileMentionResolved
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
  | BackgroundProcessRegistered
  | BackgroundProcessOutput
  | BackgroundProcessDemoted
  | BackgroundProcessExited
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
  | ChatTitleGenerated
  // Artifact events
  | ArtifactChanged
  | ArtifactSynced
  // Agent events
  | AgentCreated
  | AgentPaused
  | AgentDismissed
  | UserReturnConfirmed