/**
 * Coding Agent Event Definitions
 *
 * All events that flow through the agent's event bus.
 * Events have forkId: string | undefined - undefined means root agent.
 * forkId is REQUIRED (not optional) so TypeScript catches missing forkId at compile time.
 */


import type { ContentPart } from './content'
import type { ImageMediaType } from './content'
import type { ToolCallEvent } from '@magnitudedev/xml-act'
import type { ToolKey } from './catalog'
import type { ObservationPart } from '@magnitudedev/roles'
import type { WorkflowSkill } from '@magnitudedev/skills'
import type { TaskTypeId, TaskAssignee } from './tasks'


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
  readonly currentTurnAllowsDirectUserReply?: boolean
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
  /** Provider ID of the model used for this turn. Null when unavailable. */
  readonly providerId: string | null
  /** Model ID used for this turn. Null when unavailable. */
  readonly modelId: string | null
}

export type TurnDecision = 'continue' | 'idle' | 'finish'

export type TurnResultErrorCode =
  | 'unclosed_think'
  | 'nonexistent_agent_destination'
  | 'task_outside_assigned_subtree'
  | 'task_operation_error'

export interface TurnResultError {
  readonly code: TurnResultErrorCode
  readonly message: string
}

export type TurnResult =
  | {
      readonly success: true
      readonly turnDecision: 'continue' | 'idle'
      readonly errors?: readonly TurnResultError[]
      readonly oneshotLivenessTriggered?: boolean
    }
  | {
      readonly success: true
      readonly turnDecision: 'finish'
      readonly evidence: string
      readonly errors?: readonly TurnResultError[]
      readonly oneshotLivenessTriggered?: boolean
    }
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
  readonly toolKey: ToolKey
  readonly event: ToolCallEvent
}

export type ToolDisplay =
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
  readonly taskType: TaskTypeId
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

interface PhaseCriteriaVerdictBase {
  readonly type: 'phase_criteria_verdict'
  readonly forkId: string | null
  readonly parentForkId: string | null
  readonly criteriaIndex: number
  readonly criteriaName: string
}

export type PhaseCriteriaVerdict =
  | PhaseCriteriaVerdictBase & {
      readonly criteriaType: 'shell'
      readonly status: 'running'
      readonly command: string
      readonly pid: number
    }
  | PhaseCriteriaVerdictBase & {
      readonly criteriaType: 'shell'
      readonly status: 'passed'
      readonly command: string
    }
  | PhaseCriteriaVerdictBase & {
      readonly criteriaType: 'shell'
      readonly status: 'failed'
      readonly command: string
      readonly reason: string
    }
  | PhaseCriteriaVerdictBase & {
      readonly criteriaType: 'agent'
      readonly status: 'running'
      readonly agentId: string
    }
  | PhaseCriteriaVerdictBase & {
      readonly criteriaType: 'agent'
      readonly status: 'passed'
      readonly agentId: string
      readonly reason: string
    }
  | PhaseCriteriaVerdictBase & {
      readonly criteriaType: 'agent'
      readonly status: 'failed'
      readonly agentId: string
      readonly reason: string
    }
  | PhaseCriteriaVerdictBase & {
      readonly criteriaType: 'user'
      readonly status: 'passed'
      readonly reason: string
    }

export interface PhaseVerdictEntry {
  readonly criteriaIndex: number
  readonly criteriaName: string
  readonly passed: boolean
  readonly reason: string
}

export interface PhaseVerdict {
  readonly type: 'phase_verdict'
  readonly forkId: string | null
  readonly passed: boolean
  readonly verdicts: readonly PhaseVerdictEntry[]
  readonly nextPhasePrompt: string | null
  readonly workflowCompleted: boolean
}

export interface PhaseSubmitted {
  readonly type: 'phase_submitted'
  readonly forkId: string | null
  readonly fields: ReadonlyMap<string, string>
}

export interface PhaseCriteriaStarted {
  readonly type: 'phase_criteria_started'
  readonly forkId: string | null
  readonly criteria: readonly {
    readonly index: number
    readonly name: string
    readonly type: 'shell' | 'agent' | 'user'
  }[]
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

export interface SkillStarted {
  readonly type: 'skill_started'
  readonly forkId: string | null
  readonly source: 'user' | 'assistant'
  readonly skill: WorkflowSkill
}

export interface SkillCompleted {
  readonly type: 'skill_completed'
  readonly forkId: string | null
  readonly skillName: string
}

export type AppEvent =
  | SessionInitialized
  | OneshotTask
  | UserMessage
  | ObservationsCaptured
  | UserBashCommand
  | UserMessageReady
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
  | CompactionStarted
  | CompactionReady
  | CompactionCompleted
  | CompactionFailed
  | ContextLimitHit
  | ToolApproved
  | ToolRejected
  | ChatTitleGenerated
  | PhaseCriteriaVerdict
  | PhaseVerdict
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
  | PhaseSubmitted
  | PhaseCriteriaStarted
  | SkillActivated
  | SkillStarted
  | SkillCompleted

