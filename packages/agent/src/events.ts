/**
 * Coding Agent Event Definitions
 *
 * All events that flow through the agent's event bus.
 * Events have forkId: string | undefined - undefined means root agent.
 * forkId is REQUIRED (not optional) so TypeScript catches missing forkId at compile time.
 */


import { Brand, Option } from 'effect'
import type { ContextPart } from './content'
import type { ModelAttemptFailureSnapshot, ProviderToolCallId, ToolCallId } from '@magnitudedev/ai'
import type { ToolLifecycleEvent } from '@magnitudedev/harness'
import type { ValidationIssue } from '@magnitudedev/ai'
// ToolKey is intentionally NOT imported here. events.ts participates in a
// circular type dependency: events → toolkits → task-tools → events.
// Using the erased branded type from tools/types.ts breaks the cycle.
// See tools/types.ts for details.
import type { ToolKeyErased } from './tools/types'

import type { Skill } from '@magnitudedev/skills'
import type { RoleId } from '@magnitudedev/roles'
import type { TaskAssignee } from './tasks/types'
import type { ErrorPresentation } from './errors/present'
import type { CompletedTurn } from './window/types'
import type { CompactResult } from './compaction/context'
import type { AgentImageAttachment, MentionOccurrence, MentionResolution } from './attachments'


export type {
  AgentImageAttachment as ImageAttachment,
  AgentMentionAttachment as MentionAttachment,
  MentionOccurrence,
  MentionResolution,
} from './attachments'
// =============================================================================
// Strategy & Response Types (defined here to avoid circular imports)
// =============================================================================

export type StrategyId = 'native'

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
  readonly scratchpadPath: string
  readonly platform: 'macos' | 'linux' | 'windows'
  readonly shell: string
  readonly timezone: string
  readonly username: string
  readonly fullName: string | null  // User's full name, null if not available
  readonly git: GitContext | null  // null if not a git repo
  readonly folderStructure: string  // Truncated tree output
  readonly agentsFile: { readonly filename: string; readonly content: string } | null  // Agent instruction file if present
  readonly skills: readonly { readonly name: string; readonly description: string; readonly path: string }[] | null  // Available agent skills

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
  readonly text: string
  readonly mentions: readonly MentionOccurrence[]
  readonly attachments: readonly AgentImageAttachment[]
  readonly mode: 'text' | 'audio'
  readonly synthetic: boolean  // true when sent by autopilot
  readonly taskMode: boolean
}

export interface ObservationsCaptured {
  readonly type: 'observations_captured'
  readonly forkId: string | null
  readonly turnId: string
  readonly parts: readonly ContextPart[]
}

export type UserBashCommandId = string & Brand.Brand<'UserBashCommandId'>
export const UserBashCommandId = Brand.nominal<UserBashCommandId>()

export interface UserBashCommand {
  readonly type: 'user_bash_command'
  readonly commandId: UserBashCommandId
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
  readonly mentionResolutions: readonly MentionResolution[]
  readonly resolvedMentions?: readonly MentionResolution[]
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
  readonly toolKey: ToolKeyErased
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
  readonly content: ContextPart[]
}

export type AttemptCommitPolicy =
  | { readonly _tag: 'discardPartialAssistant' }
  | { readonly _tag: 'commitErrorOnly' }
  | { readonly _tag: 'commitCleanTurn' }

export interface TurnOutcomeEvent {
  readonly type: 'turn_outcome'
  readonly forkId: string | null
  readonly turnId: string
  readonly chainId: string
  readonly strategyId: StrategyId
  readonly outcome: TurnOutcome
  readonly commitPolicy?: AttemptCommitPolicy
  /** Actual input token count from LLM provider (via BAML Collector). Null when unavailable (e.g. Codex path, interrupted turns). */
  readonly inputTokens: number | null
  /** Output token count from LLM provider. Null when unavailable. */
  readonly outputTokens: number | null
  /** Cache read token count (Anthropic prompt caching). Null when unavailable. */
  readonly cacheReadTokens: number | null
  /** Cache write token count (Anthropic prompt caching). Null when unavailable. */
  readonly cacheWriteTokens: number | null
  /** Estimated cost of the turn in USD. Null when unavailable. */
  readonly cost: number | null
  /** Provider ID of the model used for this turn. Null when unavailable. */
  readonly providerId: string | null
  /** Model ID used for this turn. Null when unavailable. */
  readonly modelId: string | null
}

/** @deprecated xml-act paradigm only — kept for orphaned xml-act code. Use toolCallsCount on native path. */
export type TurnYieldTarget = 'user' | 'invoke' | 'worker' | 'coordinator'

/** Yield target carried in TurnCompletion when an agent explicitly yields. */
export type YieldTarget = 'user' | 'advisor' | 'workers' | 'coordinator'

export type TurnFinishReason = 'stop' | 'tool_calls' | 'length' | 'content_filter' | 'other'

export interface TurnCompletion {
  /**
   * Number of tool calls dispatched this turn.
   * Implicit turn control: chain-continue iff toolCallsCount > 0 AND yieldTarget is null.
   */
  readonly toolCallsCount: number
  /**
   * The model's finish reason from the wire protocol.
   */
  readonly finishReason: TurnFinishReason
  readonly feedback: readonly TurnFeedback[]
  /**
   * When non-null, the agent explicitly yielded and the turn should NOT retrigger.
   * - 'user': yield to user (wait for user input)
   * - 'advisor': yield to advisor (same as user for now)
   * - 'workers': yield to workers (only valid when non-idle workers exist)
   * - 'coordinator': worker yielding back to coordinator agent
   * Null means normal retrigger logic applies.
   */
  readonly yieldTarget: YieldTarget | null
}

export type TurnFeedback =
  | { readonly _tag: 'InvalidMessageDestination'; readonly destination: string; readonly message: string }
  | { readonly _tag: 'OneshotLivenessRetriggered' }

export type ProviderNotReadyDetail =
  | { readonly _tag: 'AuthFailed' }
  | { readonly _tag: 'OutOfSync' }
  | { readonly _tag: 'SubscriptionRequired'; readonly message: string }
  | {
      readonly _tag: 'UsageLimitExceeded'
      readonly message: string
      readonly window: 'five_hour' | 'weekly' | 'monthly'
      readonly resetAt: string
    }

export type ConnectionFailureDetail = {
  readonly _tag: 'ModelAttemptFailure'
  readonly failure: ModelAttemptFailureSnapshot
}

export type CancelledReason =
  | { readonly _tag: 'UserInterrupt' }
  | { readonly _tag: 'WorkerKilled' }
  | { readonly _tag: 'TurnSuperseded' }

export type SafetyStopReason =
  | { readonly _tag: 'IdenticalResponseCircuitBreaker'; readonly threshold: number }
  | { readonly _tag: 'Other'; readonly message: string }

export type UnexpectedErrorDetail =
  | { readonly _tag: 'EngineDefect'; readonly message: string }
  | { readonly _tag: 'ExecutionManagerDefect'; readonly message: string }
  | { readonly _tag: 'CortexDefect'; readonly message: string }
  | { readonly _tag: 'ProviderDefect'; readonly message: string }
  | { readonly _tag: 'ToolRuntimeDefect'; readonly message: string }
  | { readonly _tag: 'Unknown'; readonly message: string }

type WithRequestId<T> = T & { readonly requestId: string | null }

export type TurnOutcome = WithRequestId<
  | { readonly _tag: 'Completed'; readonly completion: TurnCompletion }
  | { readonly _tag: 'ToolInputValidationFailure'; readonly toolCallId: ToolCallId; readonly providerToolCallId: ProviderToolCallId; readonly toolName: string; readonly toolKey: string; readonly issue: ValidationIssue }
  | { readonly _tag: 'ToolExecutionError'; readonly toolCallId: ToolCallId; readonly providerToolCallId: ProviderToolCallId; readonly toolName: string; readonly toolKey: string; readonly error: { readonly message: string } }
  | { readonly _tag: 'GateRejected'; readonly toolCallId: ToolCallId; readonly providerToolCallId: ProviderToolCallId; readonly toolName: string }
  | { readonly _tag: 'ProviderNotReady'; readonly detail: ProviderNotReadyDetail }
  | { readonly _tag: 'ConnectionFailure'; readonly detail: ConnectionFailureDetail }
  | { readonly _tag: 'StreamFailed'; readonly failure: ModelAttemptFailureSnapshot }
  | { readonly _tag: 'ContextWindowExceeded' }
  | { readonly _tag: 'OutputTruncated' }
  | { readonly _tag: 'SafetyStop'; readonly reason: SafetyStopReason }
  | { readonly _tag: 'Cancelled'; readonly reason: CancelledReason }
  | { readonly _tag: 'Overthinking'; readonly limit: number }
  | { readonly _tag: 'UnexpectedError'; readonly detail: UnexpectedErrorDetail }
>

/** Whether a turn outcome will chain-continue (start another turn automatically).
 *  This is the single source of truth — projections that need to distinguish
 *  "still working" from "actually idle" must use this instead of checking
 *  Completed+invoke independently. */
export function outcomeWillChainContinue(outcome: TurnOutcome): boolean {
  if (outcome._tag === 'Completed' && outcome.completion.yieldTarget !== null) return false
  return (
    (outcome._tag === 'Completed' && outcome.completion.toolCallsCount > 0)
    || outcome._tag === 'ToolInputValidationFailure'
    || outcome._tag === 'ToolExecutionError'
    || outcome._tag === 'GateRejected'
    || outcome._tag === 'ConnectionFailure'
    || outcome._tag === 'ContextWindowExceeded'
    || outcome._tag === 'Overthinking'
  )
}

// =============================================================================
// Streaming Events
// =============================================================================

export type MessageDestination =
  | { readonly kind: 'user' }
  | { readonly kind: 'coordinator' }
  | { readonly kind: 'worker'; readonly agentId: string }

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
  readonly providerToolCallId: ProviderToolCallId
  readonly toolKey: ToolKeyErased
  readonly event: ToolLifecycleEvent
}

export type ToolDisplay =
  | { readonly type: 'gather'; readonly targetCount: number; readonly paths: readonly string[] }
  | { readonly type: 'write_stats'; readonly path: string; readonly linesWritten: number }

export type ToolResult =
  | { readonly status: 'success'; readonly output: unknown; readonly display?: ToolDisplay }
  | { readonly status: 'error'; readonly message: string }
  | { readonly status: 'denied'; readonly denial: string }
  | { readonly status: 'interrupted' }

export type ToolResultStatus = ToolResult['status']



// =============================================================================
// Autopilot Events
// =============================================================================

/** Autopilot started generating a continuation message */
export interface AutopilotGenerationStarted {
  readonly type: 'autopilot_generation_started'
  readonly forkId: string | null
}

/** Autopilot generation completed with success or error */
export interface AutopilotOutcome {
  readonly type: 'autopilot_outcome'
  readonly forkId: string | null
  readonly result:
    | { readonly _tag: 'success'; readonly content: string }
    | { readonly _tag: 'error'; readonly message: string }
}

/** Autopilot enabled/disabled by user */
export interface AutopilotToggled {
  readonly type: 'autopilot_toggled'
  readonly forkId: string | null
  readonly enabled: boolean
}

// =============================================================================
// Goal Events
// =============================================================================

/** A goal has been started for the root agent. The prompt window injects it once. */
export interface GoalStarted {
  readonly type: 'goal_started'
  readonly forkId: null
  readonly goalId: string
  readonly objective: string
}

/** The active goal has been explicitly marked complete with evidence. */
export interface GoalFinished {
  readonly type: 'goal_finished'
  readonly forkId: null
  readonly goalId: string
  readonly evidence: string
}

// =============================================================================
// Control
// =============================================================================

export interface Interrupt {
  readonly type: 'interrupt'
  readonly forkId: string | null
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
  readonly role: RoleId
  readonly context: string
  readonly mode: 'clone' | 'spawn'
  readonly taskId: string
  readonly message: string
  readonly outputSchema?: unknown
}

/** Agent killed by its coordinator while active */
export interface AgentKilled {
  readonly type: 'agent_killed'
  readonly forkId: string
  readonly parentForkId: string | null
  readonly agentId: string
  readonly reason: string
}

/** Agent reassigned to a different task */
export interface AgentTaskChanged {
  readonly type: 'agent_task_changed'
  readonly forkId: string
  readonly agentId: string
  readonly oldTaskId: string
  readonly newTaskId: string
}

/** Active subagent explicitly killed by user from subagent tab close confirmation */
export interface SubagentUserKilled {
  readonly type: 'worker_user_killed'
  readonly forkId: string
  readonly parentForkId: string | null
  readonly agentId: string
  readonly source: 'tab_close_confirm'
}

/** Idle subagent tab explicitly closed by user from tab close (silent durable close) */
export interface SubagentIdleClosed {
  readonly type: 'worker_idle_closed'
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
  readonly workerRole?: RoleId
  readonly message: string
  readonly workerInfo?: {
    readonly agentId: string
    readonly forkId: string
    readonly role: RoleId
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

/** Compaction outcome — discriminated by isFallback */
export type CompactionOutcome =
  | { readonly isFallback: false; readonly compactResult: CompactResult }
  | { readonly isFallback: true }

/** Compaction BAML summarization complete, ready to finalize */
export type CompactionPrepared = {
  readonly type: 'compaction_prepared'
  readonly forkId: string | null
  readonly turn: CompletedTurn
  readonly compactedMessageCount: number
  readonly inputTokens: number | null
  readonly outputTokens: number | null
  readonly refreshedContext: SessionContext | null
} & CompactionOutcome

/** Compaction injected — minimal signal that compaction results should be applied to the window */
export interface CompactionInjected {
  readonly type: 'compaction_injected'
  readonly forkId: string | null
}

/** Compaction failed */
export interface CompactionFailed {
  readonly type: 'compaction_failed'
  readonly forkId: string | null
  readonly error: string
  readonly presentation: ErrorPresentation | null
}

/** A policy trigger that could not start compaction and has been consumed. */
export interface CompactionDeclined {
  readonly type: 'compaction_declined'
  readonly forkId: string | null
  readonly reason: string
  readonly ephemeral: true
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
// Shell Detach Events
// =============================================================================

/** Published when a shell process detaches. MODEL EVENT — replayed. */
export interface ShellProcessRegistered {
  readonly type: 'shell_process_registered'
  readonly forkId: string | null
  readonly pid: number
  readonly command: string
  readonly ownerAgentId: string | undefined
  readonly startedAt: number
  readonly stdoutPath: string
  readonly stderrPath: string
}

/** Published when a detached shell process exits or is killed. MODEL EVENT — replayed. */
export interface ShellProcessExited {
  readonly type: 'shell_process_exited'
  readonly forkId: string | null
  readonly pid: number
  readonly command: string
  readonly exitCode: number
}

/** Wake trigger when a detached shell process completes. */
export interface ShellCompleted {
  readonly type: 'shell_completed'
  readonly forkId: string | null
  readonly pid: number
  readonly command: string
  readonly exitCode: number
}

/** Periodic snapshot of CPU/memory for running detached shell processes. */
export interface ShellProcessMetrics {
  readonly type: 'shell_process_metrics'
  readonly forkId: null
  readonly samples: readonly {
    readonly pid: number
    readonly cpuPercent: number
    readonly rssBytes: number
    readonly timestamp: number
  }[]
}

// =============================================================================
// Union Type
// =============================================================================
export interface Wake {
  readonly type: 'wake'
  readonly forkId: string | null
}

import type { ObserverOutcome } from './observer'
export type { ObserverOutcome }



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

export interface ChatTitleGenerated {
  readonly type: 'chat_title_generated'
  readonly forkId: null
  readonly title: string
  readonly timestamp: number
}

export type AppEvent =
  | SessionInitialized
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
  | AutopilotGenerationStarted
  | AutopilotOutcome
  | AutopilotToggled
  | GoalStarted
  | GoalFinished
  | Wake
  | ObserverOutcome
  | Interrupt
  | CompactionStarted
  | CompactionPrepared
  | CompactionInjected
  | CompactionFailed
  | CompactionDeclined
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
  | AgentTaskChanged
  | SubagentUserKilled
  | SubagentIdleClosed
  | SkillActivated
  // Shell detach events
  | ShellProcessRegistered
  | ShellProcessExited
  | ShellCompleted
  | ShellProcessMetrics
  | ChatTitleGenerated
