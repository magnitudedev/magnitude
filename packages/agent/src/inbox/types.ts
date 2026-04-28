import type { ContentPart } from '../content'
import type {
  ResolvedMention,
  ToolResultStatus,
} from '../events'


// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

export type LifecycleHookType = 'spawn' | 'idle'

export type LifecycleReminderFormatter = (agentIds: readonly string[]) => string

export interface LifecycleReminderFormatterSet {
  readonly spawn?: LifecycleReminderFormatter
  readonly idle?: LifecycleReminderFormatter
}

export type LifecycleReminderFormatterMap = Record<string, LifecycleReminderFormatterSet | undefined>

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

type ImagePart = Extract<ContentPart, { readonly type: 'image' }>

type Timestamped<K extends string> = {
  readonly kind: K
  readonly timestamp: number
}

type TimestampedText<K extends string> = Timestamped<K> & {
  readonly text: string
}

// ---------------------------------------------------------------------------
// AgentAtom
// ---------------------------------------------------------------------------

export type AgentAtom =
  | (Timestamped<'thought'> & { readonly text: string })
  | (Timestamped<'tool_call'> & {
      readonly toolCallId: string
      readonly toolName: string
      readonly attributes: Readonly<Record<string, string>>
      readonly body?: string
      readonly status: 'success' | 'error' | 'interrupted'
      readonly exitCode?: number
      readonly error?: string
    })
  | (Timestamped<'message'> & {
      readonly direction: 'to_lead' | 'from_user' | 'from_lead'
      readonly text: string
    })
  | (Timestamped<'error'> & { readonly message: string })
  | (Timestamped<'idle'> & { readonly reason?: 'stable' | 'interrupt' | 'error' })

// ---------------------------------------------------------------------------
// TimelineAttachment
// ---------------------------------------------------------------------------

export type TimelineAttachment =
  | { readonly kind: 'image'; readonly image: ImagePart; readonly filename?: string }
  | ({ readonly kind: 'mention' } & ResolvedMention)

// ---------------------------------------------------------------------------
// PhaseCriteriaPayload
// ---------------------------------------------------------------------------

type PhaseCriteriaBase<S extends string> = {
  readonly source: S
  readonly name: string
  readonly status: 'passed' | 'failed' | 'pending'
  readonly reason?: string
}

// ---------------------------------------------------------------------------
// TurnResultItem / ResultEntry
// ---------------------------------------------------------------------------

export type ToolErrorResultItem = {
  readonly kind: 'tool_error'
  readonly toolName: string
  readonly status: Exclude<ToolResultStatus, 'success'>
  readonly message?: string
}

export type ToolObservationResultItem = {
  readonly kind: 'tool_observation'
  readonly toolName: string
  readonly toolCallId: string
  readonly content: readonly ContentPart[]
}

export type MessageAckResultItem = {
  readonly kind: 'message_ack'
  readonly destination: 'parent'
  readonly chars: number
}

export type NoToolsOrMessagesResultItem = {
  readonly kind: 'no_tools_or_messages'
}

export type TurnResultItem =
  | ToolErrorResultItem
  | ToolObservationResultItem
  | MessageAckResultItem
  | NoToolsOrMessagesResultItem

export type ResultEntry =
  | { readonly kind: 'turn_results'; readonly items: readonly TurnResultItem[] }
  | { readonly kind: 'interrupted' }
  | { readonly kind: 'error'; readonly message: string }
  | { readonly kind: 'noop' }
  | { readonly kind: 'oneshot_liveness' }
  | { readonly kind: 'yield_worker_retrigger' }

// ---------------------------------------------------------------------------
// TimelineEntry
// ---------------------------------------------------------------------------

export type TimelineEntry =
  | (TimestampedText<'user_message'> & { readonly attachments: readonly TimelineAttachment[] })
  | (TimestampedText<'parent_message'>)
  | (Timestamped<'user_bash_command'> & {
      readonly command: string
      readonly cwd: string
      readonly exitCode: number
      readonly stdout: string
      readonly stderr: string
    })
  | (TimestampedText<'user_to_agent'> & { readonly agentId: string })
  | (Timestamped<'agent_block'> & {
      readonly firstAtomTimestamp: number
      readonly lastAtomTimestamp: number
      readonly agentId: string
      readonly role: string
      readonly atoms: readonly AgentAtom[]
    })
  | (Timestamped<'subagent_user_killed'> & { readonly agentId: string; readonly agentType: string })
  | (TimestampedText<'user_presence'> & { readonly confirmed: boolean })
  | (Timestamped<'lifecycle_hook'> & {
      readonly agentId: string
      readonly role: string
      readonly hookType: LifecycleHookType
      readonly taskId?: string
      readonly taskTitle?: string
    })
  | (Timestamped<'task_start_hook'> & { readonly taskId: string; readonly title: string })
  | (Timestamped<'task_idle_hook'> & { readonly taskId: string; readonly title: string; readonly agentId: string })
  | (Timestamped<'task_complete_hook'> & { readonly taskId: string; readonly title: string })
  | (Timestamped<'task_tree_dirty'> & { readonly taskId: string })
  | (Timestamped<'task_tree_view'> & { readonly renderedTree: string })
  | (Timestamped<'task_update'> & {
      readonly action: 'created' | 'cancelled' | 'completed' | 'status_changed'
      readonly taskId: string
      readonly title?: string
      readonly previousStatus?: string
      readonly nextStatus?: string
      readonly cancelledCount?: number
    })
  | (Timestamped<'observation'> & { readonly parts: readonly ContentPart[] })

// ---------------------------------------------------------------------------
// QueuedEntry
// ---------------------------------------------------------------------------

export type QueuedEntry =
  | { readonly lane: 'result'; readonly timestamp: number; readonly seq: number; readonly entry: ResultEntry; readonly coalesceKey?: string }
  | { readonly lane: 'timeline'; readonly timestamp: number; readonly seq: number; readonly entry: TimelineEntry; readonly coalesceKey?: string }
