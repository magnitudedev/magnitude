import type { ContentPart } from '../content'
import type { ObservedResult, ResolvedMention, TurnToolCall } from '../events'

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
      readonly tagName: string
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
  | { readonly kind: 'image'; readonly image: ImagePart }
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

export type PhaseCriteriaPayload =
  | (PhaseCriteriaBase<'agent'> & { readonly agentId: string })
  | (PhaseCriteriaBase<'shell'> & { readonly command: string })
  | PhaseCriteriaBase<'user'>

// ---------------------------------------------------------------------------
// ResultEntry
// ---------------------------------------------------------------------------

export type ResultEntry =
  | { readonly kind: 'tool_results'; readonly toolCalls: readonly TurnToolCall[]; readonly observedResults: readonly ObservedResult[] }
  | { readonly kind: 'interrupted' }
  | { readonly kind: 'error'; readonly message: string }
  | { readonly kind: 'noop' }
  | { readonly kind: 'oneshot_liveness' }

// ---------------------------------------------------------------------------
// TimelineEntry
// ---------------------------------------------------------------------------

export type TimelineEntry =
  | (TimestampedText<'user_message'> & { readonly attachments: readonly TimelineAttachment[] })
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
  | (TimestampedText<'workflow_phase'> & { readonly name?: string; readonly phase?: string })
  | (Timestamped<'phase_criteria'> & { readonly payload: PhaseCriteriaPayload })
  | (Timestamped<'phase_verdict'> & { readonly passed: boolean; readonly verdictText: string; readonly workflowCompleted: boolean })
  | (Timestamped<'skill_started'> & { readonly skillName: string; readonly firstPhase?: string; readonly prompt: string })
  | (Timestamped<'skill_completed'> & { readonly skillName: string })
  | (Timestamped<'lifecycle_hook'> & {
      readonly agentId: string
      readonly role: string
      readonly hookType: LifecycleHookType
      readonly taskId?: string
      readonly taskTitle?: string
    })
  | (Timestamped<'task_type_hook'> & { readonly taskId: string; readonly taskType: string; readonly title: string })
  | (Timestamped<'task_idle_hook'> & { readonly taskId: string; readonly taskType: string; readonly title: string; readonly agentId: string })
  | (Timestamped<'task_tree_dirty'> & { readonly taskId: string })
  | (Timestamped<'task_tree_view'> & { readonly renderedTree: string })
  | (Timestamped<'observation'> & { readonly parts: readonly ContentPart[] })

// ---------------------------------------------------------------------------
// QueuedEntry
// ---------------------------------------------------------------------------

export type QueuedEntry =
  | { readonly lane: 'result'; readonly timestamp: number; readonly seq: number; readonly entry: ResultEntry; readonly coalesceKey?: string }
  | { readonly lane: 'timeline'; readonly timestamp: number; readonly seq: number; readonly entry: TimelineEntry; readonly coalesceKey?: string }
