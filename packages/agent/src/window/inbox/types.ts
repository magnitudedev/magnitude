import { UserPartSchema, type AssistantMessage, type ToolResultMessage } from '@magnitudedev/ai'
import { AgentImageMediaTypeSchema, AgentMentionAttachmentSchema } from '../../attachments'
import { Schema, Option } from 'effect'

const ObserverJustificationSchema = Schema.Literal('difficulty', 'churn', 'frustration')

export const AgentAtomSchema = Schema.Union(
  Schema.Struct({ kind: Schema.Literal('thought'), timestamp: Schema.Number, text: Schema.String }),
  Schema.Struct({
    kind: Schema.Literal('tool_call'),
    timestamp: Schema.Number,
    toolCallId: Schema.String,
    toolName: Schema.String,
    attributes: Schema.Record({ key: Schema.String, value: Schema.String }),
    body: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
    status: Schema.Literal('success', 'error', 'interrupted'),
    exitCode: Schema.optionalWith(Schema.Number, { as: 'Option', exact: true }),
    error: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
  }),
  Schema.Struct({
    kind: Schema.Literal('message'),
    timestamp: Schema.Number,
    direction: Schema.Literal('to_lead', 'from_user', 'from_lead'),
    text: Schema.String,
  }),
  Schema.Struct({ kind: Schema.Literal('error'), timestamp: Schema.Number, message: Schema.String }),
  Schema.Struct({ kind: Schema.Literal('idle'), timestamp: Schema.Number, reason: Schema.optionalWith(Schema.Literal('stable', 'interrupt', 'error'), { as: 'Option', exact: true }) }),
)

export const TimelineAttachmentSchema = Schema.Union(
  Schema.Struct({
    kind: Schema.Literal('image'),
    path: Schema.String,
    filename: Schema.String,
    mediaType: AgentImageMediaTypeSchema,
    width: Schema.Number,
    height: Schema.Number,
  }),
  Schema.Struct({
    kind: Schema.Literal('mention'),
    attachment: AgentMentionAttachmentSchema,
    resolution: Schema.Union(
      Schema.Struct({
        status: Schema.Literal('resolved'),
        content: Schema.String,
        truncated: Schema.Boolean,
        originalBytes: Schema.Number,
      }),
      Schema.Struct({
        status: Schema.Literal('failed'),
        reason: Schema.String,
      }),
    ),
  }),
)

export const BackgroundProcessStatusSchema = Schema.Struct({
  pid: Schema.Number,
  command: Schema.String,
  elapsedMs: Schema.Number,
  cpuPercent: Schema.NullOr(Schema.Number),
  rssBytes: Schema.NullOr(Schema.Number),
  ownerAgentId: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
})

export const TimelineEntrySchema = Schema.Union(
  Schema.Struct({
    kind: Schema.Literal('user_message'),
    timestamp: Schema.Number,
    text: Schema.String,
    attachments: Schema.Array(TimelineAttachmentSchema),
    synthetic: Schema.optionalWith(Schema.Boolean, { as: 'Option', exact: true }),
  }),
  Schema.Struct({ kind: Schema.Literal('coordinator_message'), timestamp: Schema.Number, text: Schema.String }),
  Schema.Struct({
    kind: Schema.Literal('user_bash_command'),
    timestamp: Schema.Number,
    command: Schema.String,
    cwd: Schema.String,
    exitCode: Schema.Number,
    stdout: Schema.String,
    stderr: Schema.String,
  }),
  Schema.Struct({ kind: Schema.Literal('user_to_agent'), timestamp: Schema.Number, text: Schema.String, agentId: Schema.String }),
  Schema.Struct({
    kind: Schema.Literal('agent_block'),
    timestamp: Schema.Number,
    firstAtomTimestamp: Schema.Number,
    lastAtomTimestamp: Schema.Number,
    agentId: Schema.String,
    role: Schema.String,
    status: Schema.String,
    atoms: Schema.Array(AgentAtomSchema),
  }),
  Schema.Struct({ kind: Schema.Literal('worker_user_killed'), timestamp: Schema.Number, agentId: Schema.String, agentType: Schema.String }),
  Schema.Struct({
    kind: Schema.Literal('lifecycle_hook'),
    timestamp: Schema.Number,
    agentId: Schema.String,
    role: Schema.String,
    hookType: Schema.Literal('spawn', 'idle'),
    taskId: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
    taskTitle: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
  }),
  Schema.Struct({ kind: Schema.Literal('task_start_hook'), timestamp: Schema.Number, taskId: Schema.String, title: Schema.String }),
  Schema.Struct({ kind: Schema.Literal('task_idle_hook'), timestamp: Schema.Number, taskId: Schema.String, title: Schema.String, agentId: Schema.String }),
  Schema.Struct({ kind: Schema.Literal('task_complete_hook'), timestamp: Schema.Number, taskId: Schema.String, title: Schema.String }),
  Schema.Struct({ kind: Schema.Literal('task_tree_dirty'), timestamp: Schema.Number, taskId: Schema.String }),
  Schema.Struct({ kind: Schema.Literal('task_tree_view'), timestamp: Schema.Number, renderedTree: Schema.String }),
  Schema.Struct({
    kind: Schema.Literal('task_update'),
    timestamp: Schema.Number,
    action: Schema.Literal('created', 'cancelled', 'completed', 'status_changed'),
    taskId: Schema.String,
    title: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
    previousStatus: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
    nextStatus: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
    cancelledCount: Schema.optionalWith(Schema.Number, { as: 'Option', exact: true }),
  }),
  Schema.Struct({ kind: Schema.Literal('task_reassigned'), timestamp: Schema.Number, text: Schema.String, oldTaskId: Schema.String, newTaskId: Schema.String }),
  Schema.Struct({ kind: Schema.Literal('observation'), timestamp: Schema.Number, parts: Schema.Array(UserPartSchema) }),
  Schema.Struct({
    kind: Schema.Literal('detached_process_exited'),
    timestamp: Schema.Number,
    pid: Schema.Number,
    command: Schema.String,
    exitCode: Schema.Number,
    stdoutPath: Schema.String,
    stderrPath: Schema.String,
  }),
  Schema.Struct({
    kind: Schema.Literal('escalation'),
    timestamp: Schema.Number,
    observedForkId: Schema.NullOr(Schema.String),
    observedTurnId: Schema.String,
    justification: Schema.NullOr(ObserverJustificationSchema),
    coalesceKey: Schema.optionalWith(Schema.String, { as: 'Option', exact: true }),
  }),
  Schema.Struct({ kind: Schema.Literal('turn_start'), timestamp: Schema.Number, turnId: Schema.String }),
  Schema.Struct({ kind: Schema.Literal('turn_end'), timestamp: Schema.Number, turnId: Schema.String }),
  Schema.Struct({
    kind: Schema.Literal('background_processes'),
    timestamp: Schema.Number,
    processes: Schema.Array(BackgroundProcessStatusSchema),
  }),
)

export type AgentAtom = typeof AgentAtomSchema.Type
export type TimelineAttachment = typeof TimelineAttachmentSchema.Type
export type TimelineEntry = typeof TimelineEntrySchema.Type
export type BackgroundProcessStatus = typeof BackgroundProcessStatusSchema.Type


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
// QueuedEntry
// ---------------------------------------------------------------------------

export type QueuedEntry =
  | { readonly lane: 'timeline'; readonly timestamp: number; readonly seq: number; readonly entry: TimelineEntry; readonly coalesceKey: Option.Option<string> }
