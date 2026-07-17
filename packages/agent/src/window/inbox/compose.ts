import { Option } from 'effect'
import type { ContextPart } from '../../content'
import type { ImageAttachment, MentionOccurrence, MentionResolution } from '../../events'
import type { ObserverJustification } from '../../observer'
import type {
  AgentAtom,
  TimelineUserMessageItem,
  TimelineEntry,
} from './types'

export function toTimelineTurnStart(args: {
  timestamp: number
  turnId: string
}): TimelineEntry {
  return {
    kind: 'turn_start',
    ...args,
  }
}

export function toTimelineTurnEnd(args: {
  timestamp: number
  turnId: string
}): TimelineEntry {
  return {
    kind: 'turn_end',
    ...args,
  }
}

export interface ComposeContextDeps {
  resolveAgentByForkId(
    forkId: string,
  ): { agentId: string; role: string; parentForkId: string | null } | null
}

export function composeTimelineUserMessageItems(input: {
  readonly text: string
  readonly mentions: readonly MentionOccurrence[]
  readonly resolutions: readonly MentionResolution[]
  readonly attachments: readonly ImageAttachment[]
}): TimelineUserMessageItem[] {
  const resolutionById = new Map(input.resolutions.map(resolution => [resolution.occurrenceId, resolution]))
  const inline = input.mentions
    .filter((mention): mention is MentionOccurrence & { placement: { _tag: 'inline'; start: number; end: number } } => mention.placement._tag === 'inline')
    .sort((a, b) => a.placement.start - b.placement.start)
  const items: TimelineUserMessageItem[] = []
  let cursor = 0

  const pushBody = (text: string) => {
    if (text.length > 0) items.push({ kind: 'body', parts: [{ _tag: 'ContextText', text }] })
  }
  const pushMention = (occurrence: MentionOccurrence) => {
    const resolution = resolutionById.get(occurrence.occurrenceId)
    items.push({
      kind: 'mention',
      mention: {
        occurrence,
        resolution: resolution?.status === 'resolved'
          ? { status: 'resolved', parts: resolution.parts, truncated: resolution.truncated }
          : { status: 'failed', reason: resolution?.reason ?? 'Mention was not resolved' },
      },
    })
  }

  for (const occurrence of inline) {
    pushBody(input.text.slice(cursor, occurrence.placement.start))
    pushMention(occurrence)
    cursor = occurrence.placement.end
  }
  pushBody(input.text.slice(cursor))

  for (const occurrence of input.mentions) {
    if (occurrence.placement._tag === 'trailing') pushMention(occurrence)
  }
  for (const attachment of input.attachments) {
    items.push({ kind: 'attachment', attachmentType: 'image', parts: [attachment.image] })
  }
  return items
}

export function toTimelineUserMessage(args: {
  timestamp: number
  items: readonly TimelineUserMessageItem[]
  synthetic: Option.Option<boolean>
}): TimelineEntry {
  return {
    kind: 'user_message',
    ...args,
  }
}

export function toTimelineCoordinatorMessage(args: {
  timestamp: number
  text: string
}): TimelineEntry {
  return {
    kind: 'coordinator_message',
    ...args,
  }
}

export function toTimelineUserBashCommand(args: {
  timestamp: number
  command: string
  cwd: string
  exitCode: number
  stdout: string
  stderr: string
}): TimelineEntry {
  return {
    kind: 'user_bash_command',
    ...args,
  }
}

export function toTimelineUserToAgent(args: {
  timestamp: number
  agentId: string
  text: string
}): TimelineEntry {
  return {
    kind: 'user_to_agent',
    ...args,
  }
}

export function toTimelineAgentBlock(args: {
  timestamp: number
  firstAtomTimestamp: number
  lastAtomTimestamp: number
  agentId: string
  role: string
  status: string
  atoms: readonly AgentAtom[]
}): TimelineEntry {
  return {
    kind: 'agent_block',
    ...args,
  }
}

export function toTimelineSubagentUserKilled(args: {
  timestamp: number
  agentId: string
  agentType: string
}): TimelineEntry {
  return {
    kind: 'worker_user_killed',
    ...args,
  }
}

export function toTimelineLifecycleHook(args: {
  timestamp: number
  agentId: string
  role: string
  hookType: 'spawn' | 'idle'
  taskId: Option.Option<string>
  taskTitle: Option.Option<string>
}): TimelineEntry {
  return {
    kind: 'lifecycle_hook',
    ...args,
  }
}

export function toTimelineTaskTypeHook(args: {
  timestamp: number
  taskId: string
  title: string
}): TimelineEntry {
  return {
    kind: 'task_start_hook',
    ...args,
  }
}

export function toTimelineTaskIdleHook(args: {
  timestamp: number
  taskId: string
  title: string
  agentId: string
}): TimelineEntry {
  return {
    kind: 'task_idle_hook',
    ...args,
  }
}

export function toTimelineTaskCompleteHook(args: {
  timestamp: number
  taskId: string
  title: string
}): TimelineEntry {
  return {
    kind: 'task_complete_hook',
    ...args,
  }
}

export function toTimelineTaskTreeDirty(args: {
  timestamp: number
  taskId: string
}): TimelineEntry {
  return {
    kind: 'task_tree_dirty',
    ...args,
  }
}

export function toTimelineTaskTreeView(args: {
  timestamp: number
  renderedTree: string
}): TimelineEntry {
  return {
    kind: 'task_tree_view',
    ...args,
  }
}

export function toTimelineTaskUpdate(args: {
  timestamp: number
  action: 'created' | 'cancelled' | 'completed' | 'status_changed'
  taskId: string
  title: Option.Option<string>
  previousStatus: Option.Option<string>
  nextStatus: Option.Option<string>
  cancelledCount: Option.Option<number>
}): TimelineEntry {
  return {
    kind: 'task_update',
    ...args,
  }
}

export function toTimelineTaskReassigned(args: {
  timestamp: number
  oldTaskId: string
  newTaskId: string
}): TimelineEntry {
  return {
    kind: 'task_reassigned',
    ...args,
    text: '',
  }
}

export function toTimelineObservation(args: {
  timestamp: number
  parts: readonly ContextPart[]
}): TimelineEntry {
  return {
    kind: 'observation',
    ...args,
  }
}

export function toTimelineDetachedProcessExited(args: {
  timestamp: number
  pid: number
  command: string
  exitCode: number
  stdoutPath: string
  stderrPath: string
}): TimelineEntry {
  return {
    kind: 'detached_process_exited',
    ...args,
  }
}

export function toTimelineEscalation(args: {
  timestamp: number
  observedForkId: string | null
  observedTurnId: string
  justification: ObserverJustification | null
  coalesceKey: Option.Option<string>
}): TimelineEntry {
  return {
    kind: 'escalation',
    ...args,
  }
}

export function toTimelineBackgroundProcesses(args: {
  timestamp: number
  processes: readonly {
    readonly pid: number
    readonly command: string
    readonly elapsedMs: number
    readonly cpuPercent: number | null
    readonly rssBytes: number | null
    readonly ownerAgentId: Option.Option<string>
  }[]
}): TimelineEntry {
  return {
    kind: 'background_processes',
    ...args,
  }
}
