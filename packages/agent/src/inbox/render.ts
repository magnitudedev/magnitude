import { ContentPartBuilder, type ContentPart } from '../content'
import {
  USER_MESSAGE_RESPONSE_REMINDER,
  WORKER_PROGRESS_USER_MESSAGE_REMINDER,
} from '../prompts/lead-communication-reminders'
import type { AgentAtom, ResultEntry, TimelineEntry } from './types'
import { formatError, formatInterrupted, formatNoop, formatOneshotLiveness, formatResults, formatYieldWorkerRetrigger } from './render-results'
import { renderCompactToolCall } from './render-tool-call'

import { taskIdleReminder, taskCompleteReminder } from '../prompts/tasks/index'

export interface FormatInboxInput {
  results: readonly ResultEntry[]
  timeline: readonly TimelineEntry[]
  timezone: string | null
  supportsVision: boolean
}

export function imagePlaceholder(desc: { filename?: string; mediaType?: string; width?: number; height?: number }): string {
  const parts: string[] = ['Image placeholder: current model does not support images']
  const meta: string[] = []
  if (desc.filename) meta.push(desc.filename)
  if (desc.width && desc.height) meta.push(`${desc.width}x${desc.height}`)
  else if (desc.mediaType) meta.push(desc.mediaType)
  if (meta.length > 0) parts.push('—', meta.join(' '))
  return `[${parts.join(' ')}]`
}

function formatTime(timestamp: number, timezone: string | null): string {
  const d = new Date(timestamp)
  return new Intl.DateTimeFormat('en-GB', {
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
    timeZone: timezone ?? undefined,
  }).format(d)
}

function formatDayTime(timestamp: number, timezone: string | null): string {
  const d = new Date(timestamp)
  const date = new Intl.DateTimeFormat('sv-SE', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    timeZone: timezone ?? undefined,
  }).format(d)
  return `${date} ${formatTime(timestamp, timezone)}`
}

function dateKey(timestamp: number, timezone: string | null): string {
  const d = new Date(timestamp)
  return new Intl.DateTimeFormat('sv-SE', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    timeZone: timezone ?? undefined,
  }).format(d)
}

function minuteKey(timestamp: number, timezone: string | null): string {
  const d = new Date(timestamp)
  return new Intl.DateTimeFormat('sv-SE', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
    timeZone: timezone ?? undefined,
  }).format(d)
}

function renderAgentAtom(atom: AgentAtom): string {
  switch (atom.kind) {
    case 'thought':
      return atom.text
    case 'tool_call':
      return renderCompactToolCall({
        toolName: atom.toolName,
        attributes: { ...atom.attributes },
        body: atom.body,
      })
    case 'message': {
      const dir = atom.direction === 'to_lead' ? 'to="lead"' : atom.direction === 'from_user' ? 'from="user"' : 'from="lead"'
      return `<magnitude:message ${dir}>${atom.text}</magnitude:message>`
    }
    case 'error':
      return `<error>${atom.message}</error>`
    case 'idle':
      // xml-act yield tag — kept as literal string for history rendering
      return '<' + 'magnitude:yield_user/>'
  }
}

function renderUserMessageParts(entry: Extract<TimelineEntry, { kind: 'user_message' }>, supportsVision: boolean): ContentPart[] {
  const builder = new ContentPartBuilder()
  builder.pushText(`<magnitude:message from="user">${entry.text}</magnitude:message>`)
  for (const attachment of entry.attachments) {
    if (attachment.kind === 'image') {
      if (supportsVision) {
        builder.pushPart(attachment.image)
      } else {
        builder.pushText(imagePlaceholder({
          filename: attachment.filename,
          mediaType: attachment.image.mediaType,
          width: attachment.image.width,
          height: attachment.image.height,
        }))
      }
      continue
    }

    if (attachment.error) {
      builder.pushText(`\n<mention path="${attachment.path}" type="${attachment.contentType}" error="${attachment.error}"/>`)
      continue
    }

    const truncated = attachment.truncated ? ' truncated="true"' : ''
    const originalBytes = attachment.originalBytes ? ` original_bytes="${attachment.originalBytes}"` : ''
    builder.pushText(`\n<mention path="${attachment.path}" type="${attachment.contentType}"${truncated}${originalBytes}>${attachment.content ?? ''}</mention>`)
  }
  return builder.build()
}

function renderTimelineTextLines(entry: Exclude<TimelineEntry, { kind: 'observation' | 'lifecycle_hook' | 'task_start_hook' | 'task_idle_hook' | 'task_complete_hook' | 'task_tree_dirty' | 'task_tree_view' | 'task_update' }>): string[] {
  switch (entry.kind) {
    case 'user_message':
      return [`<magnitude:message from="user">${entry.text}</magnitude:message>`]
    case 'parent_message':
      return [`<magnitude:message from="parent">${entry.text}</magnitude:message>`]
    case 'user_bash_command':
      return [
        `<user_bash_command cwd="${entry.cwd}" exit_code="${entry.exitCode}">\n<command>${entry.command}</command>\n<stdout>${entry.stdout}</stdout>\n<stderr>${entry.stderr}</stderr>\n</user_bash_command>`,
      ]
    case 'user_to_agent':
      return [`<user-to-agent agent="${entry.agentId}">${entry.text}</user-to-agent>`]
    case 'agent_block': {
      const lines = entry.atoms.map(renderAgentAtom).join('\n')
      const status = entry.atoms[entry.atoms.length - 1]?.kind === 'idle' ? 'idle' : 'working'
      return [`<agent id="${entry.agentId}" role="${entry.role}" status="${status}">\n${lines}\n</agent>`]
    }
    case 'subagent_user_killed':
      return [`<subagent-user-killed agent="${entry.agentId}" type="${entry.agentType}"/>`]
    case 'user_presence':
      return [`<user-presence${entry.confirmed ? ' confirmed="true"' : ''}>${entry.text}</user-presence>`]
  }
}

function maybeAttentionBullet(entry: TimelineEntry, timezone: string | null): string | null {
  if (entry.kind === 'user_message') return `- user message at ${formatTime(entry.timestamp, timezone)}`
  if (entry.kind === 'user_bash_command') return `- user ran bash command at ${formatTime(entry.timestamp, timezone)}`
  if (entry.kind === 'agent_block') {
    if (entry.atoms.some((a) => a.kind === 'error')) return `- ${entry.agentId} errored at ${formatTime(entry.timestamp, timezone)}`
    if (entry.atoms[entry.atoms.length - 1]?.kind === 'idle') return `- ${entry.agentId} went idle at ${formatTime(entry.timestamp, timezone)}`
  }
  return null
}

function buildTaskIdleReminderLines(
  hooks: readonly Extract<TimelineEntry, { kind: 'task_idle_hook' }>[],
): string[] {
  if (hooks.length === 0) return []

  const byTask = new Map<string, Extract<TimelineEntry, { kind: 'task_idle_hook' }>>()
  for (const hook of hooks) {
    byTask.set(hook.taskId, hook)
  }

  return Array.from(byTask.values()).map(
    hook => taskIdleReminder(hook.agentId, hook.taskId, hook.title),
  )
}

function buildTaskCompleteReminderLines(
  hooks: readonly Extract<TimelineEntry, { kind: 'task_complete_hook' }>[],
): string[] {
  if (hooks.length === 0) return []

  const byTask = new Map<string, Extract<TimelineEntry, { kind: 'task_complete_hook' }>>()
  for (const hook of hooks) {
    byTask.set(hook.taskId, hook)
  }

  return Array.from(byTask.values()).map(
    hook => taskCompleteReminder(hook.taskId, hook.title),
  )
}

function hasWorkerToLeadMessage(entry: Extract<TimelineEntry, { kind: 'agent_block' }>): boolean {
  return entry.atoms.some(atom => atom.kind === 'message' && atom.direction === 'to_lead')
}

function renderTaskUpdateLine(entry: Extract<TimelineEntry, { kind: 'task_update' }>): string {
  if (entry.action === 'created') {
    const title = entry.title ? `: "${entry.title}"` : ''
    return `- Task ${entry.taskId} created${title}`
  }

  if (entry.action === 'cancelled') {
    const cancelledSuffix = entry.cancelledCount != null ? ` (${entry.cancelledCount} tasks removed)` : ''
    return `- Task ${entry.taskId} cancelled${cancelledSuffix}`
  }

  if (entry.action === 'completed') {
    return `- Task ${entry.taskId} completed`
  }

  const previousStatus = entry.previousStatus ?? 'unknown'
  const nextStatus = entry.nextStatus ?? 'unknown'
  return `- Task ${entry.taskId} status changed: ${previousStatus} -> ${nextStatus}`
}

export function formatInbox(input: FormatInboxInput): ContentPart[] {
  const builder = new ContentPartBuilder()

  if (input.results.length > 0) {
    builder.pushText('<turn_result>')
    for (const result of input.results) {
      if (result.kind === 'turn_results') builder.pushParts(formatResults(result.items, input.supportsVision))
      else if (result.kind === 'interrupted') builder.pushText(formatInterrupted())
      else if (result.kind === 'error') builder.pushText(formatError(result.message))
      else if (result.kind === 'oneshot_liveness') builder.pushText(formatOneshotLiveness())
      else if (result.kind === 'yield_worker_retrigger') builder.pushText(formatYieldWorkerRetrigger())
      else builder.pushText(formatNoop())
    }
    builder.pushText('\n</turn_result>\n')
  }

  if (input.timeline.length === 0) return builder.build()

  const lifecycleHooks = input.timeline.filter(
    (entry): entry is Extract<TimelineEntry, { kind: 'lifecycle_hook' }> => entry.kind === 'lifecycle_hook',
  )
  const taskIdleHooks = input.timeline.filter(
    (entry): entry is Extract<TimelineEntry, { kind: 'task_idle_hook' }> => entry.kind === 'task_idle_hook',
  )
  const taskCompleteHooks = input.timeline.filter(
    (entry): entry is Extract<TimelineEntry, { kind: 'task_complete_hook' }> => entry.kind === 'task_complete_hook',
  )
  const treeViews = input.timeline.filter(
    (entry): entry is Extract<TimelineEntry, { kind: 'task_tree_view' }> => entry.kind === 'task_tree_view',
  )
  const taskUpdates = input.timeline.filter(
    (entry): entry is Extract<TimelineEntry, { kind: 'task_update' }> => entry.kind === 'task_update',
  )
  const chronological = input.timeline.filter(
    (entry): entry is Exclude<TimelineEntry, { kind: 'lifecycle_hook' | 'task_idle_hook' | 'task_complete_hook' | 'task_start_hook' | 'task_tree_dirty' | 'task_tree_view' | 'task_update' }> =>
      entry.kind !== 'lifecycle_hook'
      && entry.kind !== 'task_idle_hook'
      && entry.kind !== 'task_complete_hook'
      && entry.kind !== 'task_start_hook'
      && entry.kind !== 'task_tree_dirty'
      && entry.kind !== 'task_tree_view'
      && entry.kind !== 'task_update',
  )

  const attentionItems: { bullet: string, kind: TimelineEntry['kind'] }[] = []
  let lastMinute: string | null = null
  let lastDateKey: string | null = null

  for (let i = 0; i < chronological.length; i++) {
    const entry = chronological[i]
    const currentMinute = minuteKey(entry.timestamp, input.timezone)
    if (currentMinute !== lastMinute) {
      const currentDate = dateKey(entry.timestamp, input.timezone)
      const showDate = lastDateKey == null || currentDate !== lastDateKey
      builder.pushText(`${builder.hasContent() ? '\n\n' : ''}--- ${showDate ? formatDayTime(entry.timestamp, input.timezone) : formatTime(entry.timestamp, input.timezone)} ---`)
      lastDateKey = currentDate
      lastMinute = currentMinute
    }

    if (entry.kind === 'observation') {
      for (const part of entry.parts) {
        if (part.type === 'text') builder.pushText(`\n${part.text}`)
        else if (input.supportsVision) builder.pushPart(part)
        else builder.pushText(imagePlaceholder({ mediaType: part.mediaType, width: part.width, height: part.height }))
      }
    } else if (entry.kind === 'user_message') {
      const parts = renderUserMessageParts(entry, input.supportsVision)
      for (const part of parts) {
        if (part.type === 'text') builder.pushText(`\n${part.text}`)
        else builder.pushPart(part)
      }
    } else {
      for (const line of renderTimelineTextLines(entry)) builder.pushText(`\n${line}`)
    }

    const bullet = maybeAttentionBullet(entry, input.timezone)
    if (bullet && (chronological.length - i - 1 > 0 || lifecycleHooks.length > 0)) {
      attentionItems.push({ bullet, kind: entry.kind })
    }
  }

  const hasUserMessage = chronological.some((entry) => entry.kind === 'user_message')
  const hasWorkerMessage = chronological.some(
    (entry) => entry.kind === 'agent_block' && hasWorkerToLeadMessage(entry),
  )

  const reminderLines = [
    //...(hasUserMessage ? [USER_MESSAGE_RESPONSE_REMINDER] : []),
    ...(hasWorkerMessage ? [WORKER_PROGRESS_USER_MESSAGE_REMINDER] : []),
    ...buildTaskIdleReminderLines(taskIdleHooks),
    ...buildTaskCompleteReminderLines(taskCompleteHooks),
  ]
  if (reminderLines.length > 0) {
    builder.pushText(`${builder.hasContent() ? '\n\n' : ''}<reminders>\n${reminderLines.map(line => `- ${line}`).join('\n')}\n</reminders>`)
  }

  if (taskUpdates.length > 0) {
    const lines = taskUpdates.map(renderTaskUpdateLine)
    builder.pushText(`${builder.hasContent() ? '\n\n' : ''}<task_updates>\n${lines.join('\n')}\n</task_updates>`)
  }

  if (treeViews.length > 0) {
    const latestTree = treeViews[treeViews.length - 1]?.renderedTree
    if (latestTree) {
      builder.pushText(`\n\n<task_tree>\n${latestTree}\n</task_tree>`)
    }
  }

  const trivialAttention = attentionItems.length === 1 && attentionItems[0]?.kind === 'user_message'

  if (attentionItems.length > 0 && !trivialAttention) {
    builder.pushText(`${builder.hasContent() ? '\n\n' : ''}<attention>\n${attentionItems.map((item) => item.bullet).join('\n')}\n</attention>`)
  }

  return builder.build()
}
