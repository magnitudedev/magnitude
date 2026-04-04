import { ContentPartBuilder, type ContentPart } from '../content'
import { formatTaskTypeReminder } from '../tasks/guidance'
import type { TaskTypeId } from '../tasks'
import type { AgentAtom, LifecycleReminderFormatterMap, PhaseCriteriaPayload, ResultEntry, TimelineEntry } from './types'
import { formatError, formatInterrupted, formatNoop, formatOneshotLiveness, formatResults } from './render-results'
import { renderCompactToolCall } from './render-tool-call'

export interface FormatInboxInput {
  results: readonly ResultEntry[]
  timeline: readonly TimelineEntry[]
  timezone: string | null
  lifecycleReminderFormatters: LifecycleReminderFormatterMap
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
        tagName: atom.tagName,
        attributes: { ...atom.attributes },
        body: atom.body,
      })
    case 'message': {
      const dir = atom.direction === 'to_lead' ? 'to="lead"' : atom.direction === 'from_user' ? 'from="user"' : 'from="lead"'
      return `<message ${dir}>${atom.text}</message>`
    }
    case 'error':
      return `<error>${atom.message}</error>`
    case 'idle':
      return atom.reason && atom.reason !== 'stable' ? `<idle reason="${atom.reason}"/>` : '<idle/>'
  }
}

function renderPhaseCriteriaPayload(payload: PhaseCriteriaPayload): string {
  if (payload.source === 'agent') {
    return `<phase_criteria name="${payload.name}" status="${payload.status}" type="agent" agent="${payload.agentId}"${payload.reason ? ` reason="${payload.reason}"` : ''}/>`
  }
  if (payload.source === 'shell') {
    return `<phase_criteria name="${payload.name}" status="${payload.status}" type="shell" command="${payload.command}"${payload.reason ? ` reason="${payload.reason}"` : ''}/>`
  }
  return `<phase_criteria name="${payload.name}" status="${payload.status}" type="user"${payload.reason ? ` reason="${payload.reason}"` : ''}/>`
}

function renderPhaseVerdict(entry: Extract<TimelineEntry, { kind: 'phase_verdict' }>): string {
  let body = entry.verdictText
  if (entry.workflowCompleted) {
    body += '\n<workflow_completed/>'
  }
  return `<phase_verdict passed="${entry.passed ? 'true' : 'false'}">${body}</phase_verdict>`
}

function renderUserMessageParts(entry: Extract<TimelineEntry, { kind: 'user_message' }>): ContentPart[] {
  const builder = new ContentPartBuilder()
  builder.pushText(`<message from="user">${entry.text}</message>`)
  for (const attachment of entry.attachments) {
    if (attachment.kind === 'image') {
      builder.pushPart(attachment.image)
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

function renderTimelineTextLines(entry: Exclude<TimelineEntry, { kind: 'observation' | 'lifecycle_hook' | 'task_type_hook' | 'task_idle_hook' | 'task_tree_dirty' | 'task_tree_view' | 'task_update' }>): string[] {
  switch (entry.kind) {
    case 'user_message':
      return [`<message from="user">${entry.text}</message>`]
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

    case 'workflow_phase': {
      const attrs = `${entry.name ? ` name="${entry.name}"` : ''}${entry.phase ? ` phase="${entry.phase}"` : ''}`
      return [`<workflow_phase${attrs}>${entry.text}</workflow_phase>`]
    }
    case 'phase_criteria':
      return [renderPhaseCriteriaPayload(entry.payload)]
    case 'phase_verdict':
      return [renderPhaseVerdict(entry)]
    case 'skill_started': {
      const attrs = `${entry.skillName ? ` name="${entry.skillName}"` : ''}${entry.firstPhase ? ` phase="${entry.firstPhase}"` : ''}`
      return [`<skill${attrs}>${entry.prompt}</skill>`]
    }
    case 'skill_completed':
      return [`<skill_completed name="${entry.skillName}"/>`]
  }
}

function maybeAttentionBullet(entry: TimelineEntry, timezone: string | null): string | null {
  if (entry.kind === 'user_message') return `- user message at ${formatTime(entry.timestamp, timezone)}`
  if (entry.kind === 'agent_block') {
    if (entry.atoms.some((a) => a.kind === 'error')) return `- ${entry.agentId} errored at ${formatTime(entry.timestamp, timezone)}`
    if (entry.atoms[entry.atoms.length - 1]?.kind === 'idle') return `- ${entry.agentId} went idle at ${formatTime(entry.timestamp, timezone)}`
  }
  return null
}

function buildLifecycleReminderLines(
  hooks: readonly Extract<TimelineEntry, { kind: 'lifecycle_hook' }>[],
  formatters: LifecycleReminderFormatterMap,
): string[] {
  if (hooks.length === 0) return []

  const byAgent = new Map<string, Extract<TimelineEntry, { kind: 'lifecycle_hook' }>>()
  for (const hook of hooks) {
    const current = byAgent.get(hook.agentId)
    if (!current) {
      byAgent.set(hook.agentId, hook)
      continue
    }
    if (current.hookType === 'spawn' && hook.hookType === 'idle') {
      byAgent.set(hook.agentId, hook)
      continue
    }
    if (
      current.hookType === 'spawn'
      && hook.hookType === 'spawn'
      && !current.taskId
      && Boolean(hook.taskId)
    ) {
      byAgent.set(hook.agentId, hook)
    }
  }

  const groups = new Map<string, { role: string, hookType: 'spawn' | 'idle', agentIds: string[] }>()
  for (const hook of Array.from(byAgent.values())) {
    const key = `${hook.role}:${hook.hookType}`
    const group = groups.get(key)
    if (!group) {
      groups.set(key, { role: hook.role, hookType: hook.hookType, agentIds: [hook.agentId] })
      continue
    }
    if (!group.agentIds.includes(hook.agentId)) group.agentIds.push(hook.agentId)
  }

  const dedup = new Set<string>()
  const lines: string[] = []
  for (const group of Array.from(groups.values())) {
    const spawnHookWithTask = group.hookType === 'spawn'
      ? hooks.find(h => h.hookType === 'spawn' && h.role === group.role && group.agentIds.includes(h.agentId) && h.taskId && h.taskTitle)
      : undefined

    const formatter = group.hookType === 'spawn'
      ? formatters[group.role]?.spawn
      : formatters[group.role]?.idle
    const text = spawnHookWithTask
      ? `Worker \`${spawnHookWithTask.role}\` assigned to and working on task ${spawnHookWithTask.taskId} ("${spawnHookWithTask.taskTitle}").`
      : formatter
        ? formatter(group.agentIds)
        : group.hookType === 'spawn'
          ? `Worker(s) ${group.agentIds.join(', ')} spawned.`
          : `Worker(s) ${group.agentIds.join(', ')} went idle.`
    if (!dedup.has(text)) {
      dedup.add(text)
      lines.push(text)
    }
  }

  return lines
}

function buildTaskTypeReminderLines(
  hooks: readonly Extract<TimelineEntry, { kind: 'task_type_hook' }>[],
): string[] {
  if (hooks.length === 0) return []

  const byType = new Map<string, Set<string>>()
  for (const hook of hooks) {
    const ids = byType.get(hook.taskType) ?? new Set<string>()
    ids.add(hook.taskId)
    byType.set(hook.taskType, ids)
  }

  const lines: string[] = []
  for (const [taskType, taskIdSet] of Array.from(byType.entries())) {
    const line = formatTaskTypeReminder(Array.from(taskIdSet), taskType as TaskTypeId)
    if (line) lines.push(line)
  }

  return lines
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
    hook => `Worker ${hook.agentId} for task ${hook.taskId} ("${hook.title}") has finished. Review output and either send feedback or mark complete.`,
  )
}

function renderTaskUpdateLine(entry: Extract<TimelineEntry, { kind: 'task_update' }>): string {
  if (entry.action === 'created') {
    const title = entry.title ? `: "${entry.title}"` : ''
    const taskType = entry.taskType ? ` (${entry.taskType})` : ''
    return `- Task ${entry.taskId} created${title}${taskType}`
  }

  if (entry.action === 'cancelled') {
    const cancelledSuffix = entry.cancelledCount != null ? ` (${entry.cancelledCount} tasks removed)` : ''
    return `- Task ${entry.taskId} cancelled${cancelledSuffix}`
  }

  if (entry.action === 'completed') {
    return `- Task ${entry.taskId} completed`
  }

  if (entry.action === 'archived') {
    return `- Task ${entry.taskId} archived`
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
      if (result.kind === 'tool_results') builder.pushParts(formatResults(result.toolCalls, result.observedResults))
      else if (result.kind === 'interrupted') builder.pushText(formatInterrupted())
      else if (result.kind === 'error') builder.pushText(formatError(result.message))
      else if (result.kind === 'oneshot_liveness') builder.pushText(formatOneshotLiveness())
      else builder.pushText(formatNoop())
    }
    builder.pushText('\n</turn_result>\n')
  }

  if (input.timeline.length === 0) return builder.build()

  const lifecycleHooks = input.timeline.filter(
    (entry): entry is Extract<TimelineEntry, { kind: 'lifecycle_hook' }> => entry.kind === 'lifecycle_hook',
  )
  const taskTypeHooks = input.timeline.filter(
    (entry): entry is Extract<TimelineEntry, { kind: 'task_type_hook' }> => entry.kind === 'task_type_hook',
  )
  const taskIdleHooks = input.timeline.filter(
    (entry): entry is Extract<TimelineEntry, { kind: 'task_idle_hook' }> => entry.kind === 'task_idle_hook',
  )
  const treeViews = input.timeline.filter(
    (entry): entry is Extract<TimelineEntry, { kind: 'task_tree_view' }> => entry.kind === 'task_tree_view',
  )
  const taskUpdates = input.timeline.filter(
    (entry): entry is Extract<TimelineEntry, { kind: 'task_update' }> => entry.kind === 'task_update',
  )
  const chronological = input.timeline.filter(
    (entry): entry is Exclude<TimelineEntry, { kind: 'lifecycle_hook' | 'task_type_hook' | 'task_idle_hook' | 'task_tree_dirty' | 'task_tree_view' | 'task_update' }> =>
      entry.kind !== 'lifecycle_hook'
      && entry.kind !== 'task_type_hook'
      && entry.kind !== 'task_idle_hook'
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
        else builder.pushPart(part)
      }
    } else if (entry.kind === 'user_message') {
      const parts = renderUserMessageParts(entry)
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

  const reminderLines = [
    ...buildLifecycleReminderLines(lifecycleHooks, input.lifecycleReminderFormatters),
    ...buildTaskTypeReminderLines(taskTypeHooks),
    ...buildTaskIdleReminderLines(taskIdleHooks),
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
