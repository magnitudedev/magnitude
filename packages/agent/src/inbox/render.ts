import { ContentPartBuilder, type ContentPart } from '../content'
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

function renderTimelineTextLines(entry: Exclude<TimelineEntry, { kind: 'observation' | 'lifecycle_hook' }>): string[] {
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

function isSimpleUserOnly(timeline: readonly TimelineEntry[]): timeline is readonly [Extract<TimelineEntry, { kind: 'user_message' }>] {
  return timeline.length === 1 && timeline[0]?.kind === 'user_message'
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
    }
  }

  const groups = new Map<string, { role: string, hookType: 'spawn' | 'idle', agentIds: string[] }>()
  for (const hook of byAgent.values()) {
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
  for (const group of groups.values()) {
    const formatter = group.hookType === 'spawn'
      ? formatters[group.role]?.spawn
      : formatters[group.role]?.idle
    const text = formatter
      ? formatter(group.agentIds)
      : group.hookType === 'spawn'
        ? `Subagent(s) ${group.agentIds.join(', ')} spawned.`
        : `Subagent(s) ${group.agentIds.join(', ')} went idle.`
    if (!dedup.has(text)) {
      dedup.add(text)
      lines.push(text)
    }
  }

  return lines
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
  const chronological = input.timeline.filter(
    (entry): entry is Exclude<TimelineEntry, { kind: 'lifecycle_hook' }> => entry.kind !== 'lifecycle_hook',
  )

  if (input.results.length === 0 && lifecycleHooks.length === 0 && isSimpleUserOnly(chronological)) {
    const parts = renderUserMessageParts(chronological[0])
    builder.pushParts(parts)
    return builder.build()
  }

  const bullets: string[] = []
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
      bullets.push(bullet)
    }
  }

  const reminderLines = buildLifecycleReminderLines(lifecycleHooks, input.lifecycleReminderFormatters)
  if (reminderLines.length > 0) {
    builder.pushText(`${builder.hasContent() ? '\n\n' : ''}<reminders>\n${reminderLines.map(line => `- ${line}`).join('\n')}\n</reminders>`)
  }

  if (bullets.length > 0) {
    builder.pushText(`${builder.hasContent() ? '\n\n' : ''}<attention>\n${bullets.join('\n')}\n</attention>`)
  }

  return builder.build()
}
