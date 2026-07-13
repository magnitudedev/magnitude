import { ContentBuilder } from '../../content'
import type { UserPart } from '@magnitudedev/ai'
import { Option } from 'effect'
import {
  WORKER_PROGRESS_USER_MESSAGE_REMINDER,
} from '../../prompts/lead-communication-reminders'
import type { AgentAtom, TimelineEntry, BackgroundProcessStatus } from './types'
import { renderCompactToolCall } from './render-tool-call'
import { renderEscalationMessage } from '../../observer'

import { taskIdleReminder, taskCompleteReminder } from '../../prompts/tasks/index'
import { createTimeBoundaryEmitter, formatTime } from '../render/time-boundaries'
import { renderTimelineUserMessageParts } from '../render/user-message-parts'

export interface RenderTimelineInput {
  timeline: readonly TimelineEntry[]
  timezone: string | null
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
      return `<message ${dir}>${atom.text}</message>`
    }
    case 'error':
      return `<error>${atom.message}</error>`
    case 'idle':
      // xml-act yield tag — kept as literal string for history rendering
      return '<' + 'yield_user/>'
  }
}

function renderTimelineTextLines(
  entry: Exclude<TimelineEntry, { kind: 'observation' | 'lifecycle_hook' | 'task_start_hook' | 'task_idle_hook' | 'task_complete_hook' | 'task_tree_dirty' | 'task_tree_view' | 'task_update' | 'task_reassigned' | 'turn_start' | 'turn_end' | 'background_processes' }>,
): string[] {
  switch (entry.kind) {
    case 'user_message':
      return [`<message from="user">${entry.text}</message>`]
    case 'coordinator_message':
      return [`<message from="coordinator">${entry.text}</message>`]
    case 'user_bash_command':
      return [
        `<user_bash_command cwd="${entry.cwd}" exit_code="${entry.exitCode}">\n<command>${entry.command}</command>\n<stdout>${entry.stdout}</stdout>\n<stderr>${entry.stderr}</stderr>\n</user_bash_command>`,
      ]
    case 'user_to_agent':
      return [`<user-to-agent agent="${entry.agentId}">${entry.text}</user-to-agent>`]
    case 'agent_block': {
      const lines = entry.atoms.map(renderAgentAtom).join('\n')
      return [`<agent id="${entry.agentId}" role="${entry.role}" status="${entry.status}">\n${lines}\n</agent>`]
    }
    case 'worker_user_killed':
      return [`<subagent-user-killed agent="${entry.agentId}" type="${entry.agentType}"/>`]
    case 'detached_process_exited':
      return [`<detached_process_exited pid="${entry.pid}" command="${entry.command}" exit_code="${entry.exitCode}">\n<stdout_path>${entry.stdoutPath}</stdout_path>\n<stderr_path>${entry.stderrPath}</stderr_path>\n</detached_process_exited>`]
    case 'escalation': {
      // Leader escalation: renders as dedicated <escalation_required> block, not here.
      // Worker escalation: render as plain notification in timeline.
      if (entry.observedForkId === null) return []
      return [`<observer_notification>${entry.justification ?? 'Observer recommends contacting advisor.'}</observer_notification>`]
    }
  }
}

function maybeAttentionBullet(entry: TimelineEntry, timezone: string | null): string | null {
  if (entry.kind === 'user_message') return `- user message at ${formatTime(entry.timestamp, timezone)}`
  if (entry.kind === 'user_bash_command') return `- user ran bash command at ${formatTime(entry.timestamp, timezone)}`
  // Leader escalation no longer renders as attention bullet — handled as dedicated block
  if (entry.kind === 'escalation' && entry.observedForkId === null) {
    return null
  }
  if (entry.kind === 'agent_block') {
    if (entry.atoms.some((a) => a.kind === 'error')) return `- ${entry.agentId} errored at ${formatTime(entry.timestamp, timezone)}`
    if (entry.status === 'idle') return `- ${entry.agentId} went idle at ${formatTime(entry.timestamp, timezone)}`
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

function formatCpu(value: number | null): string {
  if (value == null) return ''
  return `${Math.round(value)}%`
}

function formatMemory(bytes: number | null): string {
  if (bytes == null) return ''
  const mb = bytes / (1024 * 1024)
  if (mb < 1024) return `${Math.round(mb)}MB`
  const gb = mb / 1024
  if (gb < 1024) return `${gb.toFixed(1)}GB`
  return `${(gb / 1024).toFixed(1)}TB`
}

function buildBackgroundProcessesLines(
  statuses: readonly BackgroundProcessStatus[],
): string[] {
  if (statuses.length === 0) return []

  return statuses.map((proc) => {
    const ownerLabel = Option.match(proc.ownerAgentId, {
      onNone: () => '',
      onSome: (id) => ` (worker: ${id})`,
    })
    const elapsed = Math.floor(proc.elapsedMs / 1000)
    const metricsLabel = proc.cpuPercent != null && proc.rssBytes != null
      ? ` cpu ${formatCpu(proc.cpuPercent)} mem ${formatMemory(proc.rssBytes)}`
      : ''
    return `pid ${proc.pid} \`${proc.command}\` running ${elapsed}s${metricsLabel}${ownerLabel}`
  })
}

function hasWorkerToLeadMessage(entry: Extract<TimelineEntry, { kind: 'agent_block' }>): boolean {
  return entry.atoms.some(atom => atom.kind === 'message' && atom.direction === 'to_lead')
}

function renderTaskUpdateLine(entry: Extract<TimelineEntry, { kind: 'task_update' }>): string {
  if (entry.action === 'created') {
    const title = Option.getOrElse(entry.title, () => null)
    const titleSuffix = title ? `: "${title}"` : ''
    return `- Task ${entry.taskId} created${titleSuffix}`
  }

  if (entry.action === 'cancelled') {
    const cancelledCount = Option.getOrElse(entry.cancelledCount, () => null)
    const cancelledSuffix = cancelledCount != null ? ` (${cancelledCount} tasks removed)` : ''
    return `- Task ${entry.taskId} cancelled${cancelledSuffix}`
  }

  if (entry.action === 'completed') {
    return `- Task ${entry.taskId} completed`
  }

  const previousStatus = Option.getOrElse(entry.previousStatus, () => 'unknown')
  const nextStatus = Option.getOrElse(entry.nextStatus, () => 'unknown')
  return `- Task ${entry.taskId} status changed: ${previousStatus} -> ${nextStatus}`
}

export function renderTimeline(input: RenderTimelineInput): UserPart[] {
  const builder = new ContentBuilder()

  if (input.timeline.length === 0) return builder.build()

  const timeBoundaries = createTimeBoundaryEmitter(input.timezone)
  let hasWorkerMessage = false
  const lifecycleHooks: Extract<TimelineEntry, { kind: 'lifecycle_hook' }>[] = []
  const taskIdleHooks: Extract<TimelineEntry, { kind: 'task_idle_hook' }>[] = []
  const taskCompleteHooks: Extract<TimelineEntry, { kind: 'task_complete_hook' }>[] = []
  const treeViews: Extract<TimelineEntry, { kind: 'task_tree_view' }>[] = []
  const taskUpdates: Extract<TimelineEntry, { kind: 'task_update' }>[] = []
  const attentionItems: { bullet: string, kind: TimelineEntry['kind'] }[] = []
  const escalationEntries: Extract<TimelineEntry, { kind: 'escalation' }>[] = []

  const isChronological = (e: TimelineEntry): boolean =>
    e.kind === 'user_message' || e.kind === 'observation' || e.kind === 'agent_block' ||
    e.kind === 'coordinator_message' || e.kind === 'user_bash_command' || e.kind === 'user_to_agent' ||
    e.kind === 'worker_user_killed' || e.kind === 'detached_process_exited' ||
    (e.kind === 'escalation' && e.observedForkId !== null)

  const chronologicalIndices = input.timeline
    .map((e, i) => isChronological(e) ? i : -1)
    .filter(i => i !== -1)
  const lastChronologicalIndex = chronologicalIndices[chronologicalIndices.length - 1] ?? -1
  const hasAnyLifecycleHook = input.timeline.some(e => e.kind === 'lifecycle_hook')

  function emitTimeBoundary(timestamp: number) {
    const marker = timeBoundaries.next(timestamp)
    if (!marker) return
    builder.pushText(
      `${builder.hasContent() ? '\n\n' : ''}${marker}`,
    )
  }

  for (let i = 0; i < input.timeline.length; i++) {
    const entry = input.timeline[i]!

    switch (entry.kind) {
      case 'turn_start': {
        emitTimeBoundary(entry.timestamp)
        break
      }

      case 'turn_end': {
        break
      }

      case 'lifecycle_hook': {
        lifecycleHooks.push(entry)
        break
      }

      case 'task_idle_hook': {
        taskIdleHooks.push(entry)
        break
      }

      case 'task_complete_hook': {
        taskCompleteHooks.push(entry)
        break
      }

      case 'task_tree_view': {
        treeViews.push(entry)
        break
      }

      case 'task_update': {
        taskUpdates.push(entry)
        break
      }

      case 'user_message': {
        emitTimeBoundary(entry.timestamp)
        const parts = renderTimelineUserMessageParts(entry)
        for (const part of parts) {
          if (part._tag === 'TextPart') builder.pushText(`\n${part.text}`)
          else builder.pushPart(part)
        }
        const bullet = maybeAttentionBullet(entry, input.timezone)
        if (bullet && (i !== lastChronologicalIndex || hasAnyLifecycleHook)) {
          attentionItems.push({ bullet, kind: entry.kind })
        }
        break
      }

      case 'observation': {
        emitTimeBoundary(entry.timestamp)
        for (const part of entry.parts) {
          if (part._tag === 'TextPart') builder.pushText(`\n${part.text}`)
          else builder.pushPart(part)
        }
        break
      }

      case 'agent_block': {
        if (hasWorkerToLeadMessage(entry)) hasWorkerMessage = true
        emitTimeBoundary(entry.timestamp)
        for (const line of renderTimelineTextLines(entry)) {
          builder.pushText(`\n${line}`)
        }
        const bullet = maybeAttentionBullet(entry, input.timezone)
        if (bullet && (i !== lastChronologicalIndex || hasAnyLifecycleHook)) {
          attentionItems.push({ bullet, kind: entry.kind })
        }
        break
      }

      case 'coordinator_message':
      case 'user_bash_command':
      case 'user_to_agent':
      case 'worker_user_killed':
      case 'detached_process_exited':
      case 'escalation': {
        emitTimeBoundary(entry.timestamp)
        for (const line of renderTimelineTextLines(entry)) {
          builder.pushText(`\n${line}`)
        }
        // Collect leader escalations for dedicated block
        if (entry.kind === 'escalation' && entry.observedForkId === null) {
          escalationEntries.push(entry)
        }
        const bullet = maybeAttentionBullet(entry, input.timezone)
        if (bullet && (i !== lastChronologicalIndex || hasAnyLifecycleHook)) {
          attentionItems.push({ bullet, kind: entry.kind })
        }
        break
      }

      case 'background_processes': {
        const lines = buildBackgroundProcessesLines(entry.processes)
        if (lines.length > 0) {
          builder.pushText(`${builder.hasContent() ? '\n\n' : ''}<background_processes>\n${lines.map(line => `- ${line}`).join('\n')}\n</background_processes>`)
        }
        break
      }

      case 'task_start_hook':
      case 'task_tree_dirty':
      case 'task_reassigned': {
        break
      }

      default: {
        const _exhaustive: never = entry
      }
    }
  }

  const reminderLines = [
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

  // Render escalation_required block ABOVE attention (if any leader escalations)
  if (escalationEntries.length > 0) {
    for (const entry of escalationEntries) {
      if (entry.justification) {
        builder.pushText(`${builder.hasContent() ? '\n\n' : ''}${renderEscalationMessage(entry.justification)}`)
      }
    }
  }

  const trivialAttention = attentionItems.length === 1 && attentionItems[0]?.kind === 'user_message'

  if (attentionItems.length > 0 && !trivialAttention) {
    builder.pushText(`${builder.hasContent() ? '\n\n' : ''}<attention>\n${attentionItems.map((item) => item.bullet).join('\n')}\n</attention>`)
  }

  return builder.build()
}
