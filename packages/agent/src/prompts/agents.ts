/**
 * Agent communication & dispatch prompt text.
 *
 * Templates for agent-to-agent messaging, task results,
 * agent context building, and agent status reminders.
 */

import { DateTime } from 'luxon'
import type { ConversationEntry } from '../projections/conversation'
import type { ContentPart, ImageMediaType } from '../content'
import type { InspectResult, TurnToolCall } from '../events'
import type { ObservationPart } from '@magnitudedev/agent-definition'
import { formatResults, formatInterrupted, formatError } from './results'


export type CommsAttachment =
  | { readonly kind: 'image'; readonly base64: string; readonly mediaType: ImageMediaType; readonly width: number; readonly height: number }
  | { readonly kind: 'artifact'; readonly id: string; readonly content: string }

export interface AgentActivityEntry {
  readonly agentId: string
  readonly prose: string | null
  readonly toolsCalled: readonly string[]
  readonly artifactsWritten: readonly string[]
}

export type CommsEntry =
  | { readonly kind: 'user'; readonly timestamp: number; readonly text: string; readonly attachments?: readonly CommsAttachment[] }
  | { readonly kind: 'agent'; readonly from: string; readonly timestamp: number; readonly text: string; readonly attachments?: readonly CommsAttachment[] }

export type SystemEntry =
  | { readonly kind: 'tool_results'; readonly toolCalls: readonly TurnToolCall[]; readonly inspectResults: readonly InspectResult[]; readonly error?: string }
  | { readonly kind: 'reminder'; readonly text: string }
  | { readonly kind: 'interrupted' }
  | { readonly kind: 'error'; readonly message: string }
  | { readonly kind: 'fork_result'; readonly taskId: string | null; readonly role: string; readonly name: string; readonly result: unknown }
  | { readonly kind: 'agent_activity'; readonly entries: readonly AgentActivityEntry[] }
  | { readonly kind: 'autonomous_ended'; readonly taskId: string }
  | { readonly kind: 'task_feedback'; readonly text: string }
  | { readonly kind: 'observation'; readonly part: ObservationPart }

/** Build the context prompt injected into a sub-agent's fork */
export function buildAgentContext(title: string, message: string, extraContext: string): string {
  const parts: string[] = []
  parts.push('<orchestrator>')
  parts.push(`<title>${title}</title>`)
  parts.push('<message>')
  parts.push(message)
  parts.push('</message>')
  if (extraContext) {
    parts.push(extraContext)
  }
  parts.push('</orchestrator>')
  return parts.join('\n')
}

/** Format task result for parent agent when fork completes */
export function formatTaskResult(taskId: string | null | undefined, role: string, name: string, result: unknown): string {
  return `<task_result taskId="${taskId ?? 'none'}">
Agent: ${role} (${name})
Result: ${JSON.stringify(result)}
</task_result>`
}

/** Format sub-agent response for orchestrator */
export function formatAgentResponse(
  agentId: string,
  message: string,
): string {
  return `<agent_response from="${agentId}">\n${message}\n</agent_response>`
}

/** Format orchestrator message for sub-agent */
export function formatOrchestratorMessage(message: string): string {
  return `<orchestrator_message>\n${message}\n</orchestrator_message>\n<system>Use parent.message() to reply.</system>`
}

function formatTimestamp(timestamp: number, timezone: string | null): string {
  const dt = DateTime.fromMillis(timestamp, { zone: timezone ?? 'local' })
  return dt.toFormat('yyyy-LLL-dd HH:mm:ss')
}

export function formatCommsInbox(entries: readonly CommsEntry[], timezone: string | null): ContentPart[] {
  const parts: ContentPart[] = []
  const push = (text: string) => {
    const last = parts[parts.length - 1]
    if (last && last.type === 'text') {
      parts[parts.length - 1] = { type: 'text', text: last.text + text }
    } else {
      parts.push({ type: 'text', text })
    }
  }

  push('<comms>\n')
  for (const entry of entries) {
    const from = entry.kind === 'user' ? 'user' : entry.from
    const at = formatTimestamp(entry.timestamp, timezone)
    push(`<message from="${from}" at="${at}">\n`)
    push(entry.text)
    const attachments = entry.attachments ?? []
    if (attachments.length > 0) {
      push('\n<attachments>')
      for (const attachment of attachments) {
        if (attachment.kind === 'image') {
          push('\n')
          parts.push({ type: 'image', base64: attachment.base64, mediaType: attachment.mediaType, width: attachment.width, height: attachment.height })
        } else {
          push(`\n<artifact id="${attachment.id}">${attachment.content}</artifact>`)
        }
      }
      push('\n</attachments>')
    }
    push('\n</message>\n')
  }
  push('</comms>')
  return parts
}

export function formatSystemInbox(entries: readonly SystemEntry[]): ContentPart[] {
  const parts: ContentPart[] = []
  const push = (text: string) => {
    const last = parts[parts.length - 1]
    if (last && last.type === 'text') {
      parts[parts.length - 1] = { type: 'text', text: last.text + text }
    } else {
      parts.push({ type: 'text', text })
    }
  }

  push('<system>\n')
  for (const entry of entries) {
    if (entry.kind === 'tool_results') {
      const rendered = formatResults(entry.toolCalls, entry.inspectResults, entry.error)
      for (const part of rendered) {
        if (part.type === 'text') push(part.text)
        else parts.push(part)
      }
      push('\n')
    } else if (entry.kind === 'reminder') {
      push(`${entry.text}\n`)
    } else if (entry.kind === 'interrupted') {
      push(`${formatInterrupted()}\n`)
    } else if (entry.kind === 'error') {
      push(`${formatError(entry.message)}\n`)
    } else if (entry.kind === 'fork_result') {
      push(`${formatTaskResult(entry.taskId, entry.role, entry.name, entry.result)}\n`)
    } else if (entry.kind === 'agent_activity') {
      push(`${formatSubagentActivity([...entry.entries])}\n`)
    } else if (entry.kind === 'autonomous_ended' || entry.kind === 'task_feedback') {
      push(`${entry.kind === 'task_feedback' ? entry.text : ''}\n`)
    } else if (entry.kind === 'observation') {
      if (entry.part.type === 'text') push(entry.part.text + '\n')
      else parts.push({ type: 'image', base64: entry.part.base64, mediaType: entry.part.mediaType as ImageMediaType, width: entry.part.width, height: entry.part.height })
    }
  }
  push('</system>')
  return parts
}

/** Per-turn reminder showing active sub-agents */
export function formatAgentsStatus(
  agents: readonly { agentId: string; type: string; status: string }[]
): string | null {
  const active = agents.filter(a => a.status === 'running' || a.status === 'idle')
  if (active.length === 0) return null
  const lines = active.map(a => `- ${a.agentId} (${a.type}): ${a.status}`)
  return `<agents_status>\n${lines.join('\n')}\n</agents_status>`
}

export function formatSubagentActivity(
  entries: Array<{ agentId: string; prose: string | null; toolsCalled: readonly string[]; artifactsWritten?: readonly string[] }>
): string {
  const grouped = new Map<string, Array<{ prose: string | null; toolsCalled: readonly string[]; artifactsWritten: readonly string[] }>>()

  for (const entry of entries) {
    const current = grouped.get(entry.agentId) ?? []
    grouped.set(entry.agentId, [...current, {
      prose: entry.prose,
      toolsCalled: entry.toolsCalled,
      artifactsWritten: entry.artifactsWritten ?? [],
    }])
  }

  const lines: string[] = ['<agents_activity>']

  for (const [agentId, turns] of grouped) {
    lines.push(`<agent id="${agentId}">`)
    for (const turn of turns) {
      const attrs: string[] = []
      if (turn.toolsCalled.length > 0) attrs.push(`tools="${turn.toolsCalled.join(', ')}"`)
      if (turn.artifactsWritten.length > 0) attrs.push(`artifacts_written="${turn.artifactsWritten.join(', ')}"`)
      const attrStr = attrs.join(' ')
      if (turn.prose === null) {
        lines.push(`  <turn ${attrStr} />`)
      } else {
        lines.push(`  <turn ${attrStr}>${turn.prose}</turn>`)
      }
    }
    lines.push('</agent>')
  }

  lines.push('</agents_activity>')
  return lines.join('\n')
}

// TODO: maybe base this on context window size or something
const CONVERSATION_CONTEXT_MAX_CHARS = 50_000

/**
 * Build a conversation context block from ConversationProjection entries.
 * These are clean user text and orchestrator prose (no thinking, no tool calls).
 * Always keeps the first user message, then fills with most recent messages up to budget.
 */
export function buildConversationSummary(entries: readonly ConversationEntry[]): string | null {
  if (entries.length === 0) return null

  const formatted = entries.map(e =>
    `<message role="${e.role}">\n${e.text}\n</message>`
  )

  // Always keep the first entry
  const first = formatted[0]
  let result: string[] = [first]
  let totalChars = first.length

  // Fill remaining budget from most recent, working backwards
  const remaining = formatted.slice(1)
  const included: string[] = []
  for (let i = remaining.length - 1; i >= 0; i--) {
    if (totalChars + remaining[i].length > CONVERSATION_CONTEXT_MAX_CHARS) break
    included.unshift(remaining[i])
    totalChars += remaining[i].length
  }

  if (included.length < remaining.length) {
    result.push('... (earlier conversation omitted) ...')
  }

  result = result.concat(included)

  const header = 'These are the messages between the user and orchestrator. The user\'s intent as expressed in these messages is your primary reference for judging whether work meets expectations.'
  return `<conversation_context>\n${header}\n\n${result.join('\n\n')}\n</conversation_context>`
}