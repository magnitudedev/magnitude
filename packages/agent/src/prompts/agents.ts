/**
 * Agent prompt helpers.
 */

import type { ConversationEntry } from '../projections/conversation'

/** Build the context prompt injected into a sub-agent's fork */
export function buildAgentContext(
  title: string,
  message: string | undefined,
  extraContext: string | null,
  taskId: string,
  taskContract?: string,
): string {
  const parts: string[] = []
  parts.push('<lead>')
  parts.push(`<title>${title}</title>`)
  parts.push(`<task_id>${taskId}</task_id>`)
  if (taskContract?.trim()) {
    parts.push('<task_contract>')
    parts.push(taskContract.trim())
    parts.push('</task_contract>')
  }
  if (message !== undefined) {
    parts.push('<message>')
    parts.push(message)
    parts.push('</message>')
  }
  if (extraContext) {
    parts.push(extraContext)
  }
  parts.push('</lead>')
  return parts.join('\n')
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

  return `<conversation_context>\n${result.join('\n\n')}\n</conversation_context>`
}
