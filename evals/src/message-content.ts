/**
 * Helpers for safely rendering chat message content as plain text.
 * Supports legacy text formats and multimodal structured parts.
 */

import type { ChatMessage } from './types'

/**
 * Convert message content into displayable plain text.
 */
export function messageContentToDisplayText(content: unknown): string {
  return extractText(content).trim()
}

/**
 * Convenience helper for full chat messages.
 */
export function chatMessageToDisplayText(msg: ChatMessage): string {
  return messageContentToDisplayText(msg.content)
}

function extractText(value: unknown): string {
  if (typeof value === 'string') return value
  if (value == null) return ''

  if (Array.isArray(value)) {
    const parts = value
      .map((part) => extractText(part))
      .filter((part) => part.length > 0)
    return parts.join('\n')
  }

  if (typeof value !== 'object') return ''

  const part = value as Record<string, unknown>

  // Common text-bearing fields across provider/content schemas.
  for (const key of ['text', 'content', 'input_text', 'output_text', 'value']) {
    const candidate = part[key]
    if (typeof candidate === 'string') return candidate
  }

  // Nested payloads that may contain text.
  for (const key of ['data', 'payload']) {
    const nested = extractText(part[key])
    if (nested) return nested
  }

  // If we can identify a part type but no text, keep a marker.
  if (typeof part.type === 'string') {
    return `[${part.type}]`
  }

  return ''
}
