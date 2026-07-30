import type { TurnFeedback } from '../window/types'

/**
 * Render TurnFeedback items to plain text.
 * Shared between window-to-prompt (for UserMessage parts) and compaction (for token estimation).
 */
export function renderFeedbackText(feedback: readonly TurnFeedback[]): string {
  const lines: string[] = []
  for (const fb of feedback) {
    switch (fb.kind) {
      case 'message_ack':
        lines.push(`<message-sent to="${fb.destination}" chars="${fb.chars}"/>`)
        break
      case 'error':
        lines.push(`<error>${fb.message}</error>`)
        break
      case 'overthinking':
        lines.push(`<overthinking>${fb.message}</overthinking>`)
        break
      case 'tool_limit_reached':
        lines.push(`<tool_limit_reached>\nAt most ${fb.limit} tool calls are allowed per turn.\n</tool_limit_reached>`)
        break
      case 'interrupted':
        lines.push('<interrupted>The user pressed ESC and has interrupted your turn.</interrupted>')
        break

    }
  }
  return lines.join('\n')
}
