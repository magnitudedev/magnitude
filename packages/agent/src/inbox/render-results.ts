/**
 * Turn result formatting for LLM context.
 *
 * Formats tool results, errors, interrupts, observed tool output,
 * and no-action nudges into the XML structures injected into conversation history.
 */

import { TURN_CONTROL_IDLE } from '@magnitudedev/xml-act'

import { INSPECT_CHAR_LIMIT, INSPECT_TOKEN_LIMIT } from '../constants'
import { INTERRUPT_MESSAGE } from '../prompts/constants'
import { ONESHOT_LIVENESS_REMINDER } from '../prompts/error-states'
import { ContentPartBuilder, type ContentPart } from '../content'
import type { MessageAckResultItem, TurnResultItem } from './types'

function formatMessageAck(item: MessageAckResultItem): string {
  return `\n<message-sent to="${item.destination}" chars="${item.chars}"/>`
}

/** Format ordered turn results for LLM context */
export function formatResults(items: readonly TurnResultItem[]): ContentPart[] {
  const builder = new ContentPartBuilder()

  for (const item of items) {
    if (item.kind === 'tool_error') {
      if (item.status === 'interrupted') {
        builder.pushText(`\n<tool name="${item.toolKey}"><error>Interrupted</error></tool>`)
      } else {
        builder.pushText(`\n<tool name="${item.toolKey}"><error>${item.message ?? 'Unknown error'}</error></tool>`)
      }
      continue
    }

    if (item.kind === 'tool_observation') {
      const textChars = item.content.reduce((sum, part) => sum + (part.type === 'text' ? part.text.length : 0), 0)
      if (textChars > INSPECT_CHAR_LIMIT) {
        const approxTokens = Math.ceil(textChars / 4)
        builder.pushText(`\n<${item.tagName} observe="${item.query}">Output too large (~${approxTokens} tokens, limit is ${INSPECT_TOKEN_LIMIT}). Retry with a narrower observe query.</${item.tagName}>`)
        for (const part of item.content) {
          if (part.type === 'image') builder.pushPart(part)
        }
        continue
      }

      builder.pushText(`\n<${item.tagName} observe="${item.query}">`)
      for (const part of item.content) {
        if (part.type === 'text') builder.pushText(part.text)
        else builder.pushPart(part)
      }
      builder.pushText(`</${item.tagName}>`)
      continue
    }

    builder.pushText(formatMessageAck(item))
  }

  return builder.build()
}

/** Wrap interrupt message */
export function formatInterrupted(): string {
  return `<interrupted>\n${INTERRUPT_MESSAGE}\n</interrupted>`
}

/** Wrap error message */
export function formatError(message: string): string {
  return `<error>${message}</error>`
}

/** Noop turn — agent continued without taking any task/tool operations */
export function formatNoop(): string {
  return `<noop>No actions were taken. Use ${TURN_CONTROL_IDLE} if you have nothing more to do.</noop>`
}

/** Oneshot liveness reminder rendered as result feedback */
export function formatOneshotLiveness(): string {
  return `<error>${ONESHOT_LIVENESS_REMINDER}</error>`
}
