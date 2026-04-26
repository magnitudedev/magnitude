/**
 * Turn result formatting for LLM context.
 *
 * Formats tool results, errors, interrupts, observed tool output,
 * and no-action nudges into the XML structures injected into conversation history.
 */

import { renderParseError, YIELD_USER } from '@magnitudedev/xml-act'

import { INSPECT_CHAR_LIMIT, INSPECT_TOKEN_LIMIT } from '../constants'
import { INTERRUPT_MESSAGE } from '../prompts/constants'
import { ONESHOT_LIVENESS_REMINDER } from '../prompts/error-states'
import { ContentPartBuilder, type ContentPart } from '../content'
import { imagePlaceholder } from './render'
import type {
  MessageAckResultItem,
  StructuralParseErrorResultItem,
  ToolErrorResultItem,
  ToolParseErrorResultItem,
  TurnResultItem,
} from './types'

function formatMessageAck(item: MessageAckResultItem): string {
  return `\n<message-sent to="${item.destination}" chars="${item.chars}"/>`
}

function formatToolError(item: ToolErrorResultItem): string {
  if (item.status === 'interrupted') {
    return `\n<tool name="${item.tagName}"><error>Interrupted</error></tool>`
  }

  const message = item.message ?? 'Unknown error'
  return `\n<tool name="${item.tagName}"><error>${message}</error></tool>`
}

function formatParseError(item: ToolParseErrorResultItem | StructuralParseErrorResultItem): string {
  return `\n${renderParseError(item.event, item.rawResponse ?? '')}`
}

/** Format ordered turn results for LLM context */
export function formatResults(items: readonly TurnResultItem[], supportsVision: boolean = true): ContentPart[] {
  const builder = new ContentPartBuilder()

  for (const item of items) {
    if (item.kind === 'tool_error') {
      builder.pushText(formatToolError(item))
      continue
    }

    if (item.kind === 'tool_parse_error' || item.kind === 'structural_parse_error') {
      builder.pushText(formatParseError(item))
      continue
    }

    if (item.kind === 'tool_observation') {
      const textChars = item.content.reduce((sum, part) => sum + (part.type === 'text' ? part.text.length : 0), 0)
      if (textChars > INSPECT_CHAR_LIMIT) {
        const approxTokens = Math.ceil(textChars / 4)
        const observeAttr = item.query ? ` observe="${item.query}"` : ''
        builder.pushText(`\n<${item.tagName}${observeAttr}>Output too large (~${approxTokens} tokens, limit is ${INSPECT_TOKEN_LIMIT}). Retry with a narrower observe query.</${item.tagName}>`)
        for (const part of item.content) {
          if (part.type === 'image') {
            if (supportsVision) builder.pushPart(part)
            else builder.pushText(imagePlaceholder({ mediaType: part.mediaType, width: part.width, height: part.height }))
          }
        }
        continue
      }

      const observeAttr = item.query ? ` observe="${item.query}"` : ''
      builder.pushText(`\n<${item.tagName}${observeAttr}>`)
      for (const part of item.content) {
        if (part.type === 'text') builder.pushText(part.text)
        else if (supportsVision) builder.pushPart(part)
        else builder.pushText(imagePlaceholder({ mediaType: part.mediaType, width: part.width, height: part.height }))
      }
      builder.pushText(`</${item.tagName}>`)
      continue
    }

    if (item.kind === 'no_tools_or_messages') {
      builder.pushText('\n(no tools or messages were used this turn)')
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
  return `<noop>No actions were taken. Use ${YIELD_USER} if you have nothing more to do.</noop>`
}

/** Oneshot liveness reminder rendered as result feedback */
export function formatOneshotLiveness(): string {
  return `<error>${ONESHOT_LIVENESS_REMINDER}</error>`
}

/** Yield worker retrigger reminder — lead yielded to workers but none are active */
export function formatYieldWorkerRetrigger(): string {
  return `<error>You yielded to workers with <|yield:worker|> but no workers are currently active. Check your task assignments or continue working.</error>`
}
