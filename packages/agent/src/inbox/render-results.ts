/**
 * Turn result formatting for LLM context.
 *
 * Formats tool results, errors, interrupts, observed tool output,
 * and no-action nudges into the XML structures injected into conversation history.
 */


import type { TurnToolCall, ObservedResult } from '../events'
import { TURN_CONTROL_IDLE } from '@magnitudedev/xml-act'

import { INSPECT_CHAR_LIMIT, INSPECT_TOKEN_LIMIT } from '../constants'
import { INTERRUPT_MESSAGE } from '../prompts/constants'
import { ONESHOT_LIVENESS_REMINDER } from '../prompts/error-states'
import { ContentPartBuilder, type ContentPart } from '../content'

/** Format tool results for LLM context */
export function formatResults(toolCalls: readonly TurnToolCall[], observedResults: readonly ObservedResult[]): ContentPart[] {
  const builder = new ContentPartBuilder()

  for (const tc of toolCalls) {
    if (tc.result.status === 'interrupted') {
      builder.pushText(`\n<tool name="${tc.toolKey}"><error>Interrupted</error></tool>`)
    } else if (tc.result.status !== 'success') {
      builder.pushText(`\n<tool name="${tc.toolKey}"><error>${tc.result.message}</error></tool>`)
    }
  }

  for (const observed of observedResults) {
    const textChars = observed.content.reduce((sum, part) => sum + (part.type === 'text' ? part.text.length : 0), 0)
    if (textChars > INSPECT_CHAR_LIMIT) {
      const approxTokens = Math.ceil(textChars / 4)
      builder.pushText(`\n<${observed.tagName} observe="${observed.query}">Output too large (~${approxTokens} tokens, limit is ${INSPECT_TOKEN_LIMIT}). Retry with a narrower observe query.</${observed.tagName}>`)
      for (const part of observed.content) {
        if (part.type === 'image') builder.pushPart(part)
      }
      continue
    }

    builder.pushText(`\n<${observed.tagName} observe="${observed.query}">`)
    for (const part of observed.content) {
      if (part.type === 'text') builder.pushText(part.text)
      else builder.pushPart(part)
    }
    builder.pushText(`</${observed.tagName}>`)
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
