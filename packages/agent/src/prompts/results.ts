/**
 * Turn result formatting for LLM context.
 *
 * Formats tool results, errors, interrupts, observed tool output,
 * and no-action nudges into the XML structures injected into conversation history.
 */

import type { TurnToolCall, ObservedResult } from '../events'
import { TURN_CONTROL_NEXT, TURN_CONTROL_YIELD } from '@magnitudedev/xml-act'
import { INSPECT_CHAR_LIMIT, INSPECT_TOKEN_LIMIT } from '../constants'
import { INTERRUPT_MESSAGE } from './constants'
import { type ContentPart } from '../content'

/** Format tool results for LLM context */
export function formatResults(toolCalls: readonly TurnToolCall[], observedResults: readonly ObservedResult[], error?: string): ContentPart[] {
  const parts: ContentPart[] = []
  let textBuffer = '<results>'

  const pushText = (text: string) => {
    textBuffer += text
  }
  const flushText = () => {
    if (!textBuffer) return
    const last = parts[parts.length - 1]
    if (last?.type === 'text') parts[parts.length - 1] = { type: 'text', text: last.text + textBuffer }
    else parts.push({ type: 'text', text: textBuffer })
    textBuffer = ''
  }

  for (const tc of toolCalls) {
    if (tc.result.status === 'interrupted') {
      pushText(`\n<tool name="${tc.toolKey}"><error>Interrupted</error></tool>`)
    } else if (tc.result.status !== 'success') {
      pushText(`\n<tool name="${tc.toolKey}"><error>${tc.result.message}</error></tool>`)
    }
  }

  for (const observed of observedResults) {
    const textChars = observed.content.reduce((sum, part) => sum + (part.type === 'text' ? part.text.length : 0), 0)
    if (textChars > INSPECT_CHAR_LIMIT) {
      const approxTokens = Math.ceil(textChars / 4)
      pushText(`\n<${observed.tagName} observe="${observed.query}">Output too large (~${approxTokens} tokens, limit is ${INSPECT_TOKEN_LIMIT}). Retry with a narrower observe query.</${observed.tagName}>`)
      for (const part of observed.content) {
        if (part.type === 'image') {
          flushText()
          parts.push(part)
        }
      }
      continue
    }

    pushText(`\n<${observed.tagName} observe="${observed.query}">`)
    for (const part of observed.content) {
      if (part.type === 'text') pushText(part.text)
      else {
        flushText()
        parts.push(part)
      }
    }
    pushText(`</${observed.tagName}>`)
  }

  if (error) {
    pushText(`\n<error>${error}</error>`)
  }

  pushText('\n</results>')
  flushText()
  return parts
}

/** Wrap interrupt message */
export function formatInterrupted(): string {
  return `<interrupted>\n${INTERRUPT_MESSAGE}\n</interrupted>`
}

/** Wrap error message */
export function formatError(message: string): string {
  return `<error>${message}</error>`
}

/** Noop turn — agent continued without taking any actions */
export function formatNoop(): string {
  return `<noop>No actions were taken. Use <${TURN_CONTROL_YIELD}/> if you have nothing more to do, instead of <${TURN_CONTROL_NEXT}/>.</noop>`
}