/**
 * Turn result formatting for LLM context.
 *
 * Formats tool results, errors, interrupts, inspect block results,
 * and no-action nudges into the XML structures injected into conversation history.
 */

import type { TurnToolCall, InspectResult } from '../events'
import { INSPECT_CHAR_LIMIT, INSPECT_TOKEN_LIMIT } from '../constants'
import { INTERRUPT_MESSAGE } from './constants'
import { type ContentPart } from '../content'

/** Format tool results for LLM context */
export function formatResults(toolCalls: readonly TurnToolCall[], inspectResults: readonly InspectResult[], error?: string): ContentPart[] {
  const lines: string[] = ['<results>']

  for (const tc of toolCalls) {
    if (tc.result.status === 'interrupted') {
      lines.push(`<tool name="${tc.toolKey}"><error>Interrupted</error></tool>`)
    } else if (tc.result.status !== 'success') {
      lines.push(`<tool name="${tc.toolKey}"><error>${tc.result.message}</error></tool>`)
    }
  }

  if (inspectResults.length > 0) {
    lines.push('<inspect>')
    for (const ir of inspectResults) {
      if (ir.status === 'invalid_ref') {
        lines.push(`<ref tool="${ir.toolRef}">Invalid ref — "${ir.toolRef}" does not match any tool result from this response. Refs are only valid within the same response they were produced in.</ref>`)
      } else if (ir.content.length > INSPECT_CHAR_LIMIT) {
        const approxTokens = Math.ceil(ir.content.length / 4)
        lines.push(`<ref tool="${ir.toolRef}">Output too large (~${approxTokens} tokens, limit is ${INSPECT_TOKEN_LIMIT}). Use a query attribute to select a subset: &lt;ref tool="${ir.toolRef}" query="xpath expression" /&gt;</ref>`)
      } else {
        lines.push(`<ref tool="${ir.toolRef}">${ir.content}</ref>`)
      }
    }
    lines.push('</inspect>')
  }

  if (error) {
    lines.push(`<error>${error}</error>`)
  }

  lines.push('</results>')
  return [{ type: 'text' as const, text: lines.join('\n') }]
}

/** Wrap interrupt message */
export function formatInterrupted(): string {
  return `<interrupted>\n${INTERRUPT_MESSAGE}\n</interrupted>`
}

/** Wrap error message */
export function formatError(message: string): string {
  return `<error>${message}</error>`
}