/**
 * Fork context builders.
 *
 * Build the context message injected into forked agents (clone or spawn).
 */

import type { SessionContext } from '../events'
import { outputFormatString, type JsonSchema } from '@magnitudedev/llm-core'
import { buildProjectContext } from './session-context'

/**
 * Build context message for a cloned agent (inherited context).
 */
export function buildCloneContext(taskDescription: string, outputSchema?: JsonSchema): string {
  const parts: string[] = []

  // Task
  parts.push('<task>')
  parts.push(taskDescription)
  parts.push('</task>')
  parts.push('')

  // Instructions
  parts.push('<instructions>')
  parts.push('You are a background agent cloned from a parent thread.')
  parts.push('You inherited your parent\'s full conversation context.')
  parts.push('Focus solely on completing the task above.')
  parts.push('Do NOT communicate with the user directly.')
  if (outputSchema) {
    const formatInstructions = outputFormatString(outputSchema)
    parts.push('')
    parts.push('Format your output as follows:')
    parts.push(formatInstructions)
  }
  parts.push('Do NOT create nested agents unless absolutely necessary.')
  parts.push('</instructions>')

  return parts.join('\n')
}

/**
 * Build context message for a spawned agent (fresh context).
 */
export function buildSpawnContext(taskDescription: string, sessionContext?: SessionContext | null, outputSchema?: JsonSchema): string {
  const parts: string[] = []

  // Project context (if available)
  if (sessionContext) {
    parts.push('<project_context>')
    parts.push(buildProjectContext(sessionContext))
    parts.push('</project_context>')
    parts.push('')
  }

  // Task description (already wrapped in <orchestrator> XML)
  parts.push(taskDescription)

  if (outputSchema) {
    parts.push('')
    parts.push('<output_format>')
    parts.push(outputFormatString(outputSchema))
    parts.push('</output_format>')
  }

  return parts.join('\n')
}
