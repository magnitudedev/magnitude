/**
 * Fork context builders.
 *
 * Build the context message injected into forked agents (clone or spawn).
 */

import type { SessionContext } from '../events'
import { buildProjectContext } from './session-context'

// Simple JSON Schema type — sufficient for output format instructions
export interface JsonSchema {
  type?: string
  description?: string
  properties?: Record<string, JsonSchema>
  required?: string[]
  items?: JsonSchema
  enum?: unknown[]
  [key: string]: unknown
}

/**
 * Build a human-readable format instruction string from a JSON Schema.
 * This replaces the BAML `outputFormatString` utility.
 */
function outputFormatString(schema: JsonSchema): string {
  const lines: string[] = []

  function describe(s: JsonSchema, indent: string): void {
    if (s.description) {
      lines.push(`${indent}Description: ${s.description}`)
    }

    if (s.enum && Array.isArray(s.enum)) {
      lines.push(`${indent}Must be one of: ${s.enum.map(v => JSON.stringify(v)).join(', ')}`)
      return
    }

    const type = s.type ?? 'any'
    switch (type) {
      case 'object': {
        if (s.properties && typeof s.properties === 'object') {
          const required = new Set(Array.isArray(s.required) ? s.required : [])
          for (const [key, prop] of Object.entries(s.properties)) {
            const isRequired = required.has(key)
            lines.push(`${indent}- "${key}"${isRequired ? '' : ' (optional)'}:`)
            describe(prop as JsonSchema, indent + '  ')
          }
        } else {
          lines.push(`${indent}A JSON object.`)
        }
        break
      }
      case 'array': {
        lines.push(`${indent}A JSON array.`)
        if (s.items) {
          lines.push(`${indent}Each item:`)
          describe(s.items, indent + '  ')
        }
        break
      }
      case 'string':
        lines.push(`${indent}A string value.`)
        break
      case 'number':
      case 'integer':
        lines.push(`${indent}A numeric value.`)
        break
      case 'boolean':
        lines.push(`${indent}A boolean value.`)
        break
      default:
        lines.push(`${indent}Any value.`)
    }
  }

  describe(schema, '')
  return lines.join('\n')
}

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
  parts.push('You are a background agent cloned from a coordinator thread.')
  parts.push('You inherited your coordinator\'s full conversation context.')
  parts.push('Focus solely on completing the task above.')
  parts.push('Default to coordinator-facing communication unless the user messages you directly.')
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

  parts.push('<session-start>')

  // Project context (if available)
  if (sessionContext) {
    parts.push('<project-context>')
    parts.push(buildProjectContext(sessionContext))
    parts.push('</project-context>')
    parts.push('')
  }

  // Task description
  parts.push(taskDescription)

  if (outputSchema) {
    parts.push('')
    parts.push('<output_format>')
    parts.push(outputFormatString(outputSchema))
    parts.push('</output_format>')
  }

  parts.push('</session-start>')

  return parts.join('\n')
}
