/**
 * ToolResultFormatter — composable formatting for tool results.
 *
 * A function from ToolResultEntry → ToolResultPart[]. The default
 * implementation handles all six ToolResult variants. Compose by
 * holding a reference to the default and overriding only what you need.
 *
 * Usage:
 *   const format = createToolResultFormatter(toolkit)
 *   const parts = format(entry)
 *
 * Override:
 *   const defaultFormat = createToolResultFormatter(toolkit)
 *   const agentFormat: ToolResultFormatter = (entry) => {
 *     if (entry.result._tag === 'Success' && needsTruncation(entry.result)) {
 *       return formatTruncatedSuccess(entry)
 *     }
 *     return defaultFormat(entry)
 *   }
 */

import type { ToolResultPart } from '@magnitudedev/ai'
import type { ToolResultEntry, ToolResult } from '../events'
import type { Toolkit } from '../tool/toolkit'
import type { Schema } from 'effect'
import { renderToolOutput, isImageValue, toImagePart, renderTagged } from './helpers'
import { renderSchemaParams } from '@magnitudedev/utils/schema'

// ---------------------------------------------------------------------------
// Type
// ---------------------------------------------------------------------------

export type ToolResultFormatter = (entry: ToolResultEntry) => readonly ToolResultPart[]

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

export function createToolResultFormatter(toolkit: Toolkit): ToolResultFormatter {
  const schemaLookup = new Map<string, Schema.Schema.AnyNoContext>()
  for (const key of toolkit.keys) {
    const entry = toolkit.entries[key]
    const definition = entry.tool.definition
    schemaLookup.set(definition.name, definition.inputSchema)
  }

  return (entry: ToolResultEntry): readonly ToolResultPart[] => {
    return formatResult(entry.result, entry.toolName, schemaLookup)
  }
}

// ---------------------------------------------------------------------------
// Internal
// ---------------------------------------------------------------------------

function formatResult(
  result: ToolResult,
  toolName: string,
  schemaLookup: Map<string, Schema.Schema.AnyNoContext>,
): readonly ToolResultPart[] {
  switch (result._tag) {
    case "Success": {
      if (result.output === undefined) return [{ _tag: 'TextPart', text: '(no output)' }]
      if (isImageValue(result.output)) {
        return [toImagePart(result.output)]
      }
      return renderToolOutput(result.output)
    }
    case "Error":
      return [{ _tag: 'TextPart', text: `<tool_error>${result.error.message}</tool_error>` }]
    case "Denied":
      return renderTagged('denied', result.denial)
    case "Interrupted":
      return [{ _tag: 'TextPart', text: '<tool_interrupted/>' }]
    case "InputRejected":
      return formatInputRejected(result, toolName, schemaLookup)
  }
}



function formatInputRejected(
  result: Extract<ToolResult, { _tag: "InputRejected" }>,
  toolName: string,
  schemaLookup: Map<string, Schema.Schema.AnyNoContext>,
): readonly ToolResultPart[] {
  const lines: string[] = [
    `<input_rejected>`,
    `Tool input was rejected.`,
    ``,
  ]
  if (result.issue.path.length > 0) {
    lines.push(`Parameter: ${result.issue.path.join('.')}`)
  }
  lines.push(`Problem: ${result.issue.message}`)
  lines.push(``)

  const schema = schemaLookup.get(toolName)
  if (schema) {
    try {
      lines.push(renderSchemaParams(schema))
    } catch {
      lines.push(`(Parameter schema unavailable)`)
    }
  } else {
    lines.push(`(Parameter schema unavailable)`)
  }

  lines.push(``)
  lines.push(`Received:`)

  const text = lines.join('\n')
  const receivedParts = renderToolOutput(result.partialInput)
  const parts: ToolResultPart[] = [{ _tag: 'TextPart', text }]
  for (const p of receivedParts) {
    parts.push(p)
  }
  parts.push({ _tag: 'TextPart', text: '\n</input_rejected>' })
  return parts
}
