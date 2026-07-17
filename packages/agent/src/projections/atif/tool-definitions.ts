/**
 * Convert Magnitude toolkits to ATIF `agent.tool_definitions` format.
 *
 * ATIF expects OpenAI function-calling schema:
 *   { type: "function", function: { name, description, parameters } }
 *
 * Magnitude tools use Effect Schema for inputSchema. We convert through the
 * native tool-parameter boundary function.
 */

import { Schema } from 'effect'
import { materializeAgentToolkit } from '../../tools/toolkits'
import type { Toolkit } from '@magnitudedev/harness'
import { makeNativeToolParametersJsonSchema, type JsonValue } from '@magnitudedev/ai'
import type { JsonSchemaObject } from '@magnitudedev/utils/schema'

interface AtifFunctionDefinition {
  readonly [key: string]: JsonValue
  readonly name: string
  readonly description: string
  readonly parameters: JsonSchemaObject
}

export interface AtifToolDefinition {
  readonly [key: string]: JsonValue
  readonly type: 'function'
  readonly function: AtifFunctionDefinition
}

function schemaToJsonSchema(schema: Schema.Schema.AnyNoContext): JsonSchemaObject {
  return makeNativeToolParametersJsonSchema(schema)
}

export function toolDefinitionsFromToolkit(
  universe: Toolkit,
  toolKeys: readonly string[],
): readonly AtifToolDefinition[] {
  const toolkit = materializeAgentToolkit(universe, toolKeys)
  const defs: AtifToolDefinition[] = []

  for (const [key, entry] of Object.entries(toolkit.entries)) {
    // Toolkit entries are generic and type-erased at the catalog level.
    // The `.tool` property contains the HarnessTool definition with
    // name, description, and inputSchema — accessed via runtime duck-typing.
    const tool = (entry as { tool?: { name?: string; description?: string; inputSchema?: Schema.Schema.AnyNoContext } }).tool
    if (!tool) continue

    const parameters = tool.inputSchema
      ? schemaToJsonSchema(tool.inputSchema)
      : { type: 'object' as const, required: [], properties: {}, additionalProperties: false }

    defs.push({
      type: 'function',
      function: {
        name: tool.name ?? key,
        description: tool.description ?? '',
        parameters,
      },
    })
  }

  return defs
}
