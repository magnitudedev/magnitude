/**
 * ToolRegistry — Effect service.
 *
 * Per-turn, per-fork store of available tools.
 * Constructed by makeToolRegistryLive from a RegisteredTool array.
 */

import { Context, Effect, Layer, Schema } from 'effect'
import type { RegisteredTool } from '@magnitudedev/turn-engine'
import type { ToolDef } from '@magnitudedev/codecs'

// =============================================================================
// Errors
// =============================================================================

export class ToolNotFound extends Schema.TaggedError<ToolNotFound>()(
  'ToolNotFound',
  { toolName: Schema.String },
) {}

// =============================================================================
// Service shape
// =============================================================================

export interface ToolRegistryShape {
  /** Look up a tool by name. Fails with ToolNotFound if not registered. */
  readonly lookup: (toolName: string) => Effect.Effect<RegisteredTool, ToolNotFound>
  /**
   * Return all registered tools as ToolDef[] for the codec to encode
   * into the wire request's `tools: [...]` field.
   */
  readonly toolDefs: () => Effect.Effect<readonly ToolDef[]>
}

export class ToolRegistry extends Context.Tag('ToolRegistry')<
  ToolRegistry,
  ToolRegistryShape
>() {}

// =============================================================================
// Live layer factory
// =============================================================================

/**
 * Build a ToolRegistry layer from a flat list of RegisteredTools.
 * Called once per turn by Cortex.
 */
export function makeToolRegistryLive(
  tools: readonly RegisteredTool[],
): Layer.Layer<ToolRegistry> {
  return Layer.succeed(ToolRegistry, {
    lookup: (toolName) => {
      const found = tools.find(t => t.toolName === toolName)
      return found
        ? Effect.succeed(found)
        : Effect.fail(new ToolNotFound({ toolName }))
    },
    toolDefs: () =>
      Effect.succeed(
        tools.map(rt => ({
          name:        rt.toolName,
          description: rt.tool.description ?? '',
          parameters:  deriveJsonSchema(rt.tool.inputSchema),
        }) satisfies ToolDef),
      ),
  })
}

// ---------------------------------------------------------------------------
// JSON Schema derivation
// ---------------------------------------------------------------------------
// Build the JSON Schema describing each tool's input. We import both Schema
// and JSONSchema from `@effect/schema` — that is the package whose Schema
// type ToolDefinition.inputSchema uses, so types align without casts.
//
// Note: the codebase contains a transitional mix of `@effect/schema` (legacy)
// and `effect/Schema` (unified). Until the tools package finishes migrating,
// stay aligned with whichever side ToolDefinition.inputSchema comes from.
import { Schema as EffectSchema } from '@effect/schema'
import * as JSONSchema from '@effect/schema/JSONSchema'

function deriveJsonSchema(schema: EffectSchema.Schema.Any): unknown {
  try {
    return JSONSchema.make(schema)
  } catch {
    // Fallback: open object schema. JSONSchema.make can throw on schemas
    // it cannot represent (e.g. classes without a JSON projection).
    return { type: 'object' }
  }
}
