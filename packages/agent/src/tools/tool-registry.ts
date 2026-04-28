/**
 * Tool Registry Builders
 *
 * Builds RegisteredTool map from agent definitions.
 * Grammar generation (xml-act only) has been removed — native paradigm uses no GBNF.
 */

import type { RegisteredTool } from '@magnitudedev/turn-engine'
import { Effect, type Layer } from 'effect'
import type { AgentCatalogEntry } from '../catalog'
import type { ResolvedToolSet } from './resolved-toolset'

/**
 * Derive a `Map<toolName, RegisteredTool<R>>` from a ResolvedToolSet.
 */
export function buildRegisteredTools<R = never>(
  toolSet: ResolvedToolSet,
  layers: Layer.Layer<R, never, never>,
): Map<string, RegisteredTool<R>> {
  const tools = new Map<string, RegisteredTool<R>>()
  const agentDef = toolSet.agentDef

  for (const defKey of toolSet.availableKeys) {
    const entry = agentDef.tools.entries[defKey] as AgentCatalogEntry
    const tool = entry.tool
    const toolName = tool.name

    tools.set(toolName, {
      tool,
      toolName,
      groupName: tool.group ?? 'default',
      meta: { defKey },
      layerProvider: () => Effect.succeed(layers),
    })
  }

  return tools
}
