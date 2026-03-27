/**
 * Tools Index
 *
 * Builds RegisteredTool map from agent definitions for xml-act runtime.
 * The agent definition's tool set is the single source of truth
 * for what tools each agent gets.
 */

import type { RegisteredTool } from '@magnitudedev/xml-act'
import { Effect, type Layer } from 'effect'
import type { RoleDefinition } from '@magnitudedev/roles'
import type { AgentCatalogEntry } from '../catalog'
import type { XmlBinding } from '@magnitudedev/tools'

// =============================================================================
// Build registered tools from agent definition
// =============================================================================

/**
 * Derive a Map<tagName, RegisteredTool> from an RoleDefinition.
 */
export function buildRegisteredTools(
  agentDef: RoleDefinition,
  layers: Layer.Layer<never>,
): Map<string, RegisteredTool> {
  const tools = new Map<string, RegisteredTool>()

  for (const defKey of agentDef.tools.keys) {
    const entry = agentDef.tools.entries[defKey] as AgentCatalogEntry
    const tool = entry.tool

    const binding = entry.binding.toXmlTagBinding()
    const tagName = binding.tag
    const rawOutputBinding = entry.binding.toXmlOutputBinding()
    const outputBinding: XmlBinding<unknown> = { type: 'tag' as const, ...rawOutputBinding }

    tools.set(tagName, {
      tool,
      tagName,
      groupName: tool.group ?? 'default',
      binding,
      outputBinding,
      meta: { defKey },
      layerProvider: () => Effect.succeed(layers),
    })
  }

  return tools
}

export { getBindingRegistry } from './binding-registry'

// =============================================================================
// Re-exports
// =============================================================================

export { globalTools } from './globals'

export { shellTool } from './shell'
export { shellBgTool } from './shell-bg'
export { fsTools } from './fs'

export { agentTools } from './agent-tools'
export { webFetchTool } from './web-fetch-tool'
