/**
 * Tools Index
 *
 * Builds RegisteredTool map from agent definitions for xml-act runtime.
 * The agent definition's tool set is the single source of truth
 * for what tools each agent gets.
 */

import type { Tool } from '@magnitudedev/tools'
import type { XmlTagBinding, RegisteredTool } from '@magnitudedev/xml-act'
import { Effect, type Layer } from 'effect'
import type { AgentDefinition, ToolSet } from '@magnitudedev/agent-definition'

// =============================================================================
// Build registered tools from agent definition
// =============================================================================

/**
 * Derive a Map<tagName, RegisteredTool> from an AgentDefinition.
 */
export function buildRegisteredTools(
  agentDef: AgentDefinition<ToolSet, unknown>,
  layers: Layer.Layer<never>,
): Map<string, RegisteredTool> {
  const tools = new Map<string, RegisteredTool>()

  for (const [defKey, tool] of Object.entries(agentDef.tools)) {
    if (!tool) continue
    const t = tool as Tool.Any

    // Read xmlInput binding
    const xmlInput = t.bindings?.xmlInput
    if (!xmlInput) continue

    // Extract binding (strip the 'type' field to get XmlTagBinding)
    const { type: _, ...binding } = xmlInput as { type: string } & XmlTagBinding

    // Derive tag name
    const tagName = defaultXmlTagName(t)

    tools.set(tagName, {
      tool: t,
      tagName,
      groupName: t.group ?? 'default',
      binding: binding as XmlTagBinding,
      meta: { defKey },
      layerProvider: () => Effect.succeed(layers),
    })
  }

  return tools
}

export { getBindingRegistry } from './binding-registry'

/**
 * Derive the XML tag name from a tool's group and name.
 *
 * - group 'default' or no group → bare name (e.g., 'shell', 'think')
 * - named group → group-name (e.g., 'fs-read', 'agent-create', 'browser-click')
 */
export function defaultXmlTagName(tool: Tool.Any): string {
  const group = tool.group
  if (!group || group === 'default') return tool.name
  return `${group}-${tool.name}`
}

// =============================================================================
// Re-exports
// =============================================================================

export { globalTools } from './globals'

export { shellTool } from './shell'
export { shellBgTool } from './shell-bg'
export { fsTools } from './fs'

export { agentTools } from './agent-tools'
export { webFetchTool } from './web-fetch-tool'

