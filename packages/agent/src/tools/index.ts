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
import type { RoleDefinition, ToolSet } from '@magnitudedev/roles'
import { fsXmlBindings } from './fs'
import { shellXmlBinding } from './shell'
import { shellBgXmlBinding } from './shell-bg'
import { globalXmlBindings } from './globals'
import { agentCreateXmlBinding, agentKillXmlBinding } from './agent-tools'
import { browserXmlBindings } from './browser-tools'

// =============================================================================
// Build registered tools from agent definition
// =============================================================================

/**
 * Erased binding result for heterogeneous map storage.
 * Uses structural types at the erasure boundary — concrete generics are
 * verified at each defineXmlBinding call site.
 */
interface ErasedXmlBindingResult {
  readonly tool: { name: string; group?: string }
  readonly config: { readonly group?: string; readonly [key: string]: unknown }
  toXmlTagBinding(): XmlTagBinding
  toXmlOutputBinding(): object
}
type ToolTagIdentity = { name: string; group?: string }

function toRegisteredBinding(binding: { tool: { name: string; group?: string }; config: { readonly group?: string; readonly [key: string]: unknown }; toXmlTagBinding(): XmlTagBinding; toXmlOutputBinding(): object }): ErasedXmlBindingResult {
  // Existential erasure at composition root: typed binding verified at definition site,
  // erased here for heterogeneous map storage.
  return {
    tool: binding.tool,
    config: binding.config,
    toXmlTagBinding: () => binding.toXmlTagBinding(),
    toXmlOutputBinding: () => binding.toXmlOutputBinding(),
  }
}

function getBindingGroup(config: { readonly group?: string; readonly [key: string]: unknown }): string | undefined {
  return config.group
}

/**
 * Derive a Map<tagName, RegisteredTool> from an RoleDefinition.
 */
export function buildRegisteredTools(
  agentDef: RoleDefinition<ToolSet, string, unknown>,
  layers: Layer.Layer<never>,
): Map<string, RegisteredTool> {
  const tools = new Map<string, RegisteredTool>()
  const xmlBindingMap = getXmlBindingMap()

  for (const [defKey, tool] of Object.entries(agentDef.tools)) {
    if (!tool) continue
    if (!('execute' in tool) || typeof tool.execute !== 'function') continue

    const tagName = defaultXmlTagName(tool)
    const xmlBinding = xmlBindingMap.get(tagName)
    if (!xmlBinding) continue

    const binding = xmlBinding.toXmlTagBinding()
    const outputBinding = xmlBinding.toXmlOutputBinding()

    tools.set(tagName, {
      tool: {
        ...tool,
        bindings: {
          xmlInput: { type: 'tag', ...binding },
          xmlOutput: { type: 'tag', ...outputBinding },
        },
      },
      tagName,
      groupName: tool.group ?? 'default',
      binding,
      meta: { defKey },
      layerProvider: () => Effect.succeed(layers),
    })
  }

  return tools
}

export { getBindingRegistry } from './binding-registry'

/** Build a map of XML tag name → XmlBindingResult for all known tool bindings. */
export function getXmlBindingMap(): Map<string, ErasedXmlBindingResult> {
  const allBindings = [
    ...fsXmlBindings,
    shellXmlBinding,
    shellBgXmlBinding,
    ...globalXmlBindings,
    agentCreateXmlBinding,
    agentKillXmlBinding,
    ...browserXmlBindings,
  ]

  const map = new Map<string, ErasedXmlBindingResult>()
  for (const binding of allBindings) {
    const tool = binding.tool
    const group = getBindingGroup(binding.config) ?? tool.group
    const tagName = group && group !== 'default' ? `${group}-${tool.name}` : tool.name
    map.set(tagName, toRegisteredBinding(binding))
  }

  return map
}

/**
 * Derive the XML tag name from a tool's group and name.
 *
 * - group 'default' or no group → bare name (e.g., 'shell', 'think')
 * - named group → group-name (e.g., 'fs-read', 'agent-create', 'browser-click')
 */
export function defaultXmlTagName(tool: ToolTagIdentity): string {
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
