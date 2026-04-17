/**
 * Tool Registry Builders
 *
 * Builds RegisteredTool map and GBNF grammar from agent definitions.
 */

import type { RegisteredTool } from '@magnitudedev/xml-act'
import { generateGrammar, type GrammarToolDef } from '@magnitudedev/xml-act'
import { Effect, type Layer } from 'effect'
import type { RoleDefinition } from '@magnitudedev/roles'
import type { AgentCatalogEntry } from '../catalog'
import type { XmlBinding } from '@magnitudedev/tools'

/**
 * Derive a Map<tagName, RegisteredTool> from an RoleDefinition.
 */
export function buildRegisteredTools(
  agentDef: RoleDefinition,
  layers: Layer.Layer<never>,
  excludeTools?: Set<string>,
): Map<string, RegisteredTool> {
  const tools = new Map<string, RegisteredTool>()

  for (const defKey of agentDef.tools.keys) {
    if (excludeTools?.has(defKey)) continue
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

/**
 * Generate GBNF grammar from agent definition's tool catalog.
 * Used JIT before model invocation for constrained generation.
 */
export function generateToolGrammar(agentDef: RoleDefinition, excludeTools?: Set<string>): string {
  const defs: GrammarToolDef[] = []
  for (const defKey of agentDef.tools.keys) {
    if (excludeTools?.has(defKey)) continue
    const entry = agentDef.tools.entries[defKey] as AgentCatalogEntry
    const binding = entry.binding.toXmlTagBinding()
    defs.push({
      tagName: binding.tag,
      binding,
      inputSchema: entry.tool.inputSchema,
    })
  }
  return generateGrammar(defs)
}
