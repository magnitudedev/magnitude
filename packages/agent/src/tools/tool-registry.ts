/**
 * Tool Registry Builders
 *
 * Builds RegisteredTool map and GBNF grammar from agent definitions.
 */

import type { RegisteredTool } from '@magnitudedev/xml-act'
import { generateGrammar, type GrammarToolDef } from '@magnitudedev/xml-act'
import { Effect, type Layer } from 'effect'
import type { AgentCatalogEntry } from '../catalog'
import type { XmlBinding } from '@magnitudedev/tools'
import type { ResolvedToolSet } from './resolved-toolset'

/**
 * Derive a Map<tagName, RegisteredTool> from a ResolvedToolSet.
 */
export function buildRegisteredTools(
  toolSet: ResolvedToolSet,
  layers: Layer.Layer<never>,
): Map<string, RegisteredTool> {
  const tools = new Map<string, RegisteredTool>()
  const agentDef = toolSet.agentDef

  for (const defKey of toolSet.availableKeys) {
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
 * Generate GBNF grammar from a ResolvedToolSet.
 * Used JIT before model invocation for constrained generation.
 */
export function generateToolGrammar(toolSet: ResolvedToolSet): string {
  const defs: GrammarToolDef[] = []
  const agentDef = toolSet.agentDef

  for (const defKey of toolSet.availableKeys) {
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
