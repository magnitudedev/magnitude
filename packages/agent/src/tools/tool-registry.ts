/**
 * Tool Registry Builders
 *
 * Builds RegisteredTool map and GBNF grammar from agent definitions.
 */

import type { RegisteredTool, GrammarToolDef, GrammarBuildOptions } from '@magnitudedev/xml-act'
import { GrammarBuilder, deriveParameters } from '@magnitudedev/xml-act'
import { Effect, type Layer } from 'effect'
import type { AgentCatalogEntry } from '../catalog'
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

    // Tool tag name is the tool's name
    const tagName = tool.name

    tools.set(tagName, {
      tool,
      tagName,
      groupName: tool.group ?? 'default',
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
export function generateToolGrammar(
  toolSet: ResolvedToolSet,
  options: GrammarBuildOptions,
): string {
  const defs: GrammarToolDef[] = []
  const agentDef = toolSet.agentDef

  for (const defKey of toolSet.availableKeys) {
    const entry = agentDef.tools.entries[defKey] as AgentCatalogEntry
    const tool = entry.tool
    
    // Derive parameter schema from the tool's input schema
    const toolSchema = deriveParameters(tool.inputSchema.ast)
    
    // Convert to grammar tool def
    // Parameter name is the field path, type is 'scalar' or 'json'
    const parameters = [...toolSchema.parameters.values()].map(p => ({
      name: p.name,
      field: p.name, // In new format, parameter name = field path
      type: p.type === 'json_object' || p.type === 'json_array' ? 'json' as const : 'scalar' as const,
      required: p.required,
    }))
    
    defs.push({
      tagName: tool.name,
      parameters,
    })
  }
  
  return GrammarBuilder.create(defs).withOptions(options ?? {}).build()
}
