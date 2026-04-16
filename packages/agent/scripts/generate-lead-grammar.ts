/**
 * Generate the GBNF grammar for the lead agent based on real tool definitions.
 * 
 * Usage: bun run packages/agent/scripts/generate-lead-grammar.ts
 */

import { generateGrammar, type GrammarToolDef } from '@magnitudedev/xml-act'
import type { AgentCatalogEntry } from '../src/catalog'
import { leadTools } from '../src/agents/lead-shared'

const defs: GrammarToolDef[] = []
for (const defKey of leadTools.keys) {
  const entry = (leadTools.entries as Record<string, AgentCatalogEntry>)[defKey]
  const binding = entry.binding.toXmlTagBinding()
  defs.push({
    tagName: binding.tag,
    binding,
    inputSchema: entry.tool.inputSchema,
  })
}

const grammar = generateGrammar(defs)
console.log(grammar)
