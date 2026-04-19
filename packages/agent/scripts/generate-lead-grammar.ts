/**
 * Generate the GBNF grammar for the lead agent based on real tool definitions.
 * 
 * Usage: bun run packages/agent/scripts/generate-lead-grammar.ts
 */

import { GrammarBuilder, type GrammarToolDef, LEAD_YIELD_TAGS } from '@magnitudedev/xml-act'
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

const grammar = GrammarBuilder.create(defs).withYieldTags([...LEAD_YIELD_TAGS]).build()
console.log(grammar)
