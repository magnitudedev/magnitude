/**
 * Global Tools
 *
 * Built-in tools available in the sandbox:
 * - think(thought) - Internal reasoning (not shown)
 * - webSearch(query, schema?) - Search the web with optional structured output
 */

import { Effect } from 'effect'
import { Schema } from '@effect/schema'
import { defineTool } from '@magnitudedev/tools'
import { defineXmlBinding } from '@magnitudedev/xml-act'
import { webSearchTool, webSearchXmlBinding } from './web-search-tool'
import { webFetchTool, webFetchXmlBinding } from './web-fetch-tool'
import { skillTool, skillXmlBinding } from './skill'
import { phaseSubmitTool, phaseSubmitXmlBinding } from './phase-submit'

// =============================================================================
// think() - Internal reasoning
// =============================================================================

export const thinkTool = defineTool({
  name: 'think',
  group: 'default',
  description: 'Record internal reasoning (not shown to user)',
  inputSchema: Schema.Struct({ thought: Schema.String }),
  outputSchema: Schema.String,
  execute: ({ thought }, _ctx) => Effect.succeed(thought),
  label: (_input) => 'Thinking…',
})

export const thinkXmlBinding = defineXmlBinding(thinkTool, {
  input: { body: 'thought' },
  output: {},
} as const)


// =============================================================================
// Global Tools
// =============================================================================

export { skillTool, phaseSubmitTool }

export const globalTools = [thinkTool, webSearchTool, webFetchTool, skillTool, phaseSubmitTool]

export const globalXmlBindings = [thinkXmlBinding, webSearchXmlBinding, webFetchXmlBinding, skillXmlBinding, phaseSubmitXmlBinding]



