/**
 * Global Tools
 *
 * Built-in tools available in the sandbox:
 * - think(thought) - Internal reasoning (not shown)
 * - webSearch(query, schema?) - Search the web with optional structured output
 */

import { Effect } from 'effect'
import { Schema } from '@effect/schema'
import { createTool } from '@magnitudedev/tools'
import { webSearchTool } from './web-search-tool'
import { webFetchTool } from './web-fetch-tool'
import { skillTool } from './skill'

// =============================================================================
// think() - Internal reasoning
// =============================================================================

export const thinkTool = createTool({
  name: 'think',
  group: 'default',
  description: 'Record internal reasoning (not shown to user)',
  inputSchema: Schema.Struct({ thought: Schema.String }),
  outputSchema: Schema.String,
  argMapping: ['thought'],
  bindings: {
    openai: { type: 'native', mechanism: 'reasoning' },
    xmlInput: { type: 'tag', body: 'thought' },
    xmlOutput: { type: 'tag' as const },
  } as const,
  execute: ({ thought }) => Effect.succeed(thought),
})


// =============================================================================
// Global Tools
// =============================================================================

export const globalTools = [thinkTool, webSearchTool, webFetchTool, skillTool]



