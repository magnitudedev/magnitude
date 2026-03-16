/**
 * Web Search Tool
 *
 * Calls Anthropic API directly with web search enabled,
 * then optionally parses structured output via BAML's schema-aligned parser.
 */

import { Effect } from 'effect'
import { Schema } from '@effect/schema'
import { createTool, ToolErrorSchema } from '@magnitudedev/tools'
import { schemaAlignedParse, outputFormatString } from '@magnitudedev/llm-core'
import { webSearch } from './web-search'

// =============================================================================
// Tool Definition
// =============================================================================

const WebSearchError = ToolErrorSchema('WebSearchError', {})

export const webSearchTool = createTool({
  name: 'web-search',
  group: 'default',
  description: 'Search the web and optionally extract structured data',

  inputSchema: Schema.Struct({
    query: Schema.String,
    schema: Schema.optional(Schema.Record({ key: Schema.String, value: Schema.Unknown }))
  }),

  outputSchema: Schema.Unknown,
  errorSchema: WebSearchError,

  argMapping: ['query', 'schema'],
  bindings: {
    xmlInput: { type: 'tag', body: 'query' },
    xmlOutput: {
      type: 'tag' as const,
      body: 'text',
      children: [{
        field: 'sources',
        tag: 'source',
        attributes: [{ field: 'title', attr: 'title' }, { field: 'url', attr: 'url' }],
      }],
    },
  } as const,

  execute: ({ query, schema }) =>
    Effect.gen(function* () {
      const system = schema ? outputFormatString(schema) : undefined
      const response = yield* webSearch(query, { system }).pipe(
        Effect.mapError((e) => ({ _tag: 'WebSearchError' as const, message: e instanceof Error ? e.message : String(e) })),
      )

      const sources: { title: string; url: string }[] = []
      for (const r of response.results) {
        if (typeof r !== 'string') {
          for (const item of r.content) {
            sources.push({ title: item.title, url: item.url })
          }
        }
      }

      const result: { text: string; sources: { title: string; url: string }[]; data?: unknown } = {
        text: response.textResponse,
        sources,
      }

      if (schema) {
        result.data = schemaAlignedParse(response.textResponse, schema)
      }

      return result
    }),
})
