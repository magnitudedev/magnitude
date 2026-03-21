/**
 * Web Search Tool
 *
 * Calls Anthropic API directly with web search enabled,
 * then optionally parses structured output via BAML's schema-aligned parser.
 */

import { Effect } from 'effect'
import { Schema } from '@effect/schema'
import { defineTool, ToolErrorSchema } from '@magnitudedev/tools'
import { defineXmlBinding } from '@magnitudedev/xml-act'
import { schemaAlignedParse, outputFormatString } from '@magnitudedev/llm-core'
import { webSearch } from './web-search'

const WebSearchErrorSchema = ToolErrorSchema('WebSearchError', {})

// =============================================================================
// Tool Definition
// =============================================================================

export const webSearchTool = defineTool({
  name: 'web-search',
  group: 'default',
  description: 'Search the web and optionally extract structured data',

  inputSchema: Schema.Struct({
    query: Schema.String,
    // User-provided schema values are intentionally dynamic JSON-like payloads.
    schema: Schema.optional(Schema.Record({ key: Schema.String, value: Schema.Unknown }))
  }),

  outputSchema: Schema.Struct({
    text: Schema.String,
    sources: Schema.Array(Schema.Struct({ title: Schema.String, url: Schema.String })),
    // Parsed data shape depends on caller-provided schema, so this remains intentionally unknown.
    data: Schema.optional(Schema.Unknown),
  }),
  errorSchema: WebSearchErrorSchema,

  execute: ({ query, schema }, _ctx) =>
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
  label: (input) => input.query ? `Searching: ${input.query.slice(0, 50)}` : 'Searching…',
})

export const webSearchXmlBinding = defineXmlBinding(webSearchTool, {
  input: { body: 'query' },
  output: {
    body: 'text',
    children: [{
      field: 'sources',
      tag: 'source',
      attributes: [{ field: 'title', attr: 'title' }, { field: 'url', attr: 'url' }],
    }],
  },
} as const)
