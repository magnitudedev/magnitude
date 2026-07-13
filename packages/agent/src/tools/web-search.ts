/**
 * Web Search Tool
 *
 * Searches the web via ProviderClient.
 */

import { Effect, Option, Schema } from 'effect'
import { JsonRecordSchema, JsonValueSchema, type JsonValue } from '@magnitudedev/utils/schema'
import { defineHarnessTool } from '@magnitudedev/harness'
import { ProviderClient } from '@magnitudedev/sdk'
import { ToolErrorSchema } from './errors'

const WebSearchErrorSchema = ToolErrorSchema('WebSearchError', {})

export const webSearchTool = defineHarnessTool({
  definition: {
    name: 'web_search',
    description: 'Search the web and optionally extract structured data',
    inputSchema: Schema.Struct({
      query: Schema.String.annotations({ description: 'Search query string' }),
      schema: Schema.optionalWith(JsonRecordSchema.annotations({ description: 'Optional schema for structured data extraction' }), { as: 'Option', exact: true })
    }),
    outputSchema: Schema.Struct({
      text: Schema.String,
      sources: Schema.Array(Schema.Struct({ title: Schema.String, url: Schema.String })),
      data: Schema.optionalWith(JsonValueSchema, { as: 'Option', exact: true }),
    }),
  },
  errorSchema: WebSearchErrorSchema,
  execute: ({ query, schema }, _ctx) =>
    Effect.gen(function* () {
      const client = yield* ProviderClient
      const search = Option.match(schema, {
        onNone: () => client.webSearch(query),
        onSome: (schema) => client.webSearch(query, schema as Record<string, unknown>),
      })
      const result = yield* search.pipe(
        Effect.mapError((err) => ({
          _tag: 'WebSearchError' as const,
          message: err.message,
        }))
      )
      return {
        text: result.text,
        sources: [...result.sources],
        data: result.data === undefined
          ? Option.none<JsonValue>()
          : Option.some(result.data as JsonValue),
      }
    }),
})
