
/**
 * Web Search Tool
 *
 * Searches the web using either the Magnitude provider API or EXA API.
 * - If Magnitude provider is active: calls https://app.magnitude.dev/api/v1/web-search
 * - If EXA_API_KEY env var is set: calls EXA API via exa-js
 * - Otherwise: fails with clear error
 */

import { Effect } from 'effect'
import { Schema } from '@effect/schema'
import { defineTool, ToolErrorSchema } from '@magnitudedev/tools'
import { defineXmlBinding } from '@magnitudedev/xml-act'
import { AmbientServiceTag, Fork } from '@magnitudedev/event-core'
import { ProviderState } from '@magnitudedev/providers'
import { ConfigAmbient } from '../ambient/config-ambient'

const { ForkContext } = Fork

const WebSearchErrorSchema = ToolErrorSchema('WebSearchError', {})

// =============================================================================
// Magnitude API Call
// =============================================================================

function callMagnitudeWebSearch(apiKey: string, query: string, schema?: Record<string, unknown>) {
  return Effect.tryPromise({
    try: async () => {
      const body: { query: string; schema?: Record<string, unknown> } = { query }
      if (schema) {
        body.schema = schema
      }

      const response = await fetch('https://app.magnitude.dev/api/v1/web-search', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
      })

      if (!response.ok) {
        const errorBody = await response.text().catch(() => '')
        throw new Error(`Magnitude web search failed: ${response.status} ${response.statusText} - ${errorBody}`)
      }

      const data = await response.json()
      return data as { text: string; sources: { title: string; url: string }[]; data: unknown | null }
    },
    catch: (e) => ({
      _tag: 'WebSearchError' as const,
      message: e instanceof Error ? e.message : String(e),
    }),
  })
}

// =============================================================================
// EXA API Call
// =============================================================================

function callExaWebSearch(apiKey: string, query: string, schema?: Record<string, unknown>) {
  return Effect.tryPromise({
    try: async () => {
      // Dynamically import exa-js to avoid requiring it as a hard dependency
      const { default: Exa } = await import('exa-js')
      const exa = new Exa(apiKey)

      // Build search options - exa-js uses `contents` with `highlights` inside
      const baseOptions = {
        type: 'auto' as const,
        numResults: 5,
        contents: { highlights: true as const },
      }

      // If schema is provided, wrap it in the DeepOutputSchema format
      const searchOptions = schema
        ? { ...baseOptions, outputSchema: { type: 'object' as const, properties: schema } }
        : baseOptions

      const result = await exa.searchAndContents(query, searchOptions)

      // Map EXA response to common format
      const sources = result.results.map((r) => ({
        title: r.title ?? r.url,
        url: r.url,
      }))

      const text = result.results
        .map((r) => {
          const title = r.title ?? 'Untitled'
          const highlights = 'highlights' in r && Array.isArray(r.highlights)
            ? r.highlights.join('\n')
            : ''
          return `## ${title}\n${highlights}`
        })
        .join('\n\n')

      const data = result.output?.content ?? null

      return { text, sources, data }
    },
    catch: (e) => ({
      _tag: 'WebSearchError' as const,
      message: e instanceof Error ? e.message : String(e),
    }),
  })
}

// =============================================================================
// Tool Definition
// =============================================================================

export const webSearchTool = defineTool({
  name: 'web-search',
  group: 'default',
  description: 'Search the web and optionally extract structured data',

  inputSchema: Schema.Struct({
    query: Schema.String,
    schema: Schema.optional(Schema.Record({ key: Schema.String, value: Schema.Unknown }))
  }),

  outputSchema: Schema.Struct({
    text: Schema.String,
    sources: Schema.Array(Schema.Struct({ title: Schema.String, url: Schema.String })),
    data: Schema.optional(Schema.Unknown),
  }),
  errorSchema: WebSearchErrorSchema,

  execute: ({ query, schema }, _ctx) =>
    Effect.gen(function* () {
      const ambientService = yield* AmbientServiceTag
      const providerState = yield* ProviderState
      const configState = ambientService.getValue(ConfigAmbient)

      // Get current slot from fork context
      const forkCtx = yield* ForkContext
      const slot = forkCtx.slot as 'lead' | 'worker'
      
      const slotConfig = configState.bySlot[slot]
      const isMagnitudeProvider = slotConfig.providerId === 'magnitude'

      if (isMagnitudeProvider) {
        // Get Magnitude API key from ProviderState for THIS slot
        const peekResult = yield* providerState.peek(slot)
        if (!peekResult || !peekResult.auth) {
          return yield* Effect.fail({
            _tag: 'WebSearchError' as const,
            message: 'Magnitude provider is selected but no authentication is configured.',
          })
        }

        const auth = peekResult.auth
        let apiKey: string | undefined

        if (auth.type === 'api') {
          apiKey = auth.key
        } else if (auth.type === 'oauth') {
          apiKey = auth.accessToken
        }

        if (!apiKey) {
          return yield* Effect.fail({
            _tag: 'WebSearchError' as const,
            message: 'Magnitude provider authentication is missing an API key or access token.',
          })
        }

        return yield* callMagnitudeWebSearch(apiKey, query, schema)
      }

      // Check for EXA_API_KEY env var
      const exaApiKey = process.env.EXA_API_KEY
      if (exaApiKey) {
        return yield* callExaWebSearch(exaApiKey, query, schema)
      }

      // Neither Magnitude nor EXA is available
      return yield* Effect.fail({
        _tag: 'WebSearchError' as const,
        message: 'Web search is not available. Use the Magnitude provider or set the EXA_API_KEY environment variable.',
      })
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
