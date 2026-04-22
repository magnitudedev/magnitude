/**
 * Web Search Tool
 *
 * Searches the web using either the Magnitude provider API or EXA API.
 * - If Magnitude provider is active: calls https://app.magnitude.dev/api/v1/web-search
 * - If EXA_API_KEY env var is set: calls EXA API via @effect/platform HttpClient
 * - Otherwise: fails with clear error
 */

import { Effect } from 'effect'
import { Schema } from '@effect/schema'
import { defineTool, ToolErrorSchema } from '@magnitudedev/tools'
import { AmbientServiceTag, Fork } from '@magnitudedev/event-core'
import { ProviderState } from '@magnitudedev/providers'
import { ConfigAmbient } from '../ambient/config-ambient'
import { WebSearchService } from '../services/web-search-service'

const { ForkContext } = Fork

const WebSearchErrorSchema = ToolErrorSchema('WebSearchError', {})

// =============================================================================
// Tool Definition
// =============================================================================

export const webSearchTool = defineTool({
  name: 'web_search',
  group: 'default',
  description: 'Search the web and optionally extract structured data',

  inputSchema: Schema.Struct({
    query: Schema.String.annotations({ description: 'Search query string' }),
    schema: Schema.optional(Schema.Record({ key: Schema.String, value: Schema.Unknown }).annotations({ description: 'Optional schema for structured data extraction' }))
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
      const webSearchService = yield* WebSearchService
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

        return yield* webSearchService.searchMagnitude(apiKey, query, schema)
      }

      // Check for EXA_API_KEY env var
      const exaApiKey = process.env.EXA_API_KEY
      if (exaApiKey) {
        return yield* webSearchService.searchExa(exaApiKey, query, schema)
      }

      // Neither Magnitude nor EXA is available
      return yield* Effect.fail({
        _tag: 'WebSearchError' as const,
        message: 'Web search is not available. Use the Magnitude provider or set the EXA_API_KEY environment variable.',
      })
    }),

  label: (input) => input.query ? `Searching: ${input.query.slice(0, 50)}` : 'Searching…',
})
