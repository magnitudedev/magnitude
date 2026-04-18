
/**
 * WebSearchService
 *
 * Effect service for web search using @effect/platform HttpClient.
 * Provides interruptable HTTP requests via Effect's fiber system.
 */

import { Context, Effect, Layer, Duration } from 'effect'
import { Schema } from '@effect/schema'
import { HttpClient, HttpClientRequest, HttpClientError } from '@effect/platform'

// =============================================================================
// Typed Errors
// =============================================================================

export class WebSearchRateLimitError extends Schema.TaggedError<WebSearchRateLimitError>()(
  'WebSearchRateLimitError',
  { message: Schema.String, provider: Schema.String }
) {}

export class WebSearchInvalidQueryError extends Schema.TaggedError<WebSearchInvalidQueryError>()(
  'WebSearchInvalidQueryError',
  { message: Schema.String, provider: Schema.String }
) {}

export class WebSearchServiceError extends Schema.TaggedError<WebSearchServiceError>()(
  'WebSearchServiceError',
  { message: Schema.String, provider: Schema.String, status: Schema.optional(Schema.Number) }
) {}

export class WebSearchTimeoutError extends Schema.TaggedError<WebSearchTimeoutError>()(
  'WebSearchTimeoutError',
  { message: Schema.String, provider: Schema.String }
) {}

export type WebSearchError =
  | WebSearchRateLimitError
  | WebSearchInvalidQueryError
  | WebSearchServiceError
  | WebSearchTimeoutError

// =============================================================================
// Response Types
// =============================================================================

export interface WebSearchResult {
  readonly text: string
  readonly sources: ReadonlyArray<{ readonly title: string; readonly url: string }>
  readonly data: unknown | null
}

// Magnitude API response shape
interface MagnitudeResponse {
  text: string
  sources: Array<{ title: string; url: string }>
  data: unknown | null
}

// Exa Search API response shapes
interface ExaSearchResult {
  title?: string
  url: string
  highlights?: string[]
}

interface ExaOutput {
  content?: unknown
}

interface ExaSearchResponse {
  results: ExaSearchResult[]
  output?: ExaOutput
}

// =============================================================================
// Service Interface
// =============================================================================

export interface WebSearchServiceShape {
  readonly searchMagnitude: (
    apiKey: string,
    query: string,
    schema?: Record<string, unknown>
  ) => Effect.Effect<WebSearchResult, WebSearchError>

  readonly searchExa: (
    apiKey: string,
    query: string,
    schema?: Record<string, unknown>
  ) => Effect.Effect<WebSearchResult, WebSearchError>
}

export class WebSearchService extends Context.Tag('WebSearchService')<
  WebSearchService,
  WebSearchServiceShape
>() {}

// =============================================================================
// Error Classification
// =============================================================================

const classifyHttpError = (error: unknown, provider: string): WebSearchError => {
  // Check if it's an HttpClientError
  if (HttpClientError.isHttpClientError(error)) {
    // Check for ResponseError (has response status)
    if (error._tag === 'ResponseError') {
      const status = error.response.status
      
      if (status === 429) {
        return new WebSearchRateLimitError({
          message: `Rate limit exceeded (${status})`,
          provider,
        })
      }
      
      if (status === 400) {
        return new WebSearchInvalidQueryError({
          message: `Invalid query (${status})`,
          provider,
        })
      }
      
      return new WebSearchServiceError({
        message: `HTTP error: ${status}`,
        provider,
        status,
      })
    }
    
    // RequestError or other HttpClientError
    return new WebSearchServiceError({
      message: error.message,
      provider,
    })
  }
  
  // Unknown error
  return new WebSearchServiceError({
    message: error instanceof Error ? error.message : String(error),
    provider,
  })
}

// =============================================================================
// HTTP Helper
// =============================================================================

const postJson = (http: HttpClient.HttpClient) => <T>(
  url: string,
  headers: Record<string, string>,
  body: unknown,
  provider: string
): Effect.Effect<T, WebSearchError> =>
  HttpClientRequest.post(url).pipe(
    HttpClientRequest.setHeaders(headers),
    HttpClientRequest.bodyJson(body),
    Effect.mapError((e) => new WebSearchServiceError({
      message: `Request serialization failed: ${e}`,
      provider,
    })),
    Effect.flatMap((req) => http.execute(req)),
    Effect.mapError((error) => classifyHttpError(error, provider)),
    Effect.flatMap((response) =>
      response.json.pipe(
        Effect.mapError((e) => new WebSearchServiceError({
          message: `JSON parse failed: ${e}`,
          provider,
        }))
      ) as Effect.Effect<T, WebSearchError>
    ),
    // 10 second timeout — fails with WebSearchTimeoutError
    Effect.timeoutFail({
      onTimeout: () => new WebSearchTimeoutError({
        message: 'Request timed out after 10 seconds',
        provider,
      }),
      duration: Duration.seconds(10),
    })
  )

// =============================================================================
// Layer Implementation
// =============================================================================

export const WebSearchServiceLive = Layer.effect(
  WebSearchService,
  Effect.gen(function* () {
    const http = yield* HttpClient.HttpClient

    return {
      searchMagnitude: (apiKey, query, schema) =>
        Effect.gen(function* () {
          const body = schema ? { query, schema } : { query }
          
          const result = yield* postJson(http)<MagnitudeResponse>(
            'https://app.magnitude.dev/api/v1/web-search',
            { 
              'Authorization': `Bearer ${apiKey}`, 
              'Content-Type': 'application/json' 
            },
            body,
            'magnitude'
          )
          
          return {
            text: result.text,
            sources: result.sources,
            data: result.data,
          }
        }),

      searchExa: (apiKey, query, schema) =>
        Effect.gen(function* () {
          const body: Record<string, unknown> = {
            query,
            type: 'auto',
            numResults: 10,
            contents: { highlights: true },
          }
          
          if (schema) {
            body.outputSchema = { type: 'object', properties: schema }
          }

          const result = yield* postJson(http)<ExaSearchResponse>(
            'https://api.exa.ai/search',
            { 
              'x-api-key': apiKey, 
              'Content-Type': 'application/json' 
            },
            body,
            'exa'
          )

          // Map Exa response to common format
          const sources = result.results.map((r) => ({
            title: r.title ?? r.url,
            url: r.url,
          }))

          const text = result.results
            .map((r) => {
              const title = r.title ?? 'Untitled'
              const highlights = Array.isArray(r.highlights)
                ? r.highlights.join('\n')
                : ''
              return `## ${title}\n${highlights}`
            })
            .join('\n\n')

          return { 
            text, 
            sources, 
            data: result.output?.content ?? null 
          }
        }),
    }
  })
)
