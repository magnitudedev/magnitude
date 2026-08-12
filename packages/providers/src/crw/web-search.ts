import { Duration, Effect, Option, Schema } from "effect"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import {
  payloadSample,
  type WebSearchExtension,
  type WebSearchResult,
} from "@magnitudedev/ai"
import {
  CrwSearchErrorResponseSchema,
  CrwSearchRequestSchema,
  CrwSearchResponseSchema,
  type CrwSearchResult,
} from "./contract"
import {
  WebSearchInvalidResponse,
  WebSearchNotConfigured,
  WebSearchRejected,
  WebSearchRequestEncodingFailed,
  WebSearchRequestFailed,
  WebSearchResponseReadFailed,
  WebSearchStructuredOutputUnsupported,
  WebSearchTimedOut,
  type WebSearchError,
} from "../web-search-error"

const DEFAULT_BASE_URL = "https://fastcrw.com/api"
const SEARCH_PATH = "/v1/search"
const DEFAULT_RESULTS = 10
// Longer than Exa's 10s: a self-hosted engine may be cold-starting a renderer
// on the user's own machine rather than answering from a warm hosted service.
const TIMEOUT_MS = 30_000

export interface CrwWebSearchConfig {
  readonly apiKey?: string
  /** Engine base URL, e.g. `http://localhost:3000` for a self-hosted engine. */
  readonly baseUrl?: string
}

export interface CrwWebSearchInstance {
  readonly configured: boolean
  readonly webSearch: WebSearchExtension<WebSearchResult, WebSearchError, HttpClient.HttpClient>["webSearch"]
}

const decodeErrorMessage = (body: string): string =>
  Option.getOrElse(
    Schema.decodeUnknownOption(Schema.parseJson(CrwSearchErrorResponseSchema))(body).pipe(
      Option.map((parsed) => parsed.error),
    ),
    () => body.slice(0, 500),
  )

const errorReason = (error: unknown): string =>
  (error instanceof Error
    ? error.message || error.name
    : String(error)).slice(0, 1_000)

const trimTrailingSlash = (value: string): string => value.replace(/\/+$/, "")

const resultTitle = (result: CrwSearchResult): string =>
  Option.getOrElse(
    Option.flatMap(result.title, Option.fromNullable),
    () => result.url,
  )

const resultBody = (result: CrwSearchResult): string =>
  Option.getOrElse(
    Option.orElse(
      Option.flatMap(result.snippet, Option.fromNullable),
      () => Option.flatMap(result.description, Option.fromNullable),
    ),
    () => "",
  )

export function createCrwWebSearch(config?: CrwWebSearchConfig): CrwWebSearchInstance {
  const apiKey = config?.apiKey ?? process.env.CRW_API_KEY
  const normalizedApiKey = apiKey?.trim() || null
  const configuredBaseUrl = (config?.baseUrl ?? process.env.CRW_API_URL)?.trim() || null
  const endpoint = `${trimTrailingSlash(configuredBaseUrl ?? DEFAULT_BASE_URL)}${SEARCH_PATH}`

  // A self-hosted engine needs no key, so an explicit base URL is enough to
  // enable the source. The managed endpoint does need one.
  const configured = normalizedApiKey !== null || configuredBaseUrl !== null

  const webSearch: CrwWebSearchInstance["webSearch"] = (query, schema) =>
    Effect.gen(function* () {
      if (!configured) {
        return yield* new WebSearchNotConfigured()
      }

      // The engine's search endpoint has no structured-extraction equivalent.
      // Fail loudly rather than returning a success envelope with no `data`,
      // which the caller cannot tell apart from "nothing matched".
      if (schema !== undefined) {
        return yield* new WebSearchStructuredOutputUnsupported({ provider: "crw" })
      }

      const http = yield* HttpClient.HttpClient
      const requestBody = yield* Schema.decodeUnknown(CrwSearchRequestSchema)({
        query,
        limit: DEFAULT_RESULTS,
      }).pipe(
        Effect.flatMap(Schema.encode(CrwSearchRequestSchema)),
        Effect.mapError((error) => new WebSearchRequestEncodingFailed({
          provider: "crw",
          reason: errorReason(error),
        })),
      )
      const request = yield* HttpClientRequest.post(endpoint).pipe(
        HttpClientRequest.setHeaders({
          "Content-Type": "application/json",
          // Self-hosted engines run without auth; sending an empty bearer
          // token would be rejected, so omit the header entirely.
          ...(normalizedApiKey === null ? {} : { Authorization: `Bearer ${normalizedApiKey}` }),
        }),
        HttpClientRequest.bodyJson(requestBody),
        Effect.mapError((error) => new WebSearchRequestEncodingFailed({
          provider: "crw",
          reason: errorReason(error),
        })),
      )

      return yield* Effect.gen(function* () {
        const response = yield* http.execute(request).pipe(
          Effect.mapError((error) => new WebSearchRequestFailed({
            provider: "crw",
            reason: errorReason(error),
          })),
        )

        const body = yield* response.text.pipe(
          Effect.mapError((error) => new WebSearchResponseReadFailed({
            provider: "crw",
            reason: errorReason(error),
          })),
        )

        if (response.status < 200 || response.status >= 300) {
          return yield* new WebSearchRejected({
            provider: "crw",
            status: response.status,
            message: decodeErrorMessage(body),
            body: payloadSample(body),
          })
        }

        const parsed = yield* Schema.decodeUnknown(Schema.parseJson(CrwSearchResponseSchema))(body).pipe(
          Effect.mapError((error) => new WebSearchInvalidResponse({
            provider: "crw",
            body: payloadSample(body),
            issue: errorReason(error),
          })),
        )

        if (Option.getOrElse(parsed.success, () => true) === false) {
          return yield* new WebSearchRejected({
            provider: "crw",
            status: response.status,
            message: Option.getOrElse(parsed.error, () => "search failed"),
            body: payloadSample(body),
          })
        }

        // An absent payload means this is not a search response, so surface it
        // rather than reporting it to the caller as "the web returned nothing".
        if (Option.isNone(parsed.data)) {
          return yield* new WebSearchInvalidResponse({
            provider: "crw",
            body: payloadSample(body),
            issue: "response contained no results payload",
          })
        }
        const payload = parsed.data.value
        const results: readonly CrwSearchResult[] = "results" in payload
          ? payload.results
          : payload

        return {
          text: results
            .map((result) => `## ${resultTitle(result)}\n${resultBody(result)}`)
            .join("\n\n"),
          sources: results.map((result) => ({ title: resultTitle(result), url: result.url })),
        } satisfies WebSearchResult
      }).pipe(
        Effect.timeoutFail({
          onTimeout: () => new WebSearchTimedOut({ provider: "crw", timeoutMs: TIMEOUT_MS }),
          duration: Duration.millis(TIMEOUT_MS),
        }),
      )
    })

  return {
    configured,
    webSearch,
  }
}
