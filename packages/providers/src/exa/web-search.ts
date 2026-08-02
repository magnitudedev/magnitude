import { Duration, Effect, Option, Schema } from "effect"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import {
  payloadSample,
  type WebSearchExtension,
  type WebSearchResult,
} from "@magnitudedev/ai"
import {
  ExaSearchErrorResponseSchema,
  ExaSearchRequestSchema,
  ExaSearchResponseSchema,
} from "./contract"
import {
  WebSearchInvalidResponse,
  WebSearchNotConfigured,
  WebSearchRejected,
  WebSearchRequestEncodingFailed,
  WebSearchRequestFailed,
  WebSearchResponseReadFailed,
  WebSearchTimedOut,
  type WebSearchError,
} from "../web-search-error"

const DEFAULT_ENDPOINT = "https://api.exa.ai/search"

export interface ExaWebSearchConfig {
  readonly apiKey?: string
  readonly endpoint?: string
}

export interface ExaWebSearchInstance {
  readonly configured: boolean
  readonly webSearch: WebSearchExtension<WebSearchResult, WebSearchError, HttpClient.HttpClient>["webSearch"]
}

const decodeErrorMessage = (body: string): string =>
  Option.getOrElse(
    Schema.decodeUnknownOption(Schema.parseJson(ExaSearchErrorResponseSchema))(body).pipe(
      Option.map((parsed) => parsed.error),
    ),
    () => body.slice(0, 500),
  )

const errorReason = (error: unknown): string =>
  (error instanceof Error
    ? error.message || error.name
    : String(error)).slice(0, 1_000)

export function createExaWebSearch(config?: ExaWebSearchConfig): ExaWebSearchInstance {
  const apiKey = config?.apiKey ?? process.env.EXA_API_KEY
  const normalizedApiKey = apiKey?.trim() || null
  const endpoint = config?.endpoint ?? DEFAULT_ENDPOINT

  const webSearch: ExaWebSearchInstance["webSearch"] = (query, schema) =>
    Effect.gen(function* () {
      if (normalizedApiKey === null) {
        return yield* new WebSearchNotConfigured()
      }

      const http = yield* HttpClient.HttpClient
      const requestBody = yield* Schema.decodeUnknown(ExaSearchRequestSchema)({
        query,
        type: "auto",
        numResults: 10,
        contents: { highlights: true },
        ...(schema === undefined ? {} : { outputSchema: schema }),
      }).pipe(
        Effect.flatMap(Schema.encode(ExaSearchRequestSchema)),
        Effect.mapError((error) => new WebSearchRequestEncodingFailed({
          provider: "exa",
          reason: errorReason(error),
        })),
      )
      const request = yield* HttpClientRequest.post(endpoint).pipe(
        HttpClientRequest.setHeaders({
          "x-api-key": normalizedApiKey,
          "Content-Type": "application/json",
        }),
        HttpClientRequest.bodyJson(requestBody),
        Effect.mapError((error) => new WebSearchRequestEncodingFailed({
          provider: "exa",
          reason: errorReason(error),
        })),
      )

      return yield* Effect.gen(function* () {
        const response = yield* http.execute(request).pipe(
          Effect.mapError((error) => new WebSearchRequestFailed({
            provider: "exa",
            reason: errorReason(error),
          })),
        )

        if (response.status < 200 || response.status >= 300) {
          const body = yield* response.text.pipe(
            Effect.mapError((error) => new WebSearchResponseReadFailed({
              provider: "exa",
              reason: errorReason(error),
            })),
          )
          const detail = decodeErrorMessage(body)
          return yield* new WebSearchRejected({
            provider: "exa",
            status: response.status,
            message: detail,
            body: payloadSample(body),
          })
        }

        const body = yield* response.text.pipe(
          Effect.mapError((error) => new WebSearchResponseReadFailed({
            provider: "exa",
            reason: errorReason(error),
          })),
        )
        const parsed = yield* Schema.decodeUnknown(Schema.parseJson(ExaSearchResponseSchema))(body).pipe(
          Effect.mapError((error) => new WebSearchInvalidResponse({
            provider: "exa",
            body: payloadSample(body),
            issue: errorReason(error),
          })),
        )

        return {
          text: parsed.results
            .map((result) => `## ${result.title ?? result.url}\n${Option.getOrElse(result.highlights, () => []).join("\n")}`)
            .join("\n\n"),
          sources: parsed.results.map(({ title, url }) => ({ title: title ?? url, url })),
          ...(Option.isSome(parsed.output) ? { data: parsed.output.value.content } : {}),
        } satisfies WebSearchResult
      }).pipe(
        Effect.timeoutFail({
          onTimeout: () => new WebSearchTimedOut({ provider: "exa", timeoutMs: 10_000 }),
          duration: Duration.seconds(10),
        }),
      )
    })

  return {
    configured: normalizedApiKey !== null,
    webSearch,
  }
}
