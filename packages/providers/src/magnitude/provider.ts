import { Context, Data, Duration, Effect, Option, Schema } from "effect"
import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import {
  Auth,
  JsonValueSchema,
  type AuthApplicator,
  type BoundModel,
  type Provider,
  type ModelCatalog,
  type WebSearchExtension,
  type WebSearchResult,
  type BalanceExtension,
  type BalanceQuery,
  type ProviderModelCapabilities,
  type ImagePlaceholderConfig,
  type BaseCallOptions,
  type ProviderModelBindOptions,
} from "@magnitudedev/ai"
import { isEnvFlagOn } from "@magnitudedev/utils"
import type { MagnitudeModelInfo, MagnitudeAdditionalOptions } from "./contract"
import { classifyModelFamily as classifyModelFamilyRaw } from "../family-registry"
import { createMagnitudeCatalog } from "./catalog"
import type { BalanceResponse as MagnitudeBalanceResponse } from "./usage"
import type { UsagePeriod } from "./usage"
import { createMagnitudeCompatibleSpec, wrapAsBaseModel, type MagnitudeCallOptions } from "./models"
import { CLIENT_PLATFORM, CLIENT_SHELL, HEADER_PLATFORM, HEADER_SHELL, HEADER_SESSION_ID, HEADER_USE_DEDICATED } from "./client-headers"

export class WebSearchError extends Data.TaggedError("WebSearchError")<{
  readonly message: string
  readonly cause?: unknown
}> {}

export class MagnitudeClientError extends Data.TaggedError("MagnitudeClientError")<{
  readonly message: string
  readonly cause?: unknown
}> {}

const WebSearchResultSchema = Schema.Struct({
  text: Schema.String,
  sources: Schema.Array(Schema.Struct({
    title: Schema.String,
    url: Schema.String,
  })),
  data: Schema.optional(JsonValueSchema),
})

export const PROVIDER_ID = "magnitude" as const

export interface MagnitudeClientConfig {
  readonly apiKey?: string
  readonly endpoint?: string
  readonly sessionId?: string
  readonly dedicatedProvider?: string
  readonly auth?: AuthApplicator
}

const DEFAULT_ENDPOINT = "https://app.magnitude.dev/api/v1"
const LOCAL_ENDPOINT = "http://localhost:3000/api/v1"

export interface FetchBalanceOptions {
  readonly period?: UsagePeriod
  readonly days?: number
  readonly tz?: string
}

/**
 * The Magnitude provider — implements Provider<MagnitudeModelInfo, MagnitudeCallOptions>
 * & WebSearchExtension & BalanceExtension.
 */
export interface MagnitudeProvider extends Provider<MagnitudeModelInfo> {
  readonly webSearch: WebSearchExtension<WebSearchResult, WebSearchError, HttpClient.HttpClient>["webSearch"]
  readonly balance: BalanceExtension<MagnitudeBalanceResponse, MagnitudeClientError, HttpClient.HttpClient>["balance"]
}

export interface MagnitudeProviderInstance {
  readonly provider: MagnitudeProvider
  readonly catalog: ModelCatalog<MagnitudeModelInfo>
}

export function createMagnitudeProvider(config?: MagnitudeClientConfig): MagnitudeProviderInstance {
  const useLocal = isEnvFlagOn(process.env.MAGNITUDE_USE_LOCAL)
  const endpoint = config?.endpoint ?? (useLocal ? LOCAL_ENDPOINT : DEFAULT_ENDPOINT)
  const sessionId = config?.sessionId ?? null
  const dedicatedProvider = config?.dedicatedProvider || process.env.MAGNITUDE_USE_DEDICATED || undefined

  const auth: AuthApplicator = config?.auth ?? (() => {
    const apiKey = config?.apiKey ?? (useLocal ? process.env.MAGNITUDE_LOCAL_API_KEY : undefined) ?? process.env.MAGNITUDE_API_KEY
    if (!apiKey) throw new Error(
      useLocal
        ? "No API key provided. Set MAGNITUDE_LOCAL_API_KEY (or MAGNITUDE_API_KEY) environment variable, or pass apiKey in config."
        : "No API key provided. Pass apiKey in config or set MAGNITUDE_API_KEY environment variable."
    )
    return Auth.bearer(apiKey)
  })()

  const authWithHeaders: AuthApplicator = (headers) => {
    auth(headers)
    headers.set(HEADER_PLATFORM, CLIENT_PLATFORM)
    headers.set(HEADER_SHELL, CLIENT_SHELL)
    if (sessionId) headers.set(HEADER_SESSION_ID, sessionId)
    if (dedicatedProvider) headers.set(HEADER_USE_DEDICATED, dedicatedProvider)
  }

  const classifyModelFamily = (model: Omit<MagnitudeModelInfo, "modelFamilyId">): Option.Option<string> =>
    classifyModelFamilyRaw(model.providerModelId)

  const catalog = createMagnitudeCatalog({ endpoint, auth: authWithHeaders, classify: classifyModelFamily })

  const bindModel = (
    id: string,
    options?: ProviderModelBindOptions,
  ): Effect.Effect<BoundModel<BaseCallOptions>, never, never> =>
    Effect.gen(function* () {
      // Build magnitude-specific options from bind options
      const magnitudeOptions: MagnitudeAdditionalOptions = {
        ...(options?.agentId ? { agent_id: options.agentId } : {}),
        ...(options?.traits ? { traits: [...options.traits] } : {}),
        ...(options?.preferProvider ? { prefer_provider: options.preferProvider } : {}),
        ...(sessionId ? { session_id: sessionId } : {}),
      }

      const internal = createMagnitudeCompatibleSpec({ modelId: id, endpoint }).bind({
        auth: authWithHeaders,
        defaults: options?.defaults as Partial<MagnitudeCallOptions> | undefined,
        ...(options?.imagePlaceholders ? { imagePlaceholders: options.imagePlaceholders } : {}),
      })

      return wrapAsBaseModel(internal, magnitudeOptions)
    })

  const webSearch: WebSearchExtension<WebSearchResult, WebSearchError, HttpClient.HttpClient>["webSearch"] = (query, schema?) =>
    Effect.gen(function* () {
      const http = yield* HttpClient.HttpClient

      const headers = new Headers()
      authWithHeaders(headers)

      const headerRecord: Record<string, string> = {}
      headers.forEach((value, key) => {
        headerRecord[key] = value
      })
      headerRecord["Content-Type"] = "application/json"

      const body = schema ? { query, schema } : { query }

      const request = HttpClientRequest.post(`${endpoint}/web-search`).pipe(
        HttpClientRequest.setHeaders(headerRecord),
        HttpClientRequest.bodyJson(body),
      )

      const response = yield* Effect.flatMap(
        request,
        (req) => http.execute(req)
      ).pipe(
        Effect.mapError((cause) => new WebSearchError({ message: "Request failed", cause })),
        Effect.timeoutFail({
          onTimeout: () => new WebSearchError({ message: "Request timed out after 10 seconds" }),
          duration: Duration.seconds(10),
        }),
      )

      if (response.status < 200 || response.status >= 300) {
        const text = yield* response.text.pipe(Effect.orElseSucceed(() => ""))
        return yield* new WebSearchError({ message: `HTTP ${response.status}: ${text}` })
      }

      const text = yield* response.text.pipe(
        Effect.mapError((cause) => new WebSearchError({ message: "Failed to read response", cause })),
      )

      const parsed = yield* Schema.decodeUnknown(Schema.parseJson(WebSearchResultSchema))(text).pipe(
        Effect.mapError((cause) => new WebSearchError({ message: `Failed to parse response: ${text.slice(0, 200)}`, cause })),
      )

      return {
        text: parsed.text,
        sources: parsed.sources,
        data: parsed.data,
      } satisfies WebSearchResult
    })

  const balance: BalanceExtension<MagnitudeBalanceResponse, MagnitudeClientError, HttpClient.HttpClient>["balance"] = (query?) =>
    Effect.gen(function* () {
      const http = yield* HttpClient.HttpClient
      const headers = new Headers()
      authWithHeaders(headers)
      const headerRecord: Record<string, string> = {}
      headers.forEach((value, key) => { headerRecord[key] = value })

      const params = new URLSearchParams()
      if (query?.period) params.set("period", query.period)
      if (query?.days != null) params.set("days", String(query.days))
      if (query?.tz) params.set("tz", query.tz)
      const qs = params.toString()
      const url = `${endpoint}/balance${qs ? `?${qs}` : ""}`

      const request = HttpClientRequest.get(url).pipe(
        HttpClientRequest.setHeaders(headerRecord),
      )
      const response = yield* http.execute(request).pipe(
        Effect.mapError((cause) => new MagnitudeClientError({ message: "Failed to fetch balance", cause })),
      )
      if (response.status < 200 || response.status >= 300) {
        const body = yield* response.text.pipe(Effect.orElseSucceed(() => ""))
        return yield* new MagnitudeClientError({ message: `Failed to fetch balance: HTTP ${response.status} - ${body}` })
      }
      const body = yield* response.json.pipe(
        Effect.mapError((cause) => new MagnitudeClientError({ message: "Failed to read balance response", cause })),
      )
      return body as MagnitudeBalanceResponse
    })

  const provider: MagnitudeProvider = {
    id: PROVIDER_ID,
    displayName: "Magnitude",
    catalog,
    bindModel,
    classifyModelFamily,
    webSearch,
    balance,
  }

  return { provider, catalog }
}

export async function fetchBalance(
  apiKey?: string,
  endpoint?: string,
  options?: FetchBalanceOptions,
): Promise<MagnitudeBalanceResponse> {
  const { FetchHttpClient } = await import("@effect/platform")
  const instance = createMagnitudeProvider({ apiKey, endpoint })
  const query: BalanceQuery = {
    period: options?.period,
    days: options?.days,
    tz: options?.tz,
  }
  return Effect.runPromise(
    instance.provider.balance(query).pipe(Effect.provide(FetchHttpClient.layer)),
  )
}
