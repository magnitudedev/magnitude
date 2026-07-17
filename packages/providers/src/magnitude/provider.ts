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
  type UsageExtension,
  type UsageQuery,
  type BaseCallOptions,
  type ProviderModelBindOptions,
  ProviderIdSchema,
  ModelDiscoveryOperationIdSchema,
  type ModelFamilyId,
  type ProviderModelId,
} from "@magnitudedev/ai"
import { isEnvFlagOn } from "@magnitudedev/utils"
import type { MagnitudeModelInfo, MagnitudeAdditionalOptions } from "./contract"
import { classifyModelFamily as classifyModelFamilyRaw } from "../family-registry"
import { createMagnitudeCatalog, type MagnitudeAuthentication } from "./catalog"
import type { CloudUsageResponse } from "./usage"
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

export const PROVIDER_ID = ProviderIdSchema.make("magnitude")

export interface MagnitudeClientConfig {
  readonly apiKey?: string
  readonly endpoint?: string
  readonly sessionId?: string
  readonly dedicatedProvider?: string
  readonly auth?: AuthApplicator
}

const DEFAULT_ENDPOINT = "https://app.magnitude.dev/api/v1"
const LOCAL_ENDPOINT = "http://localhost:3000/api/v1"

export interface FetchUsageOptions {
  readonly period?: UsagePeriod
  readonly days?: number
  readonly tz?: string
}

/**
 * The Magnitude provider — implements Provider<MagnitudeModelInfo, MagnitudeCallOptions>
 * & WebSearchExtension & UsageExtension.
 */
export interface MagnitudeProvider extends Provider<MagnitudeModelInfo> {
  readonly webSearch: WebSearchExtension<WebSearchResult, WebSearchError, HttpClient.HttpClient>["webSearch"]
  readonly usage: UsageExtension<CloudUsageResponse, MagnitudeClientError, HttpClient.HttpClient>["usage"]
}

export interface MagnitudeProviderInstance {
  readonly provider: MagnitudeProvider
  readonly catalog: ModelCatalog<MagnitudeModelInfo>
  readonly authentication: MagnitudeAuthentication
}

export function createMagnitudeProvider(config?: MagnitudeClientConfig): MagnitudeProviderInstance {
  const useLocal = isEnvFlagOn(process.env.MAGNITUDE_USE_LOCAL)
  const endpoint = config?.endpoint ?? (useLocal ? LOCAL_ENDPOINT : DEFAULT_ENDPOINT)
  const sessionId = config?.sessionId ?? null
  const dedicatedProvider = config?.dedicatedProvider || process.env.MAGNITUDE_USE_DEDICATED || undefined

  const apiKey = config?.apiKey ?? process.env.MAGNITUDE_API_KEY
  const authentication: MagnitudeAuthentication = config?.auth !== undefined
    ? { _tag: "Configured", apply: config.auth }
    : apiKey?.trim()
      ? { _tag: "Configured", apply: Auth.bearer(apiKey) }
      : { _tag: "NotConfigured" }

  const applyClientHeaders = (headers: Headers) => {
    headers.set(HEADER_PLATFORM, CLIENT_PLATFORM)
    headers.set(HEADER_SHELL, CLIENT_SHELL)
    if (sessionId) headers.set(HEADER_SESSION_ID, sessionId)
    if (dedicatedProvider) headers.set(HEADER_USE_DEDICATED, dedicatedProvider)
  }
  const requestAuthentication: MagnitudeAuthentication = authentication._tag === "Configured"
    ? {
        _tag: "Configured",
        apply: (headers) => {
          authentication.apply(headers)
          applyClientHeaders(headers)
        },
      }
    : authentication
  const modelAuth: AuthApplicator = requestAuthentication._tag === "Configured"
    ? requestAuthentication.apply
    : applyClientHeaders

  const classifyModelFamily = (model: Omit<MagnitudeModelInfo, "modelFamilyId">): Option.Option<ModelFamilyId> =>
    classifyModelFamilyRaw(model.providerModelId)

  const catalog = createMagnitudeCatalog({ endpoint, authentication: requestAuthentication, classify: classifyModelFamily })

  const bindModel = (
    id: ProviderModelId,
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
        auth: modelAuth,
        defaults: options?.defaults as Partial<MagnitudeCallOptions> | undefined,
        ...(options?.imagePlaceholders ? { imagePlaceholders: options.imagePlaceholders } : {}),
      })

      return wrapAsBaseModel(internal, magnitudeOptions)
    })

  const webSearch: WebSearchExtension<WebSearchResult, WebSearchError, HttpClient.HttpClient>["webSearch"] = (query, schema?) =>
    Effect.gen(function* () {
      if (requestAuthentication._tag === "NotConfigured") {
        return yield* new WebSearchError({ message: "Magnitude authentication is not configured" })
      }

      const http = yield* HttpClient.HttpClient
      const headers = new Headers()
      requestAuthentication.apply(headers)

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

  const usage: UsageExtension<CloudUsageResponse, MagnitudeClientError, HttpClient.HttpClient>["usage"] = (query?) =>
    Effect.gen(function* () {
      if (requestAuthentication._tag === "NotConfigured") {
        return yield* new MagnitudeClientError({ message: "Magnitude authentication is not configured" })
      }

      const http = yield* HttpClient.HttpClient
      const headers = new Headers()
      requestAuthentication.apply(headers)
      const headerRecord: Record<string, string> = {}
      headers.forEach((value, key) => { headerRecord[key] = value })

      const params = new URLSearchParams()
      if (query?.period) params.set("period", query.period)
      if (query?.days != null) params.set("days", String(query.days))
      if (query?.tz) params.set("tz", query.tz)
      const qs = params.toString()
      const url = `${endpoint}/usage${qs ? `?${qs}` : ""}`

      const request = HttpClientRequest.get(url).pipe(
        HttpClientRequest.setHeaders(headerRecord),
      )
      const response = yield* http.execute(request).pipe(
        Effect.mapError((cause) => new MagnitudeClientError({ message: "Failed to fetch cloud usage", cause })),
      )
      if (response.status < 200 || response.status >= 300) {
        const body = yield* response.text.pipe(Effect.orElseSucceed(() => ""))
        return yield* new MagnitudeClientError({ message: `Failed to fetch cloud usage: HTTP ${response.status} - ${body}` })
      }
      const body = yield* response.json.pipe(
        Effect.mapError((cause) => new MagnitudeClientError({ message: "Failed to read cloud usage response", cause })),
      )
      return body as CloudUsageResponse
    })

  const provider: MagnitudeProvider = {
    id: PROVIDER_ID,
    displayName: "Magnitude",
    catalog,
    discoverModelProperties: () => Effect.succeed(ModelDiscoveryOperationIdSchema.make("magnitude-authoritative")),
    bindModel,
    classifyModelFamily,
    webSearch,
    usage,
  }

  return { provider, catalog, authentication }
}

export async function fetchUsage(
  apiKey?: string,
  endpoint?: string,
  options?: FetchUsageOptions,
): Promise<CloudUsageResponse> {
  const { FetchHttpClient } = await import("@effect/platform")
  const instance = createMagnitudeProvider({ apiKey, endpoint })
  const query: UsageQuery = {
    period: options?.period,
    days: options?.days,
    tz: options?.tz,
  }
  return Effect.runPromise(
    instance.provider.usage(query).pipe(Effect.provide(FetchHttpClient.layer)),
  )
}
