import * as HttpClient from "@effect/platform/HttpClient"
import * as HttpClientRequest from "@effect/platform/HttpClientRequest"
import { Effect, Option } from "effect"
import { ModelCatalogError, type ModelCatalog } from "@magnitudedev/ai"
import { classifyModelFamilyFromEvidence } from "../family-registry"
import type { ModelsDevModel } from "../catalog/models-dev"
import type {
  OpenAiCompatibleCatalogConfig,
  OpenAiCompatibleModelInfo,
  OpenAiCompatibleRawModel,
} from "./contract"

const ZERO_PRICING = { input: 0, output: 0, cached_input: null } as const

function authHeaders(auth: (headers: Headers) => void): Record<string, string> {
  const headers = new Headers()
  auth(headers)
  const record: Record<string, string> = {}
  headers.forEach((value, key) => { record[key] = value })
  return record
}

function parseLiveModels(body: unknown): readonly OpenAiCompatibleRawModel[] | null {
  if (typeof body !== "object" || body === null) return null
  const data = (body as { readonly data?: unknown }).data
  if (!Array.isArray(data)) return null
  const models: OpenAiCompatibleRawModel[] = []
  for (const value of data) {
    if (typeof value !== "object" || value === null) continue
    const entry = value as Record<string, unknown>
    if (typeof entry.id !== "string" || !entry.id.trim()) continue
    const supportedParameters = Array.isArray(entry.supported_parameters)
      ? entry.supported_parameters.filter((parameter): parameter is string => typeof parameter === "string")
      : undefined
    const rawReasoning = typeof entry.reasoning === "object" && entry.reasoning !== null
      ? entry.reasoning as Record<string, unknown>
      : undefined
    const supportedEfforts = Array.isArray(rawReasoning?.supported_efforts)
      ? rawReasoning.supported_efforts.filter((effort): effort is string => typeof effort === "string")
      : rawReasoning?.supported_efforts === null
        ? null
        : undefined
    models.push({
      id: entry.id,
      ...(typeof entry.name === "string" ? { name: entry.name } : {}),
      ...(typeof entry.display_name === "string" ? { display_name: entry.display_name } : {}),
      ...(typeof entry.context_length === "number" ? { context_length: entry.context_length } : {}),
      ...(typeof entry.context_window === "number" ? { context_window: entry.context_window } : {}),
      ...(typeof entry.max_context_length === "number" ? { max_context_length: entry.max_context_length } : {}),
      ...(typeof entry.max_output_tokens === "number" ? { max_output_tokens: entry.max_output_tokens } : {}),
      ...(typeof entry.max_tokens === "number" ? { max_tokens: entry.max_tokens } : {}),
      ...(typeof entry.description === "string" ? { description: entry.description } : {}),
      ...(typeof entry.owned_by === "string" ? { owned_by: entry.owned_by } : {}),
      ...(supportedParameters ? { supported_parameters: supportedParameters } : {}),
      ...(rawReasoning
        ? {
            reasoning: {
              ...(supportedEfforts !== undefined ? { supported_efforts: supportedEfforts } : {}),
              ...(typeof rawReasoning.default_effort === "string"
                ? { default_effort: rawReasoning.default_effort }
                : {}),
              ...(typeof rawReasoning.default_enabled === "boolean"
                ? { default_enabled: rawReasoning.default_enabled }
                : {}),
              ...(typeof rawReasoning.mandatory === "boolean"
                ? { mandatory: rawReasoning.mandatory }
                : {}),
            },
          }
        : {}),
    })
  }
  return models
}

function humanizeModelId(id: string): string {
  const leaf = id.split("/").at(-1) ?? id
  const humanized = leaf
    .replace(/[:_-]+/g, " ")
    .replace(/\b\w/g, (character) => character.toUpperCase())
  return humanized
    .replace(/\b(Glm|Gpt|Oss|Ai|Api)\b/g, (word) => word.toUpperCase())
    .replace(/\b(\d+(?:\.\d+)?)([bkt])\b/gi, (_match, amount: string, unit: string) =>
      `${amount}${unit.toUpperCase()}`)
}

const GATEWAY_REASONING_EFFORTS = ["max", "xhigh", "high", "medium", "low", "minimal", "none"] as const

function reasoningEfforts(
  raw: OpenAiCompatibleRawModel,
  metadata: ModelsDevModel | undefined,
): readonly string[] {
  const live = raw.reasoning?.supported_efforts
  if (live === null) {
    return raw.reasoning?.mandatory
      ? GATEWAY_REASONING_EFFORTS.filter((effort) => effort !== "none")
      : GATEWAY_REASONING_EFFORTS
  }
  if (live && live.length > 0) {
    const unique = live.filter((effort, index, all) => all.indexOf(effort) === index)
    return raw.reasoning?.mandatory
      ? unique.filter((effort) => effort !== "none")
      : unique
  }
  if (!metadata?.reasoning) return ["default"]
  const options = metadata.reasoning_options ?? []
  const values = options
    .filter((option) => option.type === "effort")
    .flatMap((option) => option.values ?? [])
    .filter((value): value is string => typeof value === "string")
    .filter((value, index, all) => all.indexOf(value) === index)
  const hasToggle = options.some((option) => option.type === "toggle")
  if (values.length > 0) return hasToggle ? ["none", ...values] : values
  return hasToggle ? ["none", "high"] : ["default"]
}

function enrichModel(
  raw: OpenAiCompatibleRawModel,
  metadata: ModelsDevModel | undefined,
  config: OpenAiCompatibleCatalogConfig<OpenAiCompatibleModelInfo>,
): OpenAiCompatibleModelInfo {
  const family = classifyModelFamilyFromEvidence({}, [
    raw.id,
    raw.display_name,
    raw.name,
    metadata?.family,
    metadata?.name,
  ])
  const contextWindow = raw.context_length
    ?? raw.context_window
    ?? raw.max_context_length
    ?? metadata?.limit.context
    ?? config.defaultContextWindow
    ?? 128_000
  const maxOutputTokens = raw.max_output_tokens
    ?? raw.max_tokens
    ?? metadata?.limit.output
    ?? config.defaultMaxOutputTokens
    ?? Math.min(contextWindow, 32_768)

  return {
    providerId: config.providerId,
    providerModelId: raw.id,
    modelFamilyId: Option.getOrElse(family, () => "unknown"),
    displayName: raw.display_name ?? raw.name ?? metadata?.name ?? humanizeModelId(raw.id),
    contextWindow,
    maxOutputTokens,
    capabilities: {
      vision: metadata?.modalities.input.includes("image") ?? false,
      toolCalls: metadata?.tool_call ?? true,
      structuredOutput: metadata?.structured_output ?? false,
      grammar: false,
      toolChoiceModes: config.toolChoiceModes ?? ["auto", "none", "required", "named"],
    },
    pricing: metadata?.cost
      ? {
          input: metadata.cost.input,
          output: metadata.cost.output,
          cached_input: metadata.cost.cache_read ?? null,
        }
      : ZERO_PRICING,
    reasoningEfforts: reasoningEfforts(raw, metadata),
    openWeightStatus: metadata ? (metadata.open_weights ? "open" : "closed") : "unknown",
    metadataSource: metadata ? "models.dev" : "provider",
    ...(raw.description ?? metadata?.description
      ? { description: raw.description ?? metadata?.description }
      : {}),
    ...(metadata?.family ? { upstreamFamily: metadata.family } : {}),
    ...(metadata ? { modalities: metadata.modalities } : {}),
  }
}

function isEligible(
  raw: OpenAiCompatibleRawModel,
  metadata: ModelsDevModel | undefined,
  config: OpenAiCompatibleCatalogConfig<OpenAiCompatibleModelInfo>,
): boolean {
  if (config.requireOpenWeights && metadata?.open_weights !== true) return false
  if (config.requireToolCalls && metadata?.tool_call === false) return false
  if (
    config.requireToolCalls
    && raw.supported_parameters
    && !raw.supported_parameters.includes("tools")
  ) return false
  if (metadata && !metadata.modalities.output.includes("text")) return false
  return true
}

export function createOpenAiCompatibleCatalog<
  TModel extends OpenAiCompatibleModelInfo,
>(config: OpenAiCompatibleCatalogConfig<TModel>): ModelCatalog<TModel> {
  const ttlMs = config.ttlMs ?? 5 * 60 * 1000
  let cache: readonly TModel[] | null = null
  let fetchedAt = 0

  const fetchLiveModels = Effect.gen(function* () {
    const client = yield* HttpClient.HttpClient
    const response = yield* client.execute(
      HttpClientRequest.get(`${config.endpoint.replace(/\/+$/, "")}/models`).pipe(
        HttpClientRequest.setHeaders(authHeaders(config.auth)),
      ),
    ).pipe(
      Effect.mapError((cause) => new ModelCatalogError({
        message: `Failed to fetch ${config.providerId} models`,
        cause,
      })),
    )

    if (response.status < 200 || response.status >= 300) {
      const body = yield* response.text.pipe(Effect.orElseSucceed(() => ""))
      return yield* new ModelCatalogError({
        message: `Failed to fetch ${config.providerId} models: HTTP ${response.status} - ${body}`,
        cause: { status: response.status },
      })
    }
    const body = yield* response.json.pipe(
      Effect.mapError((cause) => new ModelCatalogError({
        message: `Failed to read ${config.providerId} models`,
        cause,
      })),
    )
    const models = parseLiveModels(body)
    if (!models) {
      return yield* new ModelCatalogError({
        message: `${config.providerId} returned an invalid model list`,
      })
    }
    return models
  })

  const fetchModels: Effect.Effect<readonly TModel[], ModelCatalogError, HttpClient.HttpClient> =
    Effect.gen(function* () {
      const metadataResult = yield* Effect.either(
        config.modelsDev.getProvider(config.modelsDevProviderId),
      )
      if (metadataResult._tag === "Left" && config.requireOpenWeights) {
        return yield* metadataResult.left
      }
      const modelsDevProvider = metadataResult._tag === "Right" ? metadataResult.right : null
      const fallbackModels = Object.values(modelsDevProvider?.models ?? {}).map((model) => ({
        id: model.id,
        name: model.name,
      }))

      let rawModels: readonly OpenAiCompatibleRawModel[]
      if (config.liveCatalog === false) {
        rawModels = fallbackModels
      } else {
        const liveResult = yield* Effect.either(fetchLiveModels)
        if (liveResult._tag === "Right") {
          rawModels = liveResult.right
        } else if (config.liveCatalogFallback === "always") {
          rawModels = fallbackModels
        } else if (
          config.liveCatalogFallback === "unsupported_only"
          && typeof liveResult.left.cause === "object"
          && liveResult.left.cause !== null
          && "status" in liveResult.left.cause
          && [404, 405, 501].includes(Number(liveResult.left.cause.status))
        ) {
          rawModels = fallbackModels
        } else {
          return yield* liveResult.left
        }
      }

      const models: TModel[] = []
      for (const raw of rawModels) {
        const metadata = modelsDevProvider?.models[raw.id]
        if (!isEligible(raw, metadata, config)) continue
        const enriched = enrichModel(
          raw,
          metadata,
          config as OpenAiCompatibleCatalogConfig<OpenAiCompatibleModelInfo>,
        )
        models.push(config.mapModel ? config.mapModel(enriched) : enriched as TModel)
      }
      return models
    })

  const refresh: ModelCatalog<TModel>["refresh"] = fetchModels.pipe(
    Effect.tap((models) => Effect.sync(() => {
      cache = models
      fetchedAt = Date.now()
    })),
    Effect.catchAll((error) => cache ? Effect.succeed(cache) : Effect.fail(error)),
  )

  const list: ModelCatalog<TModel>["list"] = Effect.gen(function* () {
    if (cache && Date.now() - fetchedAt < ttlMs) return cache
    return yield* refresh
  })

  const get: ModelCatalog<TModel>["get"] = (_providerId, providerModelId) =>
    Effect.gen(function* () {
      const models = yield* list
      const model = models.find((candidate) => candidate.providerModelId === providerModelId)
      if (!model) {
        return yield* new ModelCatalogError({
          message: `Model not found: ${config.providerId}/${providerModelId}`,
        })
      }
      return model
    })

  return { list, get, refresh }
}
