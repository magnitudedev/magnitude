import { Effect, Option } from "effect"
import type { CachedData } from "../cache"
import { CatalogueCache } from "../cache"
import {
  CatalogueSchemaError,
  CatalogueTransportError,
  type CatalogueError,
  type CatalogueSource,
} from "../types"
import type { ProviderModel } from "../../lib/model/provider-model"

interface OpenRouterModel {
  readonly id: string
  readonly name: string
  readonly context_length: number
  readonly pricing: {
    readonly prompt: string
    readonly completion: string
    readonly input_cache_read?: string
    readonly input_cache_write?: string
  }
  readonly top_provider?: {
    readonly context_length?: number | null
    readonly max_completion_tokens?: number | null
  } | null
  readonly architecture?: {
    readonly input_modalities?: readonly string[]
    readonly output_modalities?: readonly string[]
  } | null
  readonly supported_parameters?: readonly string[]
  readonly created: number
  readonly expiration_date?: string | null
}

interface OpenRouterResponse {
  readonly data: readonly OpenRouterModel[]
}

const SOURCE_ID = "openrouter-api"
const API_URL = "https://openrouter.ai/api/v1/models"
const FETCH_TIMEOUT_MS = 10_000
const TTL_MS = 6 * 60 * 60 * 1000

function parsePrice(value?: string): number {
  if (!value) {
    return 0
  }
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed * 1_000_000 : 0
}

function toIsoDate(created: number): string {
  return new Date(created * 1000).toISOString()
}

function normalizeOpenRouterModels(data: OpenRouterResponse): readonly ProviderModel[] {
  return [...data.data]
    .map(
      (model): ProviderModel => ({
        id: model.id,
        providerId: "openrouter",
        providerName: "OpenRouter",
        canonicalModelId: null,
        name: model.name,
        contextWindow: model.context_length,
        maxContextTokens: null,
        maxOutputTokens: model.top_provider?.max_completion_tokens ?? null,
        supportsToolCalls: model.supported_parameters?.includes("tools") ?? false,
        supportsReasoning: model.supported_parameters?.includes("reasoning") ?? false,
        supportsVision: model.architecture?.input_modalities?.includes("image") ?? false,
        costs: {
          inputPerM: parsePrice(model.pricing.prompt),
          outputPerM: parsePrice(model.pricing.completion),
          cacheReadPerM: model.pricing.input_cache_read
            ? parsePrice(model.pricing.input_cache_read)
            : null,
          cacheWritePerM: model.pricing.input_cache_write
            ? parsePrice(model.pricing.input_cache_write)
            : null,
        },
        releaseDate: toIsoDate(model.created),
        discovery: {
          primarySource: "openrouter-api",
        },
      }),
    )
    .sort((left, right) => (right.releaseDate ?? "").localeCompare(left.releaseDate ?? ""))
}

const fetchOpenRouterResponse: Effect.Effect<OpenRouterResponse, CatalogueError> = Effect.tryPromise({
  try: async () => {
    const response = await fetch(API_URL, {
      signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
      headers: { "User-Agent": "magnitude-agent" },
    })
    if (!response.ok) {
      throw new Error(`OpenRouter API returned ${response.status}`)
    }
    return (await response.json()) as OpenRouterResponse
  },
  catch: (cause) =>
    new CatalogueTransportError({
      sourceId: SOURCE_ID,
      providerId: "openrouter",
      message: "Failed to fetch OpenRouter catalogue",
      cause,
    }),
}).pipe(
  Effect.flatMap((data) =>
    Effect.try({
      try: () => data,
      catch: (cause) =>
        new CatalogueSchemaError({
          sourceId: SOURCE_ID,
          providerId: "openrouter",
          message: "Invalid OpenRouter response",
          cause,
        }),
    }),
  ),
)

function resolveFromCache(
  sourceId: string,
  ttlMs: number,
  fetchFresh: Effect.Effect<OpenRouterResponse, CatalogueError>,
): Effect.Effect<OpenRouterResponse | null, never, CatalogueCache> {
  return Effect.gen(function* () {
    const cache = yield* CatalogueCache
    const cached = yield* cache.load<OpenRouterResponse>(sourceId)

    if (cached && cache.isValid(cached)) {
      return cached.data
    }

    const fresh = yield* fetchFresh.pipe(Effect.option)
    if (Option.isSome(fresh)) {
      yield* cache.save(sourceId, fresh.value, ttlMs)
      return fresh.value
    }

    if (cached && cache.isStale(cached)) {
      return cached.data
    }

    return null
  })
}

export const openRouterCatalogueSource: CatalogueSource = {
  id: SOURCE_ID,
  fetch: Effect.gen(function* () {
    const cache = yield* Effect.serviceOption(CatalogueCache)

    const data = yield* Option.match(cache, {
      onNone: () => fetchOpenRouterResponse.pipe(Effect.option),
      onSome: (service) =>
        resolveFromCache(SOURCE_ID, TTL_MS, fetchOpenRouterResponse).pipe(
          Effect.provideService(CatalogueCache, service),
          Effect.map(Option.fromNullable),
        ),
    })

    return Option.match(data, {
      onNone: (): readonly ProviderModel[] => [],
      onSome: normalizeOpenRouterModels,
    })
  }),
}
