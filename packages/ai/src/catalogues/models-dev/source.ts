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
import { getAllProviders } from "../../providers/registry"

interface ModelsDevModel {
  readonly id: string
  readonly name: string
  readonly family?: string
  readonly tool_call: boolean
  readonly reasoning: boolean
  readonly attachment?: boolean
  readonly temperature?: boolean
  readonly release_date?: string
  readonly status?: string
  readonly cost?: {
    readonly input: number
    readonly output: number
    readonly cache_read?: number
    readonly cache_write?: number
  }
  readonly limit?: {
    readonly context?: number
    readonly output?: number
    readonly input?: number
  }
  readonly modalities?: {
    readonly input?: readonly string[]
    readonly output?: readonly string[]
  }
}

interface ModelsDevProvider {
  readonly id: string
  readonly name: string
  readonly env: readonly string[]
  readonly npm?: string
  readonly api?: string
  readonly models: Record<string, ModelsDevModel>
}

type ModelsDevResponse = Record<string, ModelsDevProvider>

const SOURCE_ID = "models.dev"
const API_URL = "https://models.dev/api.json"
const FETCH_TIMEOUT_MS = 10_000
const TTL_MS = 24 * 60 * 60 * 1000

function resolveCanonicalModelId(modelId: string): ProviderModel["canonicalModelId"] {
  for (const provider of getAllProviders()) {
    for (const model of provider.models) {
      if (model.id === modelId) {
        return model.canonicalModelId
      }
      if (model.canonicalModelId === modelId) {
        return model.canonicalModelId
      }
    }
  }
  return null
}

function normalizeProvider(providerId: string, data: ModelsDevResponse): readonly ProviderModel[] {
  const providerData = data[providerId]
  if (!providerData?.models) {
    return []
  }

  const models = Object.values(providerData.models).flatMap((model): readonly ProviderModel[] => {
    if (!model.limit?.context || !model.cost || !model.family) {
      return []
    }

    return [
      {
        id: model.id,
        providerId,
        providerName: providerData.name ?? providerId,
        canonicalModelId: resolveCanonicalModelId(model.id),
        name: model.name,
        contextWindow: model.limit.context,
        maxContextTokens: null,
        maxOutputTokens: model.limit.output ?? null,
        supportsToolCalls: model.tool_call,
        supportsReasoning: model.reasoning,
        supportsVision: model.modalities?.input?.includes("image") ?? false,
        costs: {
          inputPerM: model.cost.input,
          outputPerM: model.cost.output,
          cacheReadPerM: model.cost.cache_read ?? null,
          cacheWritePerM: model.cost.cache_write ?? null,
        },
        releaseDate: model.release_date ?? "1970-01-01T00:00:00.000Z",
        discovery: {
          primarySource: "models.dev",
        },
      },
    ]
  })

  return [...models].sort((left, right) =>
    (right.releaseDate ?? "").localeCompare(left.releaseDate ?? ""),
  )
}

const fetchModelsDevResponse: Effect.Effect<ModelsDevResponse, CatalogueError> = Effect.tryPromise({
  try: async () => {
    const response = await fetch(API_URL, {
      signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
      headers: { "User-Agent": "magnitude-agent" },
    })
    if (!response.ok) {
      throw new Error(`models.dev API returned ${response.status}`)
    }
    return (await response.json()) as ModelsDevResponse
  },
  catch: (cause) =>
    new CatalogueTransportError({
      sourceId: SOURCE_ID,
      providerId: "*",
      message: "Failed to fetch models.dev catalogue",
      cause,
    }),
}).pipe(
  Effect.flatMap((data) =>
    Effect.try({
      try: () => data,
      catch: (cause) =>
        new CatalogueSchemaError({
          sourceId: SOURCE_ID,
          providerId: "*",
          message: "Invalid models.dev response",
          cause,
        }),
    }),
  ),
)

function resolveFromCache(
  sourceId: string,
  ttlMs: number,
  fetchFresh: Effect.Effect<ModelsDevResponse, CatalogueError>,
): Effect.Effect<ModelsDevResponse | null, never, CatalogueCache> {
  return Effect.gen(function* () {
    const cache = yield* CatalogueCache
    const cached = yield* cache.load<ModelsDevResponse>(sourceId)

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

export const modelsDevCatalogueSource: CatalogueSource = {
  id: SOURCE_ID,
  fetch: () =>
    Effect.gen(function* () {
      const cache = yield* Effect.serviceOption(CatalogueCache)

      const data = yield* Option.match(cache, {
        onNone: () => fetchModelsDevResponse.pipe(Effect.option),
        onSome: (service) =>
          resolveFromCache(SOURCE_ID, TTL_MS, fetchModelsDevResponse).pipe(
            Effect.provideService(CatalogueCache, service),
            Effect.map(Option.fromNullable),
          ),
      })

      return Option.match(data, {
        onNone: () => new Map<string, readonly ProviderModel[]>(),
        onSome: (response) => {
          const modelsByProvider = new Map<string, readonly ProviderModel[]>()

          for (const provider of getAllProviders()) {
            if (provider.family !== "cloud" || provider.id === "magnitude") {
              continue
            }

            const models = normalizeProvider(provider.id, response)
            if (models.length > 0) {
              modelsByProvider.set(provider.id, models)
            }
          }

          return modelsByProvider
        },
      })
    }),
}
