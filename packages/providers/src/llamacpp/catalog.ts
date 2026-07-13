import { Effect, Option } from "effect"
import * as HttpClient from "@effect/platform/HttpClient"
import { ModelCatalogError, type ModelCatalog, type ModelCatalogConfig } from "@magnitudedev/ai"
import type { LlamaCppModelInfo, LlamaCppRawModel } from "./contract"
import {
  fetchModelList,
  fetchServerProps,
  deriveDisplayName,
  deriveContextWindow,
  detectVision,
  type LlamaCppDiscoveryConfig,
} from "./discovery"

const ZERO_PRICING = { input: 0, output: 0, cached_input: null } as const

type LlamaCppModelWithoutFamily = Omit<LlamaCppModelInfo, "modelFamilyId">

/**
 * Llama.cpp model catalog implementation.
 *
 * Discovers models from a local Llama.cpp server via GET /v1/models,
 * classifies each into a family, and filters out unclassified models.
 */
export interface LlamaCppCatalogConfig {
  readonly endpoint: string
  readonly auth: (headers: Headers) => void
  readonly ttlMs?: number
  readonly classify: (model: LlamaCppModelWithoutFamily) => Option.Option<string>
}

export function createLlamaCppCatalog(
  config: LlamaCppCatalogConfig,
): ModelCatalog<LlamaCppModelInfo> {
  const { endpoint, auth, ttlMs = 5 * 60 * 1000, classify } = config

  let cache: readonly LlamaCppModelInfo[] | null = null
  let fetchedAt = 0

  const discoveryConfig: LlamaCppDiscoveryConfig = { endpoint, auth }

  const fetchModels: Effect.Effect<readonly LlamaCppModelInfo[], ModelCatalogError, HttpClient.HttpClient> =
    Effect.gen(function* () {
      const rawModels = yield* fetchModelList(discoveryConfig).pipe(
        Effect.mapError((cause) => new ModelCatalogError({ message: cause.message, cause })),
      )

      const serverProps = yield* fetchServerProps(discoveryConfig)

      const classified: LlamaCppModelInfo[] = []
      for (const raw of rawModels) {
        const displayName = deriveDisplayName(raw)
        const contextWindow = deriveContextWindow(raw, serverProps)
        const vision = detectVision(raw)

        const model: LlamaCppModelWithoutFamily = {
          providerModelId: raw.id,
          providerId: "llamacpp",
          displayName,
          contextWindow,
          maxOutputTokens: Math.min(contextWindow, 8192),
          capabilities: { vision },
          pricing: ZERO_PRICING,
          reasoningEfforts: ["none"],
          ...(raw.meta?.n_ctx !== undefined ? { serverContextSize: raw.meta.n_ctx } : {}),
        }

        const familyOption = classify(model)
        if (Option.isNone(familyOption)) continue
        classified.push({ ...model, modelFamilyId: familyOption.value })
      }

      return classified
    })

  const list: ModelCatalog<LlamaCppModelInfo>["list"] = Effect.gen(function* () {
    if (cache && Date.now() - fetchedAt < ttlMs) {
      return cache
    }
    const models = yield* fetchModels
    cache = models
    fetchedAt = Date.now()
    return models
  })

  const get: ModelCatalog<LlamaCppModelInfo>["get"] = (_providerId, providerModelId) =>
    Effect.gen(function* () {
      const models = yield* list
      const model = models.find((m) => m.providerModelId === providerModelId)
      if (!model) {
        return yield* new ModelCatalogError({ message: `Model not found: ${providerModelId}` })
      }
      return model
    })

  const refresh: ModelCatalog<LlamaCppModelInfo>["refresh"] = Effect.gen(function* () {
    const models = yield* fetchModels
    cache = models
    fetchedAt = Date.now()
    return models
  })

  return { list, get, refresh }
}
