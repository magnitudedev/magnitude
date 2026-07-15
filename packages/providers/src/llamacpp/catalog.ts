import { Effect, Option } from "effect"
import * as HttpClient from "@effect/platform/HttpClient"
import { ModelCatalogError, type ModelCatalog } from "@magnitudedev/ai"
import type { LlamaCppModelInfo, LlamaCppRawModel } from "./contract"
import {
  fetchModelList,
  fetchServerProps,
  checkServerHealth,
  deriveDisplayName,
  deriveContextWindow,
  detectVision,
  deriveMetadataName,
  deriveModelArchitecture,
  deriveSourceModelPath,
  deriveTokenizerModel,
  deriveTokenizerPre,
  type LlamaCppDiscoveryConfig,
} from "./discovery"

const ZERO_PRICING = { input: 0, output: 0, cached_input: null } as const

type LlamaCppModelWithoutFamily = Omit<LlamaCppModelInfo, "modelFamilyId">

/**
 * Llama.cpp model catalog implementation.
 *
 * Discovers models from a local Llama.cpp server via GET /v1/models,
 * enriches known model families, and keeps unclassified local models visible.
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
  let cacheTtlMs = ttlMs

  const discoveryConfig: LlamaCppDiscoveryConfig = { endpoint, auth }
  const fetchModels: Effect.Effect<{
    readonly models: readonly LlamaCppModelInfo[]
    readonly cacheTtlMs: number
  }, ModelCatalogError, HttpClient.HttpClient> =
    Effect.gen(function* () {
      const health = yield* checkServerHealth(endpoint)
      if (health.status !== "ready") {
        return {
          models: [],
          cacheTtlMs: health.status === "loading" ? 10_000 : 30_000,
        }
      }

      const rawModels = yield* fetchModelList(discoveryConfig).pipe(
        Effect.mapError((cause) => new ModelCatalogError({ message: cause.message, cause })),
      )

      const serverProps = yield* fetchServerProps(discoveryConfig)

      const models: LlamaCppModelInfo[] = []
      for (const raw of rawModels) {
        const modelProps = rawModels.length === 1 ? serverProps : null
        const sourceModelPath = deriveSourceModelPath(raw, modelProps, rawModels.length)
        const displayName = deriveDisplayName(raw, modelProps, sourceModelPath)
        const contextWindow = deriveContextWindow(raw, modelProps)
        const vision = detectVision(raw, modelProps)
        const metadataName = deriveMetadataName(raw)
        const modelArchitecture = deriveModelArchitecture(raw)
        const tokenizerModel = deriveTokenizerModel(raw)
        const tokenizerPre = deriveTokenizerPre(raw)

        const model: LlamaCppModelWithoutFamily = {
          providerModelId: raw.id,
          providerId: "llamacpp",
          displayName,
          contextWindow,
          maxOutputTokens: Math.min(contextWindow, 8192),
          capabilities: { vision },
          pricing: ZERO_PRICING,
          reasoningEfforts: ["none"],
          ...(sourceModelPath ? { sourceModelPath } : {}),
          ...(metadataName ? { metadataName } : {}),
          ...(modelArchitecture ? { modelArchitecture } : {}),
          ...(tokenizerModel ? { tokenizerModel } : {}),
          ...(tokenizerPre ? { tokenizerPre } : {}),
          ...(modelProps?.nCtx !== undefined
            ? { serverContextSize: modelProps.nCtx }
            : raw.meta?.n_ctx !== undefined
              ? { serverContextSize: raw.meta.n_ctx }
              : {}),
        }

        const familyOption = classify(model)
        models.push({
          ...model,
          modelFamilyId: Option.getOrElse(familyOption, () => "unknown"),
        })
      }

      return { models, cacheTtlMs: ttlMs }
    })

  const list: ModelCatalog<LlamaCppModelInfo>["list"] = Effect.gen(function* () {
    if (cache && Date.now() - fetchedAt < cacheTtlMs) {
      return cache
    }
    const result = yield* fetchModels
    cache = result.models
    cacheTtlMs = result.cacheTtlMs
    fetchedAt = Date.now()
    return result.models
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
    const result = yield* fetchModels
    cache = result.models
    cacheTtlMs = result.cacheTtlMs
    fetchedAt = Date.now()
    return result.models
  })

  return { list, get, refresh }
}
