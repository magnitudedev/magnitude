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
import { readLocalGgufMetadata } from "./gguf-metadata"

const ZERO_PRICING = { input: 0, output: 0, cached_input: null } as const

type LlamaCppModelWithoutFamily = Omit<LlamaCppModelInfo, "modelFamilyId">

function isLoopbackEndpoint(endpoint: string): boolean {
  try {
    const hostname = new URL(endpoint).hostname.toLowerCase()
    return hostname === "localhost"
      || hostname === "0.0.0.0"
      || hostname === "[::1]"
      || hostname === "::1"
      || /^127(?:\.\d{1,3}){3}$/.test(hostname)
  } catch {
    return false
  }
}

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
  const canReadLocalModelFiles = isLoopbackEndpoint(endpoint)

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
        const ggufMetadata = yield* readLocalGgufMetadata(
          canReadLocalModelFiles ? sourceModelPath : undefined,
        )
        const enrichedRaw: LlamaCppRawModel = ggufMetadata
          ? {
              ...raw,
              meta: {
                ...(raw.meta ?? {}),
                ...(ggufMetadata.generalName ? { "general.name": ggufMetadata.generalName } : {}),
                ...(ggufMetadata.generalBasename
                  ? { "general.basename": ggufMetadata.generalBasename }
                  : {}),
                ...(ggufMetadata.generalSizeLabel
                  ? { "general.size_label": ggufMetadata.generalSizeLabel }
                  : {}),
                ...(ggufMetadata.generalFinetune
                  ? { "general.finetune": ggufMetadata.generalFinetune }
                  : {}),
                ...(ggufMetadata.generalVersion
                  ? { "general.version": ggufMetadata.generalVersion }
                  : {}),
                ...(ggufMetadata.architecture
                  ? { "general.architecture": ggufMetadata.architecture }
                  : {}),
                ...(ggufMetadata.tokenizerModel
                  ? { "tokenizer.ggml.model": ggufMetadata.tokenizerModel }
                  : {}),
                ...(ggufMetadata.tokenizerPre
                  ? { "tokenizer.ggml.pre": ggufMetadata.tokenizerPre }
                  : {}),
              },
            }
          : raw
        const displayName = deriveDisplayName(enrichedRaw, modelProps, sourceModelPath)
        const contextWindow = deriveContextWindow(enrichedRaw, modelProps)
        const vision = detectVision(enrichedRaw, modelProps)
        const metadataName = deriveMetadataName(enrichedRaw)
        const modelArchitecture = deriveModelArchitecture(enrichedRaw)
        const tokenizerModel = deriveTokenizerModel(enrichedRaw)
        const tokenizerPre = deriveTokenizerPre(enrichedRaw)

        const model: LlamaCppModelWithoutFamily = {
          providerModelId: raw.id,
          providerId: "llamacpp",
          displayName,
          contextWindow,
          maxOutputTokens: Math.min(contextWindow, 8192),
          capabilities: {
            vision,
            toolCalls: true,
            structuredOutput: true,
            grammar: true,
            toolChoiceModes: ["auto", "none", "required", "named"],
          },
          pricing: ZERO_PRICING,
          reasoningEfforts: ["none"],
          ...(ggufMetadata
            ? { metadataSource: "local_metadata" as const }
            : raw.meta
              ? { metadataSource: "provider" as const }
              : {}),
          modalities: {
            input: vision ? ["text", "image"] : ["text"],
            output: ["text"],
          },
          ...(sourceModelPath ? { sourceModelPath } : {}),
          ...(metadataName ? { metadataName } : {}),
          ...(modelArchitecture ? { modelArchitecture } : {}),
          ...(tokenizerModel ? { tokenizerModel } : {}),
          ...(tokenizerPre ? { tokenizerPre } : {}),
          ...(ggufMetadata?.baseModelNames.length
            ? { baseModelNames: ggufMetadata.baseModelNames }
            : {}),
          ...(ggufMetadata?.baseModelRepositories.length
            ? { baseModelRepositories: ggufMetadata.baseModelRepositories }
            : {}),
          ...(modelProps?.nCtx !== undefined
            ? { serverContextSize: modelProps.nCtx }
            : enrichedRaw.meta?.n_ctx !== undefined
              ? { serverContextSize: enrichedRaw.meta.n_ctx }
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
