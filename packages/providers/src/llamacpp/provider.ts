import { Effect, Option } from "effect"
import type * as HttpClient from "@effect/platform/HttpClient"
import type {
  BoundModel,
  ModelCatalog,
  Provider,
  BaseCallOptions,
  ProviderModelBindOptions,
} from "@magnitudedev/ai"
import { createLlamaCppCatalog } from "./catalog"
import { checkServerHealth } from "./discovery"
import { createLlamaCppCompatibleSpec, wrapAsBaseModel } from "./models"
import type {
  LlamaCppCallOptions,
  LlamaCppDiscoveryResult,
  LlamaCppModelInfo,
} from "./contract"
import {
  classifyModelFamily as classifyModelFamilyRaw,
  classifyModelFamilyFromMetadata,
  modelFamilyMetadataConflicts,
} from "../family-registry"

export const PROVIDER_ID = "llamacpp" as const

export interface LlamaCppClientConfig {
  readonly endpoint?: string
  readonly apiKey?: string
  readonly sessionId?: string
  readonly auth?: (headers: Headers) => void
}

const DEFAULT_ENDPOINT = "http://127.0.0.1:8080"

export type LlamaCppProvider = Provider<LlamaCppModelInfo>

export interface LlamaCppProviderInstance {
  readonly provider: LlamaCppProvider
  readonly catalog: ModelCatalog<LlamaCppModelInfo>
  readonly checkStatus: Effect.Effect<LlamaCppDiscoveryResult, never, HttpClient.HttpClient>
}

export function createLlamaCppProvider(config?: LlamaCppClientConfig): LlamaCppProviderInstance {
  const configuredEndpoint = config?.endpoint?.trim().replace(/\/+$/, "")
  const endpoint = configuredEndpoint || DEFAULT_ENDPOINT

  const auth = config?.auth ?? (() => {
    const apiKey = config?.apiKey
    if (!apiKey) return undefined
    return (headers: Headers) => {
      headers.set("Authorization", `Bearer ${apiKey}`)
    }
  })()

  const classifyModelFamily = (
    model: Omit<LlamaCppModelInfo, "modelFamilyId">,
  ): Option.Option<string> => {
    const metadata = {
      architecture: model.modelArchitecture,
      tokenizerModel: model.tokenizerModel,
      tokenizerPre: model.tokenizerPre,
    }
    const structuredFamily = classifyModelFamilyFromMetadata(metadata)
    if (Option.isSome(structuredFamily)) return structuredFamily

    const candidates = [
      model.metadataName,
      ...(model.baseModelNames ?? []),
      ...(model.baseModelRepositories ?? []),
      model.sourceModelPath,
      model.displayName,
      model.providerModelId,
      model.modelArchitecture,
    ]
    for (const candidate of candidates) {
      if (!candidate) continue
      const family = classifyModelFamilyRaw(candidate)
      if (
        Option.isSome(family) &&
        !modelFamilyMetadataConflicts(family.value, metadata)
      ) return family
    }
    return Option.none()
  }

  const catalog = createLlamaCppCatalog({
    endpoint,
    auth: auth ?? ((_: Headers) => {}),
    classify: classifyModelFamily,
  })

  const bindModel = (
    providerModelId: string,
    options?: ProviderModelBindOptions,
  ): Effect.Effect<BoundModel<BaseCallOptions>, never, never> =>
    Effect.succeed(
      wrapAsBaseModel(
        createLlamaCppCompatibleSpec({ modelId: providerModelId, endpoint }).bind({
          auth: auth ?? (() => {}),
          defaults: options?.defaults as Partial<LlamaCppCallOptions> | undefined,
          ...(options?.imagePlaceholders ? { imagePlaceholders: options.imagePlaceholders } : {}),
        }),
      ),
    )

  const provider: LlamaCppProvider = {
    id: PROVIDER_ID,
    displayName: "Llama.cpp",
    catalog,
    bindModel,
    classifyModelFamily,
  }

  const checkStatus: LlamaCppProviderInstance["checkStatus"] = Effect.gen(function* () {
    const health = yield* checkServerHealth(endpoint)
    if (health.status === "not_found") {
      return {
        models: [],
        status: "not_found",
        endpoint,
        hint: "Start one with e.g. llama-server -m /path/to/model.gguf",
      }
    }
    if (health.status === "loading") {
      return {
        models: [],
        status: "loading",
        endpoint,
        message: "llama-server is loading a model",
      }
    }
    if (health.status === "error") {
      return {
        models: [],
        status: "error",
        endpoint,
        message: health.message,
      }
    }

    return yield* catalog.list.pipe(
      Effect.map((models): LlamaCppDiscoveryResult => ({
        models,
        status: "ok",
        endpoint,
      })),
      Effect.catchAll((cause) => Effect.succeed({
        models: [],
        status: "error" as const,
        endpoint,
        message: cause.message,
      })),
    )
  })

  return { provider, catalog, checkStatus }
}
