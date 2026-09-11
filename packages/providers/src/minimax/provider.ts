import { Effect } from "effect"
import {
  Auth,
  ModelDiscoveryOperationIdSchema,
  type AuthApplicator,
  type BaseCallOptions,
  type BoundModel,
  type ModelCatalog,
  type Provider,
  type ProviderModelBindOptions,
  type ProviderModelId,
  type StreamStartFailure,
} from "@magnitudedev/ai"
import { classifyModelFamily } from "../family-registry"
import { createMiniMaxCatalog, MINIMAX_PROVIDER_ID } from "./catalog"
import type {
  MiniMaxAuthentication,
  MiniMaxEndpointConfig,
  MiniMaxModelInfo,
  MiniMaxRegion,
} from "./contract"
import { createMiniMaxCompatibleSpec, wrapAsBaseModel } from "./models"

export const MINIMAX_ENDPOINTS: Readonly<Record<MiniMaxRegion, MiniMaxEndpointConfig>> = {
  global_en: {
    openaiBaseUrl: "https://api.minimax.io/v1",
    anthropicBaseUrl: "https://api.minimax.io/anthropic",
  },
  cn_zh: {
    openaiBaseUrl: "https://api.minimaxi.com/v1",
    anthropicBaseUrl: "https://api.minimaxi.com/anthropic",
  },
}

export interface MiniMaxClientConfig {
  readonly apiKey?: string
  readonly region?: MiniMaxRegion
}

export interface MiniMaxProviderInstance<TPreparation = never> {
  readonly provider: Provider<MiniMaxModelInfo, TPreparation>
  readonly catalog: ModelCatalog<MiniMaxModelInfo>
  readonly authentication: MiniMaxAuthentication
  readonly endpoints: MiniMaxEndpointConfig
}

export const createMiniMaxProvider = <TPreparation = never>(
  config?: MiniMaxClientConfig,
): MiniMaxProviderInstance<TPreparation> => {
  const apiKey = (config?.apiKey ?? process.env.MINIMAX_API_KEY)?.trim()
  const authentication: MiniMaxAuthentication = apiKey
    ? { _tag: "Configured" }
    : { _tag: "NotConfigured" }
  const auth: AuthApplicator = apiKey
    ? Auth.bearer(apiKey)
    : () => { throw new Error("MiniMax authentication is not configured") }
  const endpoints = MINIMAX_ENDPOINTS[config?.region ?? "global_en"]
  const catalog = createMiniMaxCatalog()

  const bindModel = (
    providerModelId: ProviderModelId,
    options?: ProviderModelBindOptions,
  ): Effect.Effect<BoundModel<BaseCallOptions, StreamStartFailure, TPreparation>> => Effect.sync(() => {
    const model = createMiniMaxCompatibleSpec({
      modelId: providerModelId,
      endpoint: endpoints.openaiBaseUrl,
      reasoningMode: providerModelId === "MiniMax-M3" ? "configurable" : "always_on",
    }).bind({
      auth,
      ...(options?.defaults === undefined ? {} : { defaults: options.defaults }),
      ...(options?.imagePlaceholders ? { imagePlaceholders: options.imagePlaceholders } : {}),
    })
    return wrapAsBaseModel(model)
  })

  return {
    provider: {
      id: MINIMAX_PROVIDER_ID,
      displayName: "MiniMax",
      catalog,
      bindModel,
      classifyModelFamily: (model) => classifyModelFamily(model.providerModelId),
      discoverModelProperties: () => Effect.succeed(
        ModelDiscoveryOperationIdSchema.make("minimax-authoritative"),
      ),
    },
    catalog,
    authentication,
    endpoints,
  }
}
