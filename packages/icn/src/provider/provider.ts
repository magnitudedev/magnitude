import { Effect, Option } from "effect"
import type * as HttpClient from "@effect/platform/HttpClient"
import type {
  BaseCallOptions,
  BoundModel,
  ModelCatalog,
  ModelDiscoveryOperationId,
  ModelPropertyDiscoveryError,
  ModelPropertyDiscoveryRequest,
  Provider,
  ProviderModelBindOptions,
  ProviderModelId,
} from "@magnitudedev/ai"
import { ProviderIdSchema } from "@magnitudedev/ai"
import type { LocalModelInfo } from "./contract"

export const PROVIDER_ID = ProviderIdSchema.make("local")

/** Effect-native local provider source backed by the single managed ICN client. */
export interface LocalProviderSource {
  readonly catalog: ModelCatalog<LocalModelInfo>
  readonly discoverModelProperties: (
    request: ModelPropertyDiscoveryRequest,
  ) => Effect.Effect<ModelDiscoveryOperationId, ModelPropertyDiscoveryError>
  readonly bindModel: (
    providerModelId: ProviderModelId,
    options?: ProviderModelBindOptions,
  ) => Effect.Effect<BoundModel<BaseCallOptions>, never>
  readonly status: Effect.Effect<{
    readonly status: "ok" | "loading" | "not_found" | "error"
    readonly message?: string
    readonly hint?: string
  }, never, HttpClient.HttpClient>
}

export interface LocalProviderInstance {
  readonly provider: Provider<LocalModelInfo>
  readonly checkStatus: LocalProviderSource["status"]
}

export function createLocalProvider(source: LocalProviderSource): LocalProviderInstance {
  return {
    provider: {
      id: PROVIDER_ID,
      displayName: "Local",
      catalog: source.catalog,
      discoverModelProperties: source.discoverModelProperties,
      bindModel: source.bindModel,
      classifyModelFamily: () => Option.none(),
    },
    checkStatus: source.status,
  }
}
