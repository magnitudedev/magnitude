import { Effect } from "effect"
import {
  Auth,
  type BaseCallOptions,
  type BoundModel,
  type Provider,
  type ProviderModelBindOptions,
} from "@magnitudedev/ai"
import { classifyModelFamilyFromEvidence } from "../family-registry"
import type { OpenAiCompatibleProviderInstance } from "../openai-compatible"
import { createKimiApiCatalog } from "./catalog"
import type { KimiApiCallOptions, KimiApiClientConfig, KimiApiModelInfo } from "./contract"
import { createKimiApiCompatibleSpec, wrapAsBaseModel } from "./models"

export const PROVIDER_ID = "kimi-api" as const
export const DEFAULT_KIMI_API_ENDPOINT = "https://api.moonshot.ai/v1"

export type KimiApiProvider = Provider<KimiApiModelInfo>
export type KimiApiProviderInstance = OpenAiCompatibleProviderInstance<KimiApiModelInfo>

export function createKimiApiProvider(config: KimiApiClientConfig): KimiApiProviderInstance {
  const endpoint = config.endpoint?.trim().replace(/\/+$/, "") || DEFAULT_KIMI_API_ENDPOINT
  const auth = config.auth ?? Auth.bearer(config.apiKey ?? "")
  const classifyModelFamily: KimiApiProvider["classifyModelFamily"] = (model) =>
    classifyModelFamilyFromEvidence({}, [model.upstreamFamily, model.providerModelId, model.displayName])
  const catalog = createKimiApiCatalog({ ...config, endpoint, auth })

  const bindModel = (
    providerModelId: string,
    options?: ProviderModelBindOptions,
  ): Effect.Effect<BoundModel<BaseCallOptions>, never, never> => Effect.succeed(
    wrapAsBaseModel(
      createKimiApiCompatibleSpec({ modelId: providerModelId, endpoint }).bind({
        auth,
        defaults: options?.defaults as Partial<KimiApiCallOptions> | undefined,
        ...(options?.imagePlaceholders ? { imagePlaceholders: options.imagePlaceholders } : {}),
      }),
    ),
  )

  return {
    catalog,
    provider: {
      id: PROVIDER_ID,
      displayName: "Kimi API",
      catalog,
      bindModel,
      classifyModelFamily,
    },
  }
}

