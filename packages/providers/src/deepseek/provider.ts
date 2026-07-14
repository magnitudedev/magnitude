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
import { createDeepSeekCatalog } from "./catalog"
import type {
  DeepSeekCallOptions,
  DeepSeekClientConfig,
  DeepSeekModelInfo,
} from "./contract"
import { createDeepSeekCompatibleSpec, wrapAsBaseModel } from "./models"

export const PROVIDER_ID = "deepseek" as const
export const DEFAULT_DEEPSEEK_ENDPOINT = "https://api.deepseek.com"

export type DeepSeekProvider = Provider<DeepSeekModelInfo>
export type DeepSeekProviderInstance = OpenAiCompatibleProviderInstance<DeepSeekModelInfo>

export function createDeepSeekProvider(
  config: DeepSeekClientConfig,
): DeepSeekProviderInstance {
  const endpoint = config.endpoint?.trim().replace(/\/+$/, "") || DEFAULT_DEEPSEEK_ENDPOINT
  const auth = config.auth ?? Auth.bearer(config.apiKey ?? "")
  const classifyModelFamily: DeepSeekProvider["classifyModelFamily"] = (model) =>
    classifyModelFamilyFromEvidence({}, [
      model.providerModelId,
      model.displayName,
      model.upstreamFamily,
    ])
  const catalog = createDeepSeekCatalog({ ...config, endpoint, auth })

  const bindModel = (
    providerModelId: string,
    options?: ProviderModelBindOptions,
  ): Effect.Effect<BoundModel<BaseCallOptions>, never, never> => Effect.succeed(
    wrapAsBaseModel(
      createDeepSeekCompatibleSpec({ modelId: providerModelId, endpoint }).bind({
        auth,
        defaults: options?.defaults as Partial<DeepSeekCallOptions> | undefined,
        ...(options?.imagePlaceholders ? { imagePlaceholders: options.imagePlaceholders } : {}),
      }),
    ),
  )

  return {
    catalog,
    provider: {
      id: PROVIDER_ID,
      displayName: "DeepSeek API",
      catalog,
      bindModel,
      classifyModelFamily,
    },
  }
}
