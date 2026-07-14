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
import { createOpenRouterCatalog } from "./catalog"
import type { OpenRouterCallOptions, OpenRouterClientConfig, OpenRouterModelInfo } from "./contract"
import { createOpenRouterCompatibleSpec, wrapAsBaseModel } from "./models"

export const PROVIDER_ID = "openrouter" as const
export const DEFAULT_OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1"

export type OpenRouterProvider = Provider<OpenRouterModelInfo>
export type OpenRouterProviderInstance = OpenAiCompatibleProviderInstance<OpenRouterModelInfo>

export function createOpenRouterProvider(config: OpenRouterClientConfig): OpenRouterProviderInstance {
  const endpoint = config.endpoint?.trim().replace(/\/+$/, "") || DEFAULT_OPENROUTER_ENDPOINT
  const auth = config.auth ?? Auth.bearer(config.apiKey ?? "")
  const catalog = createOpenRouterCatalog({ ...config, endpoint, auth })
  const classifyModelFamily: OpenRouterProvider["classifyModelFamily"] = (model) =>
    classifyModelFamilyFromEvidence({}, [model.upstreamFamily, model.providerModelId, model.displayName])

  const bindModel = (
    providerModelId: string,
    options?: ProviderModelBindOptions,
  ): Effect.Effect<BoundModel<BaseCallOptions>, never, never> => Effect.succeed(
    wrapAsBaseModel(
      createOpenRouterCompatibleSpec({ modelId: providerModelId, endpoint }).bind({
        auth,
        defaults: options?.defaults as Partial<OpenRouterCallOptions> | undefined,
        ...(options?.imagePlaceholders ? { imagePlaceholders: options.imagePlaceholders } : {}),
      }),
    ),
  )

  return {
    catalog,
    provider: {
      id: PROVIDER_ID,
      displayName: "OpenRouter",
      catalog,
      bindModel,
      classifyModelFamily,
    },
  }
}

