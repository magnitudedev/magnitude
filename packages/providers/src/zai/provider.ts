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
import { createZaiCatalog } from "./catalog"
import type { ZaiCallOptions, ZaiClientConfig, ZaiModelInfo } from "./contract"
import { createZaiCompatibleSpec, wrapAsBaseModel } from "./models"

export const PROVIDER_ID = "zai" as const
export const DEFAULT_ZAI_ENDPOINT = "https://api.z.ai/api/paas/v4"

export type ZaiProvider = Provider<ZaiModelInfo>
export type ZaiProviderInstance = OpenAiCompatibleProviderInstance<ZaiModelInfo>

export function createZaiProvider(config: ZaiClientConfig): ZaiProviderInstance {
  const endpoint = config.endpoint?.trim().replace(/\/+$/, "") || DEFAULT_ZAI_ENDPOINT
  const auth = config.auth ?? Auth.bearer(config.apiKey ?? "")
  const catalog = createZaiCatalog({ ...config, endpoint, auth })
  const classifyModelFamily: ZaiProvider["classifyModelFamily"] = (model) =>
    classifyModelFamilyFromEvidence({}, [model.providerModelId, model.displayName, model.upstreamFamily])

  const bindModel = (
    providerModelId: string,
    options?: ProviderModelBindOptions,
  ): Effect.Effect<BoundModel<BaseCallOptions>, never, never> => Effect.succeed(
    wrapAsBaseModel(
      createZaiCompatibleSpec({ modelId: providerModelId, endpoint }).bind({
        auth,
        defaults: options?.defaults as Partial<ZaiCallOptions> | undefined,
        ...(options?.imagePlaceholders ? { imagePlaceholders: options.imagePlaceholders } : {}),
      }),
    ),
  )

  return {
    catalog,
    provider: {
      id: PROVIDER_ID,
      displayName: "Z.AI API",
      catalog,
      bindModel,
      classifyModelFamily,
    },
  }
}
