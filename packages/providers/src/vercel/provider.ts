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
import { createVercelCatalog } from "./catalog"
import type { VercelCallOptions, VercelClientConfig, VercelModelInfo } from "./contract"
import { createVercelCompatibleSpec, wrapAsBaseModel } from "./models"

export const PROVIDER_ID = "vercel" as const
export const DEFAULT_VERCEL_ENDPOINT = "https://ai-gateway.vercel.sh/v1"

export type VercelProvider = Provider<VercelModelInfo>
export type VercelProviderInstance = OpenAiCompatibleProviderInstance<VercelModelInfo>

export function createVercelProvider(config: VercelClientConfig): VercelProviderInstance {
  const endpoint = config.endpoint?.trim().replace(/\/+$/, "") || DEFAULT_VERCEL_ENDPOINT
  const auth = config.auth ?? Auth.bearer(config.apiKey ?? "")
  const catalog = createVercelCatalog({ ...config, endpoint, auth })
  const classifyModelFamily: VercelProvider["classifyModelFamily"] = (model) =>
    classifyModelFamilyFromEvidence({}, [model.providerModelId, model.displayName, model.upstreamFamily])

  const bindModel = (
    providerModelId: string,
    options?: ProviderModelBindOptions,
  ): Effect.Effect<BoundModel<BaseCallOptions>, never, never> => Effect.succeed(
    wrapAsBaseModel(
      createVercelCompatibleSpec({ modelId: providerModelId, endpoint }).bind({
        auth,
        defaults: options?.defaults as Partial<VercelCallOptions> | undefined,
        ...(options?.imagePlaceholders ? { imagePlaceholders: options.imagePlaceholders } : {}),
      }),
    ),
  )

  return {
    catalog,
    provider: {
      id: PROVIDER_ID,
      displayName: "Vercel AI Gateway",
      catalog,
      bindModel,
      classifyModelFamily,
    },
  }
}
