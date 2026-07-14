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
import { createZaiCodingPlanCatalog } from "./catalog"
import type {
  ZaiCodingPlanCallOptions,
  ZaiCodingPlanClientConfig,
  ZaiCodingPlanModelInfo,
} from "./contract"
import { createZaiCodingPlanCompatibleSpec, wrapAsBaseModel } from "./models"

export const PROVIDER_ID = "zai-coding-plan" as const
export const DEFAULT_ZAI_CODING_PLAN_ENDPOINT = "https://api.z.ai/api/coding/paas/v4"

export type ZaiCodingPlanProvider = Provider<ZaiCodingPlanModelInfo>
export type ZaiCodingPlanProviderInstance = OpenAiCompatibleProviderInstance<ZaiCodingPlanModelInfo>

export function createZaiCodingPlanProvider(
  config: ZaiCodingPlanClientConfig,
): ZaiCodingPlanProviderInstance {
  const endpoint = config.endpoint?.trim().replace(/\/+$/, "") || DEFAULT_ZAI_CODING_PLAN_ENDPOINT
  const auth = config.auth ?? Auth.bearer(config.apiKey ?? "")
  const catalog = createZaiCodingPlanCatalog({ ...config, endpoint, auth })
  const classifyModelFamily: ZaiCodingPlanProvider["classifyModelFamily"] = (model) =>
    classifyModelFamilyFromEvidence({}, [model.upstreamFamily, model.providerModelId, model.displayName])

  const bindModel = (
    providerModelId: string,
    options?: ProviderModelBindOptions,
  ): Effect.Effect<BoundModel<BaseCallOptions>, never, never> => Effect.succeed(
    wrapAsBaseModel(
      createZaiCodingPlanCompatibleSpec({ modelId: providerModelId, endpoint }).bind({
        auth,
        defaults: options?.defaults as Partial<ZaiCodingPlanCallOptions> | undefined,
        ...(options?.imagePlaceholders ? { imagePlaceholders: options.imagePlaceholders } : {}),
      }),
    ),
  )

  return {
    catalog,
    provider: {
      id: PROVIDER_ID,
      displayName: "GLM Coding Plan",
      catalog,
      bindModel,
      classifyModelFamily,
    },
  }
}
