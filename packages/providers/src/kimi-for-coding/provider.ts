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
import { createKimiForCodingCatalog, KIMI_FOR_CODING_MODEL_ID } from "./catalog"
import type {
  KimiForCodingCallOptions,
  KimiForCodingClientConfig,
  KimiForCodingModelInfo,
} from "./contract"
import { createKimiForCodingCompatibleSpec, wrapAsBaseModel } from "./models"

export const PROVIDER_ID = "kimi-for-coding" as const
export const DEFAULT_KIMI_FOR_CODING_ENDPOINT = "https://api.kimi.com/coding/v1"

export type KimiForCodingProvider = Provider<KimiForCodingModelInfo>
export type KimiForCodingProviderInstance = OpenAiCompatibleProviderInstance<KimiForCodingModelInfo>

export function createKimiForCodingProvider(
  config: KimiForCodingClientConfig,
): KimiForCodingProviderInstance {
  const endpoint = config.endpoint?.trim().replace(/\/+$/, "") || DEFAULT_KIMI_FOR_CODING_ENDPOINT
  const auth = config.auth ?? Auth.bearer(config.apiKey ?? "")
  const catalog = createKimiForCodingCatalog()
  const classifyModelFamily: KimiForCodingProvider["classifyModelFamily"] = (model) =>
    classifyModelFamilyFromEvidence({}, [model.upstreamFamily, model.displayName, model.providerModelId])

  const bindModel = (
    _providerModelId: string,
    options?: ProviderModelBindOptions,
  ): Effect.Effect<BoundModel<BaseCallOptions>, never, never> => Effect.succeed(
    wrapAsBaseModel(
      createKimiForCodingCompatibleSpec({ modelId: KIMI_FOR_CODING_MODEL_ID, endpoint }).bind({
        auth,
        defaults: options?.defaults as Partial<KimiForCodingCallOptions> | undefined,
        ...(options?.imagePlaceholders ? { imagePlaceholders: options.imagePlaceholders } : {}),
      }),
    ),
  )

  return {
    catalog,
    provider: {
      id: PROVIDER_ID,
      displayName: "Kimi Code",
      catalog,
      bindModel,
      classifyModelFamily,
    },
  }
}
