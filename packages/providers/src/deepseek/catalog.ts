import type { ModelCatalog } from "@magnitudedev/ai"
import { createModelsDevClient } from "../catalog/models-dev"
import { createOpenAiCompatibleCatalog } from "../openai-compatible"
import type { DeepSeekClientConfig, DeepSeekModelInfo } from "./contract"

export interface DeepSeekCatalogConfig extends DeepSeekClientConfig {
  readonly endpoint: string
  readonly auth: (headers: Headers) => void
}

export function createDeepSeekCatalog(
  config: DeepSeekCatalogConfig,
): ModelCatalog<DeepSeekModelInfo> {
  return createOpenAiCompatibleCatalog<DeepSeekModelInfo>({
    providerId: "deepseek",
    endpoint: config.endpoint,
    auth: config.auth,
    modelsDevProviderId: "deepseek",
    modelsDev: config.modelsDev ?? createModelsDevClient(),
    requireToolCalls: true,
    toolChoiceModes: ["auto", "none", "required", "named"],
    mapModel: (model) => ({ ...model, providerId: "deepseek" }),
  })
}

