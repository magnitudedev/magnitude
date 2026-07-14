import type { ModelCatalog } from "@magnitudedev/ai"
import { createModelsDevClient } from "../catalog/models-dev"
import { createOpenAiCompatibleCatalog } from "../openai-compatible"
import type { OpenRouterClientConfig, OpenRouterModelInfo } from "./contract"

export interface OpenRouterCatalogConfig extends OpenRouterClientConfig {
  readonly endpoint: string
  readonly auth: (headers: Headers) => void
}

export function createOpenRouterCatalog(
  config: OpenRouterCatalogConfig,
): ModelCatalog<OpenRouterModelInfo> {
  return createOpenAiCompatibleCatalog<OpenRouterModelInfo>({
    providerId: "openrouter",
    endpoint: config.endpoint,
    auth: config.auth,
    modelsDevProviderId: "openrouter",
    modelsDev: config.modelsDev ?? createModelsDevClient(),
    requireOpenWeights: true,
    requireToolCalls: true,
    toolChoiceModes: ["auto", "none", "required", "named"],
    mapModel: (model) => ({ ...model, providerId: "openrouter" }),
  })
}

