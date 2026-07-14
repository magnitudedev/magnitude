import type { ModelCatalog } from "@magnitudedev/ai"
import { createModelsDevClient } from "../catalog/models-dev"
import { createOpenAiCompatibleCatalog } from "../openai-compatible"
import type { VercelClientConfig, VercelModelInfo } from "./contract"

export interface VercelCatalogConfig extends VercelClientConfig {
  readonly endpoint: string
  readonly auth: (headers: Headers) => void
}

export function createVercelCatalog(config: VercelCatalogConfig): ModelCatalog<VercelModelInfo> {
  return createOpenAiCompatibleCatalog<VercelModelInfo>({
    providerId: "vercel",
    endpoint: config.endpoint,
    auth: config.auth,
    modelsDevProviderId: "vercel",
    modelsDev: config.modelsDev ?? createModelsDevClient(),
    requireOpenWeights: true,
    requireToolCalls: true,
    toolChoiceModes: ["auto", "none", "required", "named"],
    mapModel: (model) => ({ ...model, providerId: "vercel" }),
  })
}

