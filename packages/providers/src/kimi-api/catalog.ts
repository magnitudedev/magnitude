import type { ModelCatalog } from "@magnitudedev/ai"
import { createModelsDevClient } from "../catalog/models-dev"
import { createOpenAiCompatibleCatalog } from "../openai-compatible"
import type { KimiApiClientConfig, KimiApiModelInfo } from "./contract"

export interface KimiApiCatalogConfig extends KimiApiClientConfig {
  readonly endpoint: string
  readonly auth: (headers: Headers) => void
}

export function createKimiApiCatalog(
  config: KimiApiCatalogConfig,
): ModelCatalog<KimiApiModelInfo> {
  return createOpenAiCompatibleCatalog<KimiApiModelInfo>({
    providerId: "kimi-api",
    endpoint: config.endpoint,
    auth: config.auth,
    modelsDevProviderId: "moonshotai",
    modelsDev: config.modelsDev ?? createModelsDevClient(),
    requireToolCalls: true,
    toolChoiceModes: ["auto", "none"],
    mapModel: (model) => ({ ...model, providerId: "kimi-api" }),
  })
}

