import type { ModelCatalog } from "@magnitudedev/ai"
import { createModelsDevClient } from "../catalog/models-dev"
import { createOpenAiCompatibleCatalog } from "../openai-compatible"
import type { ZaiClientConfig, ZaiModelInfo } from "./contract"

export interface ZaiCatalogConfig extends ZaiClientConfig {
  readonly endpoint: string
  readonly auth: (headers: Headers) => void
}

function withZaiReasoningControls(model: ZaiModelInfo): ZaiModelInfo {
  return model.providerModelId.toLowerCase().startsWith("glm-5.2")
    ? { ...model, reasoningEfforts: ["none", "high", "max"] }
    : model
}

export function createZaiCatalog(config: ZaiCatalogConfig): ModelCatalog<ZaiModelInfo> {
  return createOpenAiCompatibleCatalog<ZaiModelInfo>({
    providerId: "zai",
    endpoint: config.endpoint,
    auth: config.auth,
    modelsDevProviderId: "zai",
    modelsDev: config.modelsDev ?? createModelsDevClient(),
    liveCatalogFallback: "unsupported_only",
    requireToolCalls: true,
    toolChoiceModes: ["auto", "none"],
    mapModel: (model) => withZaiReasoningControls({ ...model, providerId: "zai" }),
  })
}
