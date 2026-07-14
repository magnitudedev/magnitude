import type { ModelCatalog } from "@magnitudedev/ai"
import { createModelsDevClient } from "../catalog/models-dev"
import { createOpenAiCompatibleCatalog } from "../openai-compatible"
import type { ZaiClientConfig, ZaiModelInfo } from "./contract"

export interface ZaiCatalogConfig extends ZaiClientConfig {
  readonly endpoint: string
  readonly auth: (headers: Headers) => void
}

function withZaiReasoningControls(model: ZaiModelInfo): ZaiModelInfo {
  const modelId = model.providerModelId.toLowerCase()
  if (modelId.startsWith("glm-5.2")) {
    return { ...model, reasoningEfforts: ["none", "high", "max"] }
  }
  if (/^glm-(?:4\.[5-9]|5(?:$|[.-]))/.test(modelId)) {
    return { ...model, reasoningEfforts: ["none", "high"] }
  }
  return model
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
