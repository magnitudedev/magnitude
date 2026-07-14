import type { ModelCatalog } from "@magnitudedev/ai"
import { createModelsDevClient } from "../catalog/models-dev"
import { createOpenAiCompatibleCatalog } from "../openai-compatible"
import type { ZaiCodingPlanClientConfig, ZaiCodingPlanModelInfo } from "./contract"

export interface ZaiCodingPlanCatalogConfig extends ZaiCodingPlanClientConfig {
  readonly endpoint: string
  readonly auth: (headers: Headers) => void
}

function withZaiReasoningControls(model: ZaiCodingPlanModelInfo): ZaiCodingPlanModelInfo {
  return model.providerModelId.toLowerCase().startsWith("glm-5.2")
    ? { ...model, reasoningEfforts: ["none", "high", "max"] }
    : model
}

export function createZaiCodingPlanCatalog(
  config: ZaiCodingPlanCatalogConfig,
): ModelCatalog<ZaiCodingPlanModelInfo> {
  return createOpenAiCompatibleCatalog<ZaiCodingPlanModelInfo>({
    providerId: "zai-coding-plan",
    endpoint: config.endpoint,
    auth: config.auth,
    modelsDevProviderId: "zai-coding-plan",
    modelsDev: config.modelsDev ?? createModelsDevClient(),
    liveCatalogFallback: "unsupported_only",
    requireToolCalls: true,
    toolChoiceModes: ["auto", "none"],
    mapModel: (model) => withZaiReasoningControls({
      ...model,
      providerId: "zai-coding-plan",
    }),
  })
}
