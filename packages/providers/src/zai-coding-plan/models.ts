import type { ModelSpec } from "@magnitudedev/ai"
import { createOpenAiCompatibleSpec, wrapOpenAiCompatibleAsBaseModel } from "../openai-compatible"
import type { ZaiCodingPlanCallOptions } from "./contract"
import { classifyZaiCodingPlanRejectedResponse } from "./errors"

export interface ZaiCodingPlanCompatibleSpecConfig {
  readonly modelId: string
  readonly endpoint: string
}

export function createZaiCodingPlanCompatibleSpec(
  config: ZaiCodingPlanCompatibleSpecConfig,
): ModelSpec<ZaiCodingPlanCallOptions> {
  return createOpenAiCompatibleSpec({
    ...config,
    providerName: "GLM Coding Plan",
    reasoningRequestMode: config.modelId.toLowerCase().startsWith("glm-5.2")
      ? "thinking-effort"
      : "thinking-toggle",
    classifyRejectedResponse: classifyZaiCodingPlanRejectedResponse,
  })
}

export { wrapOpenAiCompatibleAsBaseModel as wrapAsBaseModel }
