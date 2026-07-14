import type { ModelSpec } from "@magnitudedev/ai"
import { createOpenAiCompatibleSpec, wrapOpenAiCompatibleAsBaseModel } from "../openai-compatible"
import type { ZaiCallOptions } from "./contract"
import { classifyZaiRejectedResponse } from "./errors"

export interface ZaiCompatibleSpecConfig {
  readonly modelId: string
  readonly endpoint: string
}

export function createZaiCompatibleSpec(config: ZaiCompatibleSpecConfig): ModelSpec<ZaiCallOptions> {
  return createOpenAiCompatibleSpec({
    ...config,
    providerName: "Z.AI API",
    reasoningRequestMode: config.modelId.toLowerCase().startsWith("glm-5.2")
      ? "thinking-effort"
      : "thinking-toggle",
    classifyRejectedResponse: classifyZaiRejectedResponse,
  })
}

export { wrapOpenAiCompatibleAsBaseModel as wrapAsBaseModel }
