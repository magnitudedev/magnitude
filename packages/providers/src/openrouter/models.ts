import type { ModelSpec } from "@magnitudedev/ai"
import { createOpenAiCompatibleSpec, wrapOpenAiCompatibleAsBaseModel } from "../openai-compatible"
import type { OpenRouterCallOptions } from "./contract"
import { classifyOpenRouterRejectedResponse } from "./errors"

export interface OpenRouterCompatibleSpecConfig {
  readonly modelId: string
  readonly endpoint: string
}

export function createOpenRouterCompatibleSpec(
  config: OpenRouterCompatibleSpecConfig,
): ModelSpec<OpenRouterCallOptions> {
  return createOpenAiCompatibleSpec({
    ...config,
    providerName: "OpenRouter",
    reasoningField: "reasoning",
    preserveReasoningDetails: true,
    reasoningRequestMode: "openrouter",
    classifyRejectedResponse: classifyOpenRouterRejectedResponse,
  })
}

export { wrapOpenAiCompatibleAsBaseModel as wrapAsBaseModel }
