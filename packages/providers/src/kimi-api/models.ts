import type { ModelSpec } from "@magnitudedev/ai"
import { createOpenAiCompatibleSpec, wrapOpenAiCompatibleAsBaseModel } from "../openai-compatible"
import type { KimiApiCallOptions } from "./contract"
import { classifyKimiApiRejectedResponse } from "./errors"

export interface KimiApiCompatibleSpecConfig {
  readonly modelId: string
  readonly endpoint: string
}

export function createKimiApiCompatibleSpec(
  config: KimiApiCompatibleSpecConfig,
): ModelSpec<KimiApiCallOptions> {
  return createOpenAiCompatibleSpec({
    ...config,
    providerName: "Kimi API",
    reasoningRequestMode: "thinking-toggle",
    classifyRejectedResponse: classifyKimiApiRejectedResponse,
  })
}

export { wrapOpenAiCompatibleAsBaseModel as wrapAsBaseModel }
