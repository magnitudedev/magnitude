import type { ModelSpec } from "@magnitudedev/ai"
import {
  createOpenAiCompatibleSpec,
  wrapOpenAiCompatibleAsBaseModel,
} from "../openai-compatible"
import { classifyDeepSeekRejectedResponse } from "./errors"
import type { DeepSeekCallOptions } from "./contract"

export interface DeepSeekCompatibleSpecConfig {
  readonly modelId: string
  readonly endpoint: string
}

export function createDeepSeekCompatibleSpec(
  config: DeepSeekCompatibleSpecConfig,
): ModelSpec<DeepSeekCallOptions> {
  return createOpenAiCompatibleSpec({
    ...config,
    providerName: "DeepSeek",
    reasoningRequestMode: "thinking-effort",
    classifyRejectedResponse: classifyDeepSeekRejectedResponse,
  })
}

export { wrapOpenAiCompatibleAsBaseModel as wrapAsBaseModel }
