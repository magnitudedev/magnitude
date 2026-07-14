import type { ModelSpec } from "@magnitudedev/ai"
import { createOpenAiCompatibleSpec, wrapOpenAiCompatibleAsBaseModel } from "../openai-compatible"
import type { KimiForCodingCallOptions } from "./contract"
import { classifyKimiForCodingRejectedResponse } from "./errors"

export interface KimiForCodingCompatibleSpecConfig {
  readonly modelId: string
  readonly endpoint: string
}

export function createKimiForCodingCompatibleSpec(
  config: KimiForCodingCompatibleSpecConfig,
): ModelSpec<KimiForCodingCallOptions> {
  return createOpenAiCompatibleSpec({
    ...config,
    providerName: "Kimi Code",
    reasoningRequestMode: "kimi",
    classifyRejectedResponse: classifyKimiForCodingRejectedResponse,
  })
}

export { wrapOpenAiCompatibleAsBaseModel as wrapAsBaseModel }
