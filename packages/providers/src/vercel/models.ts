import type { ModelSpec } from "@magnitudedev/ai"
import { createOpenAiCompatibleSpec, wrapOpenAiCompatibleAsBaseModel } from "../openai-compatible"
import type { VercelCallOptions } from "./contract"
import { classifyVercelRejectedResponse } from "./errors"

export interface VercelCompatibleSpecConfig {
  readonly modelId: string
  readonly endpoint: string
}

export function createVercelCompatibleSpec(
  config: VercelCompatibleSpecConfig,
): ModelSpec<VercelCallOptions> {
  return createOpenAiCompatibleSpec({
    ...config,
    providerName: "Vercel AI Gateway",
    preserveReasoningDetails: true,
    reasoningRequestMode: "openai",
    classifyRejectedResponse: classifyVercelRejectedResponse,
  })
}

export { wrapOpenAiCompatibleAsBaseModel as wrapAsBaseModel }
