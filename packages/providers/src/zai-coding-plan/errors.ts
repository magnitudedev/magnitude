import type { ProviderCall, RejectedHttpResponse } from "@magnitudedev/ai"
import { classifyOpenAiCompatibleRejectedResponse } from "../openai-compatible"

export const classifyZaiCodingPlanRejectedResponse = (
  call: ProviderCall,
  response: RejectedHttpResponse,
) => classifyOpenAiCompatibleRejectedResponse("GLM Coding Plan", call, response)
