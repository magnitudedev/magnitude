import type { ProviderCall, RejectedHttpResponse } from "@magnitudedev/ai"
import { classifyOpenAiCompatibleRejectedResponse } from "../openai-compatible"

export const classifyOpenRouterRejectedResponse = (
  call: ProviderCall,
  response: RejectedHttpResponse,
) => classifyOpenAiCompatibleRejectedResponse("OpenRouter", call, response)

