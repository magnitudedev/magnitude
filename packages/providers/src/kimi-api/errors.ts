import type { ProviderCall, RejectedHttpResponse } from "@magnitudedev/ai"
import { classifyOpenAiCompatibleRejectedResponse } from "../openai-compatible"

export const classifyKimiApiRejectedResponse = (
  call: ProviderCall,
  response: RejectedHttpResponse,
) => classifyOpenAiCompatibleRejectedResponse("Kimi API", call, response)

