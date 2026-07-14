import type { ProviderCall, RejectedHttpResponse } from "@magnitudedev/ai"
import { classifyOpenAiCompatibleRejectedResponse } from "../openai-compatible"

export const classifyDeepSeekRejectedResponse = (
  call: ProviderCall,
  response: RejectedHttpResponse,
) => classifyOpenAiCompatibleRejectedResponse("DeepSeek", call, response)

