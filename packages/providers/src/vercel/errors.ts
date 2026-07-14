import type { ProviderCall, RejectedHttpResponse } from "@magnitudedev/ai"
import { classifyOpenAiCompatibleRejectedResponse } from "../openai-compatible"

export const classifyVercelRejectedResponse = (
  call: ProviderCall,
  response: RejectedHttpResponse,
) => classifyOpenAiCompatibleRejectedResponse("Vercel AI Gateway", call, response)

