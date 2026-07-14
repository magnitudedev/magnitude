import type { ProviderCall, RejectedHttpResponse } from "@magnitudedev/ai"
import { classifyOpenAiCompatibleRejectedResponse } from "../openai-compatible"

export const classifyZaiRejectedResponse = (call: ProviderCall, response: RejectedHttpResponse) =>
  classifyOpenAiCompatibleRejectedResponse("Z.AI API", call, response)

