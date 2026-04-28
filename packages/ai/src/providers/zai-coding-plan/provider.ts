import { classifyGenericError } from "../../lib/errors/classify"
import type { ProviderDefinition } from "../../lib/execution/provider-definition"
import { models } from "./models"

export const zaiCodingPlanProvider: ProviderDefinition = {
  id: "zai-coding-plan",
  name: "Z.AI Coding Plan",
  family: "cloud",
  defaultBaseUrl: "https://api.z.ai/api/coding/paas/v4",
  authMethods: [{ type: "api-key", label: "API key", envKeys: ["ZHIPU_API_KEY"] }],
  codecId: "native-chat-completions",
  classifyError: classifyGenericError,
  models,
}
