import { classifyGenericError } from "../../lib/errors/classify"
import type { ProviderDefinition } from "../../lib/execution/provider-definition"
import { models } from "./models"

export const zaiProvider: ProviderDefinition = {
  id: "zai",
  name: "Z.AI",
  family: "cloud",
  defaultBaseUrl: "https://api.z.ai/api/paas/v4",
  authMethods: [{ type: "api-key", label: "API key", envKeys: ["ZHIPU_API_KEY"] }],
  codecId: "native-chat-completions",
  classifyError: classifyGenericError,
  models,
}
