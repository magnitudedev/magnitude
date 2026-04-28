import { classifyGenericError } from "../../lib/errors/classify"
import type { ProviderDefinition } from "../../lib/execution/provider-definition"
import { models } from "./models"

export const deepseekProvider: ProviderDefinition = {
  id: "deepseek",
  name: "DeepSeek",
  family: "cloud",
  defaultBaseUrl: "https://api.deepseek.com/v1",
  authMethods: [{ type: "api-key", label: "API key", envKeys: ["DEEPSEEK_API_KEY"] }],
  codecId: "native-chat-completions",
  classifyError: classifyGenericError,
  models,
}
