import { classifyGenericError } from "../../lib/errors/classify"
import type { ProviderDefinition } from "../../lib/execution/provider-definition"
import { models } from "./models"

export const openrouterProvider: ProviderDefinition = {
  id: "openrouter",
  name: "OpenRouter",
  family: "cloud",
  defaultBaseUrl: "https://openrouter.ai/api/v1",
  authMethods: [{ type: "api-key", label: "API key", envKeys: ["OPENROUTER_API_KEY"] }],
  codecId: "native-chat-completions",
  classifyError: classifyGenericError,
  models,
}
