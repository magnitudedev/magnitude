import { classifyGenericError } from "../../lib/errors/classify"
import type { ProviderDefinition } from "../../lib/execution/provider-definition"
import { moonshotAiModels } from "./models"

export const moonshotAiProvider: ProviderDefinition = {
  id: "moonshotai",
  name: "Moonshot AI",
  family: "cloud",
  defaultBaseUrl: "https://api.moonshot.ai/v1",
  authMethods: [{ type: "api-key", label: "API key", envKeys: ["MOONSHOT_API_KEY"] }],
  codecId: "native-chat-completions",
  classifyError: classifyGenericError,
  models: moonshotAiModels,
}
