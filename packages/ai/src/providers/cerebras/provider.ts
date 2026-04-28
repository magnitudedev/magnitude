import { classifyGenericError } from "../../lib/errors/classify"
import type { ProviderDefinition } from "../../lib/execution/provider-definition"
import { models } from "./models"

export const cerebrasProvider: ProviderDefinition = {
  id: "cerebras",
  name: "Cerebras",
  family: "cloud",
  defaultBaseUrl: "https://api.cerebras.ai/v1",
  authMethods: [{ type: "api-key", label: "API key", envKeys: ["CEREBRAS_API_KEY"] }],
  codecId: "native-chat-completions",
  classifyError: classifyGenericError,
  models,
}
