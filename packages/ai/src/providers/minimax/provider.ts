import { classifyGenericError } from "../../lib/errors/classify"
import type { ProviderDefinition } from "../../lib/execution/provider-definition"
import { models } from "./models"

export const minimaxProvider: ProviderDefinition = {
  id: "minimax",
  name: "MiniMax",
  family: "cloud",
  defaultBaseUrl: "https://api.minimax.io/anthropic",
  authMethods: [{ type: "api-key", label: "API key", envKeys: ["MINIMAX_API_KEY"] }],
  codecId: "native-chat-completions",
  classifyError: classifyGenericError,
  models,
}
