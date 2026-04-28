import { classifyGenericError } from "../../lib/errors/classify"
import type { ProviderDefinition } from "../../lib/execution/provider-definition"
import { models } from "./models"

export const anthropicProvider: ProviderDefinition = {
  id: "anthropic",
  name: "Anthropic",
  family: "cloud",
  authMethods: [
    { type: "oauth-pkce", label: "Claude Pro/Max subscription" },
    { type: "api-key", label: "API key", envKeys: ["ANTHROPIC_API_KEY"] },
  ],
  codecId: "native-chat-completions",
  classifyError: classifyGenericError,
  models,
}
