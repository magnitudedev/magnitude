import { classifyGenericError } from "../../lib/errors/classify"
import type { ProviderDefinition } from "../../lib/execution/provider-definition"
import { models } from "./models"

export const fireworksAiProvider: ProviderDefinition = {
  id: "fireworks-ai",
  name: "Fireworks AI",
  family: "cloud",
  defaultBaseUrl: "https://api.fireworks.ai/inference/v1",
  authMethods: [{ type: "api-key", label: "API key", envKeys: ["FIREWORKS_API_KEY"] }],
  codecId: "native-chat-completions",
  classifyError: classifyGenericError,
  models,
}
