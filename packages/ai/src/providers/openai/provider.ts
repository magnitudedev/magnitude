import { classifyGenericError } from "../../lib/errors/classify"
import type { ProviderDefinition } from "../../lib/execution/provider-definition"
import { models } from "./models"

export const openaiProvider: ProviderDefinition = {
  id: "openai",
  name: "OpenAI",
  family: "cloud",
  defaultBaseUrl: "https://api.openai.com/v1",
  authMethods: [
    { type: "oauth-browser", label: "ChatGPT Pro/Plus (browser)" },
    { type: "oauth-device", label: "ChatGPT Pro/Plus (headless)" },
    { type: "api-key", label: "API key", envKeys: ["OPENAI_API_KEY"] },
  ],
  codecId: "native-chat-completions",
  classifyError: classifyGenericError,
  models,
}
