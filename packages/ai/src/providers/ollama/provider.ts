import { classifyGenericError } from "../../lib/errors/classify"
import type { ProviderDefinition } from "../../lib/execution/provider-definition"
import { models } from "./models"

export const ollamaProvider: ProviderDefinition = {
  id: "ollama",
  name: "Ollama",
  family: "local",
  defaultBaseUrl: "http://localhost:11434/v1",
  authMethods: [
    { type: "none", label: "Local endpoint" },
    { type: "api-key", label: "Optional API key", envKeys: ["OLLAMA_API_KEY"] },
  ],
  codecId: "native-chat-completions",
  classifyError: classifyGenericError,
  models,
}
