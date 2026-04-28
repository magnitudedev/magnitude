import { classifyGenericError } from "../../lib/errors/classify"
import type { ProviderDefinition } from "../../lib/execution/provider-definition"
import { models } from "./models"

export const llamaCppProvider: ProviderDefinition = {
  id: "llama.cpp",
  name: "llama.cpp",
  family: "local",
  defaultBaseUrl: "http://localhost:8080",
  authMethods: [
    { type: "none", label: "Local endpoint" },
    { type: "api-key", label: "Optional API key", envKeys: ["LLAMA_CPP_API_KEY"] },
  ],
  codecId: "native-chat-completions",
  classifyError: classifyGenericError,
  models,
}
