import { classifyGenericError } from "../../lib/errors/classify"
import type { ProviderDefinition } from "../../lib/execution/provider-definition"

export const openAiCompatibleLocalProvider: ProviderDefinition = {
  id: "openai-compatible-local",
  name: "OpenAI-compatible local",
  family: "local",
  authMethods: [
    { type: "none", label: "Local endpoint" },
    { type: "api-key", label: "Optional API key", envKeys: ["OPENAI_COMPAT_LOCAL_API_KEY"] },
  ],
  codecId: "native-chat-completions",
  classifyError: classifyGenericError,
  models: [],
}
