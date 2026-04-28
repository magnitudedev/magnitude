import { classifyGenericError } from "../../lib/errors/classify"
import type { ProviderDefinition } from "../../lib/execution/provider-definition"

export const lmstudioProvider: ProviderDefinition = {
  id: "lmstudio",
  name: "LM Studio",
  family: "local",
  defaultBaseUrl: "http://localhost:1234/v1",
  authMethods: [
    { type: "none", label: "Local endpoint" },
    { type: "api-key", label: "Optional API key", envKeys: ["LMSTUDIO_API_KEY"] },
  ],
  codecId: "native-chat-completions",
  classifyError: classifyGenericError,
  models: [],
}
