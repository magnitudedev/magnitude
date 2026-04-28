import { classifyGenericError } from "../../lib/errors/classify"
import type { ProviderDefinition } from "../../lib/execution/provider-definition"
import { kimiForCodingModels } from "./models"

export const kimiForCodingProvider: ProviderDefinition = {
  id: "kimi-for-coding",
  name: "Kimi for Coding",
  family: "cloud",
  defaultBaseUrl: "https://api.kimi.com/coding",
  authMethods: [{ type: "api-key", label: "API key", envKeys: ["KIMI_API_KEY"] }],
  codecId: "native-chat-completions",
  classifyError: classifyGenericError,
  models: kimiForCodingModels,
}
