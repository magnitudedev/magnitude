import { classifyGenericError } from "../../lib/errors/classify"
import type { ProviderDefinition } from "../../lib/execution/provider-definition"
import { models } from "./models"

export const vercelProvider: ProviderDefinition = {
  id: "vercel",
  name: "Vercel AI Gateway",
  family: "cloud",
  defaultBaseUrl: "https://ai-gateway.vercel.sh/v1",
  authMethods: [{ type: "api-key", label: "API key", envKeys: ["AI_GATEWAY_API_KEY"] }],
  codecId: "native-chat-completions",
  classifyError: classifyGenericError,
  models,
}
