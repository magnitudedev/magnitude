import { classifyGenericError } from "../../lib/errors/classify"
import type { ProviderDefinition } from "../../lib/execution/provider-definition"
import { models } from "./models"

export const magnitudeProvider: ProviderDefinition = {
  id: "magnitude",
  name: "Magnitude",
  family: "cloud",
  defaultBaseUrl: "https://app.magnitude.dev/api/v1",
  authMethods: [{ type: "api-key", label: "API key", envKeys: ["MAGNITUDE_API_KEY"] }],
  codecId: "native-chat-completions",
  classifyError: classifyGenericError,
  models,
}
