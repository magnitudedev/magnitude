import type { Codec } from "../codec/codec"
import { nativeChatCompletionsCodec } from "../codec/native-chat-completions"
import { openAIChatCompletionsDriver } from "../driver/openai-chat-completions"
import type { Driver } from "../driver/driver"
import type { ProviderModel } from "../model/provider-model"
import type { ResolvedAuth } from "../auth/types"
import type { BoundModel } from "./bound-model"
import type { ProviderDefinition } from "./provider-definition"

const codecRegistry: Record<string, Codec> = {
  "native-chat-completions": nativeChatCompletionsCodec,
}

const driverRegistry: Record<string, Driver> = {
  "native-chat-completions": openAIChatCompletionsDriver,
}

function getAuthToken(auth: ResolvedAuth): string {
  switch (auth._tag) {
    case "ApiKeyAuth":
      return auth.apiKey
    case "OAuthAuth":
      return auth.accessToken
    case "NoAuth":
      return ""
  }
}

export function bindModel(
  provider: ProviderDefinition,
  model: ProviderModel,
  auth: ResolvedAuth,
): BoundModel {
  const codec = codecRegistry[provider.codecId]
  if (!codec) {
    throw new Error(`Unknown codec: ${provider.codecId}`)
  }

  const driver = driverRegistry[provider.codecId]
  if (!driver) {
    throw new Error(`No driver registered for codec: ${provider.codecId}`)
  }

  return {
    provider,
    model,
    codec,
    driver,
    authToken: getAuthToken(auth),
    endpoint: provider.defaultBaseUrl ?? "",
  }
}
