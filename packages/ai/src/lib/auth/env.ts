import type { ProviderDefinition } from "../execution/provider-definition"
import type { ResolvedAuth } from "./types"

export function resolveEnvAuth(provider: ProviderDefinition): ResolvedAuth | null {
  for (const method of provider.authMethods) {
    if (method.type !== "api-key" || !method.envKeys) continue

    for (const envKey of method.envKeys) {
      const value = process.env[envKey]
      if (value) {
        return {
          _tag: "ApiKeyAuth",
          apiKey: value,
        }
      }
    }
  }

  return null
}
