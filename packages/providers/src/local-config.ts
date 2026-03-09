/**
 * Runtime config for the local provider (Ollama, LM Studio, etc.).
 * Separate file to avoid circular dependency between provider-state and client-registry-builder.
 */

let localBaseUrl: string | null = null
let localModelId: string | null = null

/**
 * Set the local provider's base URL and model ID override.
 */
export function setLocalProviderConfig(baseUrl: string, modelId: string): void {
  localBaseUrl = baseUrl
  localModelId = modelId
}

/**
 * Get the local provider's config overrides.
 */
export function getLocalProviderConfig(): { baseUrl: string | null; modelId: string | null } {
  return { baseUrl: localBaseUrl, modelId: localModelId }
}
