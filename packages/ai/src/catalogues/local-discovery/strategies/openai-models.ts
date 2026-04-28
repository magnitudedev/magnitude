import { Effect } from "effect"
import type { ProviderModel } from "../../../lib/model/provider-model"
import { LocalDiscoveryError } from "./errors"

interface OpenAIModelsResponse {
  readonly data?: ReadonlyArray<{
    readonly id?: string
    readonly name?: string
    readonly capabilities?: readonly string[]
  }>
}

const ZERO_COSTS = {
  inputPerM: 0,
  outputPerM: 0,
  cacheReadPerM: null,
  cacheWritePerM: null,
} as const

function normalizeBaseUrl(baseUrl: string): string {
  return baseUrl.replace(/\/+$/, "")
}

function openAIModelsUrl(baseUrl: string): string {
  const normalized = normalizeBaseUrl(baseUrl)
  return normalized.endsWith("/v1") ? `${normalized}/models` : `${normalized}/v1/models`
}

async function fetchJson<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init)
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} for ${url}`)
  }
  return (await response.json()) as T
}

function toModel(
  id: string,
  providerId: string,
  providerName: string,
  name?: string,
  maxContextTokens: number | null = null,
  supportsVision = false,
): ProviderModel {
  const now = new Date().toISOString()
  return {
    id,
    providerId,
    providerName,
    canonicalModelId: null,
    name: name || id,
    contextWindow: maxContextTokens ?? 200_000,
    maxContextTokens,
    maxOutputTokens: null,
    supportsToolCalls: true,
    supportsReasoning: false,
    supportsVision,
    costs: ZERO_COSTS,
    releaseDate: now,
    discovery: {
      primarySource: "local",
      fetchedAt: now,
    },
  }
}

export function discoverOpenAIModels(
  providerId: string,
  providerName: string,
  baseUrl: string,
): Effect.Effect<readonly ProviderModel[], LocalDiscoveryError> {
  return Effect.tryPromise({
    try: async () => {
      const payload = await fetchJson<OpenAIModelsResponse>(openAIModelsUrl(baseUrl))
      return (payload.data ?? [])
        .map((model) => {
          const id = model.id?.trim()
          if (!id) {
            return null
          }
          const supportsVision = model.capabilities?.includes("multimodal") ?? false
          return toModel(id, providerId, providerName, model.name?.trim() || id, null, supportsVision)
        })
        .filter((model): model is ProviderModel => model !== null)
    },
    catch: (error) => new LocalDiscoveryError({ message: "discovery failed", cause: error }),
  })
}
