import { Context } from "effect"
import type * as HttpClient from "@effect/platform/HttpClient"
import { Effect } from "effect"
import type { BoundModel, ProviderRejection, WebSearchResult, BalanceQuery, BaseCallOptions, ProviderModelBindOptions, ProviderModel } from "@magnitudedev/ai"
import type { ModelCatalog } from "@magnitudedev/ai"
import { makeFileBackedModelCatalog } from "@magnitudedev/ai"
import {
  createMagnitudeProvider,
  createLlamaCppProvider,
  createDeepSeekProvider,
  createKimiApiProvider,
  createKimiForCodingProvider,
  createOpenRouterProvider,
  createVercelProvider,
  createZaiProvider,
  createZaiCodingPlanProvider,
  createModelsDevClient,
  SUPPORTED_PROVIDER_DEFINITIONS,
  makeProviderRegistry,
  type ConfiguredProviderInstance,
  type MagnitudeProviderInstance,
  type LlamaCppProviderInstance,
  type MagnitudeClientConfig,
  type MagnitudeCallOptions,
  type MagnitudeAdditionalOptions,
  type MagnitudeClientError,
  type MagnitudeModelInfo,
  type WebSearchError,
  type FetchBalanceOptions,
  type BalanceResponse,
} from "@magnitudedev/providers"
import type { ProviderInfo as RegistryProviderInfo } from "@magnitudedev/providers"

// =============================================================================
// Re-exported types with provider-agnostic names
// =============================================================================

export type { ProviderRejection, BaseCallOptions, ProviderModelBindOptions, ProviderModel } from "@magnitudedev/ai"
export type ProviderClientError = MagnitudeClientError
export type ProviderRegistryInfo = RegistryProviderInfo

export interface ProviderClientConfig extends MagnitudeClientConfig {
  readonly providerConnections?: Readonly<Record<string, ProviderConnectionConfig>>
}

export interface ProviderConnectionConfig {
  readonly apiKey?: string
  readonly endpoint?: string
  readonly authSource?: "env" | "file" | "default" | "none"
}

export type {
  MagnitudeModelInfo,
  FetchBalanceOptions,
  BalanceResponse,
  MagnitudeCallOptions,
  MagnitudeAdditionalOptions,
} from "@magnitudedev/providers"
export type { WebSearchResult, BalanceQuery } from "@magnitudedev/ai"
export type { WebSearchError } from "@magnitudedev/providers"
export type { UsagePeriod } from "@magnitudedev/protocol"

// =============================================================================
// Re-exported constants and helpers
// =============================================================================

export {
  classifyMagnitudeRejectedResponse,
  tryParseErrorBody,
  type ParsedMagnitudeApiError,
} from "@magnitudedev/providers"
export {
  SUPPORTED_PROVIDER_DEFINITIONS,
  type SupportedProviderDefinition,
  type ProviderAuthKind,
} from "@magnitudedev/providers"
export { makeFileBackedModelCatalog } from "@magnitudedev/ai"
export {
  createMagnitudeCompatibleSpec,
  DEFAULT_LLAMACPP_ENDPOINT,
} from "@magnitudedev/providers"

// =============================================================================
// Runtime config (provider-specific env vars read behind the boundary)
// =============================================================================

export interface ProviderRuntimeConfig {
  readonly preferProvider?: string
  readonly disableTraits: boolean
}

// =============================================================================
// Provider Client Shape
// =============================================================================

/**
 * The provider client boundary. ONE method to resolve any model from any
 * registered provider. No per-provider methods, no qualified ID parsing.
 */
export interface ProviderClientShape {
  readonly catalog: ModelCatalog<ProviderModel>
  readonly listProviders: Effect.Effect<readonly ProviderRegistryInfo[], never, HttpClient.HttpClient>
  readonly sessionId: string | null
  readonly resolveModel: (
    providerId: string,
    providerModelId: string,
    options?: ProviderModelBindOptions,
  ) => Effect.Effect<BoundModel<BaseCallOptions>, never, never>
  readonly webSearch: (
    query: string,
    schema?: Record<string, unknown>,
  ) => Effect.Effect<WebSearchResult, WebSearchError, HttpClient.HttpClient>
  readonly balance: (
    query?: BalanceQuery,
  ) => Effect.Effect<BalanceResponse, ProviderClientError, HttpClient.HttpClient>
  readonly runtimeConfig: ProviderRuntimeConfig
}

// =============================================================================
// Provider Client Tag
// =============================================================================

/** @effect-expect-leaking HttpClient */
export class ProviderClient extends Context.Tag("ProviderClient")<
  ProviderClient,
  ProviderClientShape
>() {}

// =============================================================================
// Factory
// =============================================================================

export function createProviderClient(config?: ProviderClientConfig): ProviderClientShape {
  const magnitudeInstance: MagnitudeProviderInstance = createMagnitudeProvider(config)
  const sessionId = config?.sessionId ?? null
  const connections = config?.providerConnections ?? {}
  const modelsDev = createModelsDevClient()

  const llamacppInstance: LlamaCppProviderInstance = createLlamaCppProvider({
    endpoint: connections.llamacpp?.endpoint,
    apiKey: connections.llamacpp?.apiKey,
  })

  const configuredProviders: ConfiguredProviderInstance[] = []
  const addCloudProvider = (
    providerId: string,
    create: (connection: ProviderConnectionConfig) => { readonly provider: ConfiguredProviderInstance["provider"] },
  ) => {
    const connection = connections[providerId]
    if (!connection?.apiKey?.trim()) return
    const instance = create(connection)
    configuredProviders.push({
      provider: instance.provider,
      authStatus: { _tag: "authenticated" },
      authKind: "api",
      authSource: connection.authSource ?? "file",
    })
  }

  addCloudProvider("deepseek", (connection) => createDeepSeekProvider({ ...connection, modelsDev }))
  addCloudProvider("kimi-api", (connection) => createKimiApiProvider({ ...connection, modelsDev }))
  addCloudProvider("kimi-for-coding", (connection) => createKimiForCodingProvider(connection))
  addCloudProvider("zai", (connection) => createZaiProvider({ ...connection, modelsDev }))
  addCloudProvider("zai-coding-plan", (connection) => createZaiCodingPlanProvider({ ...connection, modelsDev }))
  addCloudProvider("openrouter", (connection) => createOpenRouterProvider({ ...connection, modelsDev }))
  addCloudProvider("vercel", (connection) => createVercelProvider({ ...connection, modelsDev }))

  const magnitudeConfigured = Boolean(config?.apiKey?.trim() || config?.auth)
  const providerDefinitions = SUPPORTED_PROVIDER_DEFINITIONS.map((definition) => {
    const connection = connections[definition.id]
    const configured = definition.id === "magnitude"
      ? magnitudeConfigured
      : definition.id === "llamacpp"
        ? true
        : Boolean(connection?.apiKey?.trim())
    return {
      id: definition.id,
      displayName: definition.displayName,
      authStatus: definition.id === "llamacpp"
        ? { _tag: "no_auth_required" as const }
        : configured
          ? { _tag: "authenticated" as const }
          : { _tag: "not_configured" as const, reason: "API key is not configured" },
      authKind: definition.authKind,
      authSource: definition.id === "llamacpp"
        ? connections.llamacpp?.authSource ?? "default"
        : definition.id === "magnitude"
          ? connections.magnitude?.authSource ?? (magnitudeConfigured ? "file" as const : "none" as const)
          : connection?.authSource ?? "none",
    }
  })

  const registry = makeProviderRegistry({
    magnitude: magnitudeConfigured ? magnitudeInstance : null,
    configuredProviders,
    discoverableProviders: [llamacppInstance],
    providerDefinitions,
  })

  return {
    catalog: registry.aggregatedCatalog,
    listProviders: registry.listProviders,
    sessionId,
    resolveModel: (providerId, providerModelId, options) =>
      registry.resolveModel(providerId, providerModelId, options),
    webSearch: magnitudeInstance.provider.webSearch,
    balance: magnitudeInstance.provider.balance,
    runtimeConfig: {
      preferProvider: process.env.MAGNITUDE_PREFER_PROVIDER || undefined,
      disableTraits: !!process.env.MAGNITUDE_DISABLE_TRAITS,
    },
  }
}
