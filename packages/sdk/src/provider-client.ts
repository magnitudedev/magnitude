import { Context } from "effect"
import type * as HttpClient from "@effect/platform/HttpClient"
import { Effect } from "effect"
import { ModelCatalogError } from "@magnitudedev/ai"
import type { BoundModel, ProviderRejection, WebSearchResult, BalanceQuery, BaseCallOptions, ProviderModelBindOptions, ProviderModel } from "@magnitudedev/ai"
import type { ModelCatalog } from "@magnitudedev/ai"
import { makeFileBackedModelCatalog } from "@magnitudedev/ai"
import {
  createMagnitudeProvider,
  createLlamaCppProvider,
  makeFixedEndpointBackend,
  makeProviderRegistry,
  type MagnitudeProviderInstance,
  type LlamaCppProviderComponents,
  type LlamaCppProviderBackend,
  LlamaCppProviderBackendError,
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
export { ModelCatalogError } from "@magnitudedev/ai"
export type ProviderClientError = MagnitudeClientError
export type ProviderRegistryInfo = RegistryProviderInfo

export interface ProviderClientConfig extends MagnitudeClientConfig {
  readonly llamacppEndpoint?: string
  readonly llamacppApiKey?: string
  readonly llamacppBackend?: LlamaCppProviderBackend
}

export type {
  MagnitudeModelInfo,
  FetchBalanceOptions,
  BalanceResponse,
  MagnitudeCallOptions,
  MagnitudeAdditionalOptions,
} from "@magnitudedev/providers"
export type { LlamaCppProviderBackend } from "@magnitudedev/providers"
export { LlamaCppProviderBackendError } from "@magnitudedev/providers"
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
export { makeFileBackedModelCatalog } from "@magnitudedev/ai"
export {
  createMagnitudeCompatibleSpec,
  DEFAULT_LLAMACPP_ENDPOINT,
  MagnitudeModelListResponseSchema,
  toMagnitudeModelInfo,
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

  const llamacpp: LlamaCppProviderComponents = createLlamaCppProvider(
    config?.llamacppBackend ?? makeFixedEndpointBackend({
      endpoint: config?.llamacppEndpoint,
      apiKey: config?.llamacppApiKey,
    }),
  )

  const registry = makeProviderRegistry({
    magnitude: magnitudeInstance,
    discoverableProviders: [llamacpp],
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
