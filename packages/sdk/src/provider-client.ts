import { Context } from "effect"
import type * as HttpClient from "@effect/platform/HttpClient"
import { Effect } from "effect"
import { ModelCatalogError } from "@magnitudedev/ai"
import type {
  BaseCallOptions,
  BoundModel,
  ModelDiscoveryOperationId,
  ModelPropertyDiscoveryError,
  ModelPropertyDiscoveryRequest,
  ProviderId,
  ProviderModel,
  ProviderModelBindOptions,
  ProviderModelId,
  ProviderRejection,
  RequestAttribution,
  UsageQuery,
  WebSearchResult,
} from "@magnitudedev/ai"
import type { ModelCatalog } from "@magnitudedev/ai"
import { makeFileBackedModelCatalog } from "@magnitudedev/ai"
import {
  createMagnitudeProvider,
  makeProviderRegistry,
  type DiscoverableProviderInstance,
  type MagnitudeProviderInstance,
  type MagnitudeClientConfig,
  type MagnitudeCallOptions,
  type MagnitudeAdditionalOptions,
  type MagnitudeClientError,
  type MagnitudeModelInfo,
  type WebSearchError,
  type FetchUsageOptions,
  type CloudUsageResponse,
  type ProviderCatalogOutcome,
} from "@magnitudedev/providers"
import type { ProviderInfo as RegistryProviderInfo } from "@magnitudedev/providers"

// =============================================================================
// Re-exported types with provider-agnostic names
// =============================================================================

export type {
  ProviderRejection,
  BaseCallOptions,
  ProviderModelBindOptions,
  ProviderModel,
  ProviderModelAvailability,
  ProviderModelDisabledReason,
  ProviderId,
  ProviderModelId,
  ModelFamilyId,
} from "@magnitudedev/ai"
export {
  AVAILABLE_PROVIDER_MODEL,
  isProviderModelAvailable,
  ModelCatalogError,
  ProviderIdSchema,
  ProviderModelIdSchema,
  ModelFamilyIdSchema,
  ProviderModelAvailabilitySchema,
  ProviderModelSchema,
  ModelDiscoveryOperationIdSchema,
  ModelPropertyDiscoveryErrorSchema,
  ModelPropertyDiscoveryRequestSchema,
  ModelPropertyNameSchema,
  ReasoningEffortSchema,
  ReasoningProperty,
  VisionProperty,
} from "@magnitudedev/ai"
export type {
  ModelDiscoveryOperationId,
  ModelPropertyDiscoveryError,
  ModelPropertyDiscoveryRequest,
  ModelPropertyName,
  ReasoningEffort,
} from "@magnitudedev/ai"
export type ProviderClientError = MagnitudeClientError
export type ProviderRegistryInfo = RegistryProviderInfo
export type { ProviderCatalogOutcome } from "@magnitudedev/providers"

export interface ProviderClientConfig extends MagnitudeClientConfig {
  readonly discoverableProviders?: readonly DiscoverableProviderInstance[]
}

export type {
  MagnitudeModelInfo,
  FetchUsageOptions,
  CloudUsageResponse,
  MagnitudeCallOptions,
  MagnitudeAdditionalOptions,
} from "@magnitudedev/providers"
export type { WebSearchResult, UsageQuery } from "@magnitudedev/ai"
export type { WebSearchError } from "@magnitudedev/providers"
export type { UsagePeriod } from "@magnitudedev/protocol"

// =============================================================================
// Re-exported constants and helpers
// =============================================================================

export {
  classifyModelFamilyFromEvidence,
  classifyMagnitudeRejectedResponse,
  tryParseErrorBody,
  type ParsedMagnitudeApiError,
} from "@magnitudedev/providers"
export { makeFileBackedModelCatalog } from "@magnitudedev/ai"
export {
  createMagnitudeCompatibleSpec,
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
  readonly catalogs: {
    readonly list: Effect.Effect<readonly ProviderCatalogOutcome[], never, HttpClient.HttpClient>
    readonly refresh: (providerId?: ProviderId) => Effect.Effect<readonly ProviderCatalogOutcome[], never, HttpClient.HttpClient>
  }
  readonly listProviders: Effect.Effect<readonly ProviderRegistryInfo[], never, HttpClient.HttpClient>
  readonly sessionId: string | null
  readonly resolveModel: (
    providerId: ProviderId,
    providerModelId: ProviderModelId,
    options?: ProviderModelBindOptions,
  ) => Effect.Effect<BoundModel<BaseCallOptions>, never, never>
  readonly discoverModelProperties: (
    providerId: ProviderId,
    request: ModelPropertyDiscoveryRequest,
  ) => Effect.Effect<ModelDiscoveryOperationId, ModelPropertyDiscoveryError>
  readonly requestAttribution: (
    providerId: ProviderId,
    providerModelId: ProviderModelId,
    key: string,
  ) => RequestAttribution
  readonly webSearch: (
    query: string,
    schema?: Record<string, unknown>,
  ) => Effect.Effect<WebSearchResult, WebSearchError, HttpClient.HttpClient>
  readonly usage: (
    query?: UsageQuery,
  ) => Effect.Effect<CloudUsageResponse, ProviderClientError, HttpClient.HttpClient>
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

  const registry = makeProviderRegistry({
    magnitude: magnitudeInstance,
    discoverableProviders: config?.discoverableProviders ?? [],
  })

  return {
    catalog: registry.aggregatedCatalog,
    catalogs: registry.catalogs,
    listProviders: registry.listProviders,
    sessionId,
    resolveModel: (providerId, providerModelId, options) =>
      registry.resolveModel(providerId, providerModelId, options),
    discoverModelProperties: registry.discoverModelProperties,
    requestAttribution: (_providerId, _providerModelId, key) => ({ key, requestStarted: Effect.void }),
    webSearch: magnitudeInstance.provider.webSearch,
    usage: magnitudeInstance.provider.usage,
    runtimeConfig: {
      preferProvider: process.env.MAGNITUDE_PREFER_PROVIDER || undefined,
      disableTraits: !!process.env.MAGNITUDE_DISABLE_TRAITS,
    },
  }
}
