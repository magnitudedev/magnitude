import { Data, Effect } from "effect"
import type * as HttpClient from "@effect/platform/HttpClient"
import type { AuthApplicator } from "../auth/auth"
import type { ProviderId, ProviderModel, ProviderModelId } from "./model"

export class ModelCatalogError extends Data.TaggedError("ModelCatalogError")<{
  readonly message: string
  readonly cause?: unknown
}> {}

/**
 * Provider-agnostic model catalog. Lists, gets, and refreshes models
 * from a provider's model listing endpoint.
 *
 * Generic over the model type `T` (which must extend `ProviderModel`).
 * Code that only needs provider-agnostic fields can use `ModelCatalog`
 * (defaults to `ProviderModel`). Concrete providers use
 * `ModelCatalog<MyModelInfo>` to preserve provider-specific fields.
 */
export interface ModelCatalog<T extends ProviderModel = ProviderModel> {
  /** Returns cached models if fresh, otherwise fetches. */
  readonly list: Effect.Effect<readonly T[], ModelCatalogError, HttpClient.HttpClient>
  /** Finds a model by provider ID + provider model ID. Fails if not found. */
  readonly get: (providerId: ProviderId, providerModelId: ProviderModelId) => Effect.Effect<T, ModelCatalogError, HttpClient.HttpClient>
  /** Forces a fresh fetch, replacing the cache. */
  readonly refresh: Effect.Effect<readonly T[], ModelCatalogError, HttpClient.HttpClient>
}

/**
 * Config for a basic HTTP-backed catalog implementation.
 * Concrete providers use this to configure their catalog factory.
 */
export interface ModelCatalogConfig {
  readonly endpoint: string
  readonly auth: AuthApplicator
  readonly ttlMs?: number
}
