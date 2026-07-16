import type { Effect, Option } from "effect"
import type { BoundModel } from "../model/bound-model"
import type { ModelFamilyId, ProviderId, ProviderModel, ProviderModelId } from "./model"
import type { ModelCatalog } from "./catalog"
import type { ProviderModelCapabilities, ImagePlaceholderConfig } from "../model/capabilities"
import type { BaseCallOptions, ToolChoice } from "./call-options"

/**
 * The base provider interface — defines what a provider is.
 * Provider-agnostic: no specific provider IDs, no specific model families.
 *
 * Every provider returns `BoundModel<BaseCallOptions>` — the universal call
 * options shape. Provider-specific options are baked in at bind time inside
 * the provider's `model()` implementation, never seen by the agent layer.
 */
export interface Provider<
  TModel extends ProviderModel = ProviderModel,
> {
  readonly id: ProviderId
  readonly displayName: string
  readonly catalog: ModelCatalog<TModel>

  /**
   * Bind a model for inference.
   * `providerModelId` is the provider-specific model ID (ProviderModel.providerModelId).
   * Returns a `BoundModel<BaseCallOptions>` — provider-specific options are
   * baked in at bind time and invisible to the caller.
   */
  readonly bindModel: (
    providerModelId: ProviderModelId,
    options?: ProviderModelBindOptions,
  ) => Effect.Effect<BoundModel<BaseCallOptions>, never, never>

  /**
   * Classify a model into a known model family.
   * Takes the provider model without `modelFamilyId` (which is what
   * we're computing). Returns the family ID, or None if the model
   * cannot be classified. Each provider's catalog decides whether an
   * unclassified model is excluded or surfaced with an unknown family.
   */
  readonly classifyModelFamily: (model: Omit<TModel, "modelFamilyId">) => Option.Option<ModelFamilyId>
}

/**
 * Options passed when binding a model for inference.
 * Provider-specific fields (traits, agentId) are ignored by providers
 * that don't use them.
 */
export interface ProviderModelBindOptions {
  readonly defaults?: Partial<BaseCallOptions>
  readonly capabilities?: ProviderModelCapabilities
  readonly imagePlaceholders?: ImagePlaceholderConfig
  /** Agent ID — used by providers that support tracing/metadata. Ignored by others. */
  readonly agentId?: string
  /** Role ID — used by providers that support tracing/metadata. Ignored by others. */
  readonly roleId?: string
  /** Traits — Magnitude-specific, ignored by other providers. */
  readonly traits?: readonly string[]
  /** Prefer a specific upstream provider — Magnitude-specific, ignored by others. */
  readonly preferProvider?: string
}

// ── Provider extensions ──────────────────────────────────────────────
// Provider-specific capabilities (web search, balance) are separate
// interfaces, not part of the base Provider. Only providers that
// implement them declare conformance.

export interface WebSearchResult {
  readonly text: string
  readonly sources: ReadonlyArray<{ readonly title: string; readonly url: string }>
  readonly data?: unknown
}

export interface WebSearchExtension<TResult = WebSearchResult, TError = unknown, R = unknown> {
  readonly webSearch: (
    query: string,
    schema?: Record<string, unknown>,
  ) => Effect.Effect<TResult, TError, R>
}

export interface BalanceQuery {
  readonly period?: string
  readonly days?: number
  readonly tz?: string
}

export interface BalanceResponse {
  readonly balance: number
  readonly usage?: unknown
}

export interface BalanceExtension<TResponse = BalanceResponse, TError = unknown, R = unknown> {
  readonly balance: (
    query?: BalanceQuery,
  ) => Effect.Effect<TResponse, TError, R>
}
