import type {
  AuthApplicator,
  ModelCatalog,
  Provider,
  ProviderModel,
} from "@magnitudedev/ai"
import type { ModelsDevClient } from "../catalog/models-dev"

export interface OpenAiCompatibleRawModel {
  readonly id: string
  readonly name?: string
  readonly display_name?: string
  readonly context_length?: number
  readonly context_window?: number
  readonly max_context_length?: number
  readonly max_output_tokens?: number
  readonly max_tokens?: number
  readonly description?: string
  readonly owned_by?: string
  /** Provider-authoritative capability flags used by Kimi-compatible catalogs. */
  readonly supports_reasoning?: boolean
  readonly supports_image_in?: boolean
  readonly supports_video_in?: boolean
  readonly supported_parameters?: readonly string[]
  readonly reasoning?: {
    readonly supported_efforts?: readonly string[] | null
    readonly default_effort?: string
    readonly default_enabled?: boolean
    readonly mandatory?: boolean
  }
}

export interface OpenAiCompatibleModelsResponse {
  readonly data: readonly OpenAiCompatibleRawModel[]
}

export interface OpenAiCompatibleModelInfo extends ProviderModel {
  readonly providerId: string
}

export type LiveCatalogFallback = "never" | "unsupported_only" | "always"

export interface OpenAiCompatibleCatalogConfig<
  TModel extends OpenAiCompatibleModelInfo,
> {
  readonly providerId: TModel["providerId"]
  readonly endpoint: string
  readonly auth: AuthApplicator
  readonly modelsDevProviderId: string
  readonly modelsDev: ModelsDevClient
  readonly ttlMs?: number
  readonly liveCatalog?: boolean
  readonly liveCatalogFallback?: LiveCatalogFallback
  readonly requireOpenWeights?: boolean
  readonly requireToolCalls?: boolean
  readonly defaultContextWindow?: number
  readonly defaultMaxOutputTokens?: number
  readonly toolChoiceModes?: readonly ("auto" | "none" | "required" | "named")[]
  readonly mapModel?: (model: OpenAiCompatibleModelInfo) => TModel
}

export interface OpenAiCompatibleProviderInstance<
  TModel extends OpenAiCompatibleModelInfo,
> {
  readonly provider: Provider<TModel>
  readonly catalog: ModelCatalog<TModel>
}
