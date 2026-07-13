/**
 * Llama.cpp provider contract types.
 */

import type { ProviderModel } from "@magnitudedev/ai"

/**
 * A model served by a local Llama.cpp server.
 * Extends ProviderModel with Llama.cpp-specific metadata.
 */
export interface LlamaCppModelInfo extends ProviderModel {
  readonly providerId: "llamacpp"
  /** Server-configured context window (from /slots or /props), if available */
  readonly serverContextSize?: number
}

/**
 * Raw model entry from GET /v1/models (OpenAI-compatible format).
 */
export interface LlamaCppRawModel {
  readonly id: string
  readonly object: string
  readonly created?: number
  readonly owned_by?: string
  readonly meta?: LlamaCppModelMeta
}

/**
 * Metadata returned by Llama.cpp server in the model listing.
 */
export interface LlamaCppModelMeta {
  readonly n_ctx?: number
  readonly n_ctx_train?: number
  readonly n_vocab?: number
  readonly n_params?: number
  readonly size?: number
  readonly general_architecture?: string
  readonly general_name?: string
  readonly general_basename?: string
  readonly general_version?: string
  readonly general_finetune?: string
  readonly general_size_label?: string
}

/**
 * Response from GET /v1/models.
 */
export interface LlamaCppModelsResponse {
  readonly object: string
  readonly data: readonly LlamaCppRawModel[]
}

/**
 * Call options for Llama.cpp inference.
 * Llama.cpp supports the standard OpenAI-compatible options plus
 * temperature, top_p, and stop sequences.
 */
export interface LlamaCppCallOptions {
  readonly maxTokens?: number
  readonly toolChoice?: LlamaCppToolChoice
  readonly reasoningEffort?: string
  readonly temperature?: number
  readonly topP?: number
  readonly stop?: readonly string[]
}

export type LlamaCppToolChoice =
  | "none"
  | "auto"
  | "required"
  | { readonly type: "function"; readonly function: { readonly name: string } }
  | { readonly type: "grammar"; readonly grammar: string }
  | { readonly type: "allowed_tools"; readonly allowed_tools: { readonly mode: "auto" | "required"; readonly tools: ReadonlyArray<{ readonly type: "function"; readonly function: { readonly name: string } }> } }
