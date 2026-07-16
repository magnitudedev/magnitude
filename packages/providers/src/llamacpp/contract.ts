/**
 * Llama.cpp provider contract types.
 */

import { Schema } from "effect"
import { ProviderModelSchema } from "@magnitudedev/ai"

/**
 * A model served by a local Llama.cpp server.
 * Extends ProviderModel with Llama.cpp-specific metadata.
 */
export const LlamaCppProviderId = Schema.Literal("llamacpp")
export const LlamaCppModelInfoSchema = Schema.Struct({
  ...ProviderModelSchema.fields,
  providerId: LlamaCppProviderId,
})
export type LlamaCppModelInfo = Schema.Schema.Type<typeof LlamaCppModelInfoSchema>

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
