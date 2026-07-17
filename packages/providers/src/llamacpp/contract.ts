/**
 * Llama.cpp provider contract types.
 */

import { Schema } from "effect"
import { ProviderModelFields } from "@magnitudedev/ai"

/**
 * A model served by a local Llama.cpp server.
 * Extends ProviderModel with Llama.cpp-specific metadata.
 */
export const LlamaCppProviderId = Schema.Literal("llamacpp").pipe(Schema.brand("ProviderId"))
export const LlamaServedModelIdSchema = Schema.String.pipe(Schema.minLength(1), Schema.maxLength(4096), Schema.brand("LlamaServedModelId"))
export type LlamaServedModelId = Schema.Schema.Type<typeof LlamaServedModelIdSchema>
export const LlamaServingRouteIdSchema = Schema.String.pipe(Schema.minLength(1), Schema.maxLength(8192), Schema.brand("LlamaServingRouteId"))
export type LlamaServingRouteId = Schema.Schema.Type<typeof LlamaServingRouteIdSchema>
export const LlamaCppModelInfoSchema = Schema.Struct({
  ...ProviderModelFields,
  providerId: LlamaCppProviderId,
}).pipe(Schema.filter((model) => {
  const reasoning = model.properties.reasoning
  return reasoning._tag !== "Cached"
    && reasoning._tag !== "Resolved"
    && reasoning._tag !== "Refreshing"
    || reasoning.value.includes(model.defaultReasoningEffort)
}, { message: () => "Discovered reasoning efforts must contain defaultReasoningEffort" }))
export type LlamaCppModelInfo = Schema.Schema.Type<typeof LlamaCppModelInfoSchema>

/**
 * Call options for Llama.cpp inference.
 * Llama.cpp supports the standard OpenAI-compatible options plus
 * temperature, top_p, and stop sequences.
 */
export interface LlamaCppCallOptions {
  readonly maxTokens?: number
  readonly toolChoice?: LlamaCppToolChoice
  readonly chatTemplateKwargs?: Readonly<Record<string, unknown>>
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
