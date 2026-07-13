import type { ToolCallId } from "../prompt/ids"

/**
 * Tool choice — the universal shape every provider accepts.
 * OpenAI-compatible: "none" | "auto" | "required" | named function | grammar | allowed_tools.
 */
export type ToolChoice =
  | "none"
  | "auto"
  | "required"
  | { readonly type: "function"; readonly function: { readonly name: string } }
  | { readonly type: "grammar"; readonly grammar: string }
  | {
      readonly type: "allowed_tools"
      readonly allowed_tools: {
        readonly mode: "auto" | "required"
        readonly tools: ReadonlyArray<{ readonly type: "function"; readonly function: { readonly name: string } }>
      }
    }

/**
 * Universal call options that every provider accepts.
 * Provider-specific options (e.g. magnitudeAdditionalOptions) are NOT here —
 * they're baked in at bind time inside the provider's `model()` implementation.
 */
export interface BaseCallOptions {
  readonly maxTokens?: number
  readonly toolChoice?: ToolChoice
  readonly reasoningEffort?: string
  readonly generateToolCallId?: () => ToolCallId
}
