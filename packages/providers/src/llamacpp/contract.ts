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
  /** Whether Magnitude manages the model runtime or connects to an external server. */
  readonly ownership: "managed" | "external"
  readonly residency?: "loaded" | "sleeping" | "unloaded" | "loading" | "failed" | "unknown"
  readonly productRank?: number
  readonly externalPriority?: number
  readonly servedModelId?: string
  readonly externalServerId?: string
  readonly managedArtifactId?: string
  /** Server-configured context window (from /slots or /props), if available */
  readonly serverContextSize?: number
  /** Underlying model path, when llama-server exposes one separately from its API ID. */
  readonly sourceModelPath?: string
  /** Human-readable name from parsed GGUF metadata or a compatible server response. */
  readonly metadataName?: string
  /** Architecture identifier from parsed GGUF metadata or a compatible server response. */
  readonly modelArchitecture?: string
  /** Broad GGUF tokenizer implementation, such as gpt2 or llama. */
  readonly tokenizerModel?: string
  /** llama.cpp pre-tokenizer identifier embedded in GGUF metadata. */
  readonly tokenizerPre?: string
  /** Human-readable base-model names embedded in GGUF metadata. */
  readonly baseModelNames?: readonly string[]
  /** Base-model repositories embedded in GGUF metadata. */
  readonly baseModelRepositories?: readonly string[]
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
