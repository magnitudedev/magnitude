/**
 * Llama.cpp provider contract types.
 */

import type { ProviderModel } from "@magnitudedev/ai"
import type { LlamaCppServedModel } from "@magnitudedev/llamacpp/client"

/**
 * A model served by a local Llama.cpp server.
 * Extends ProviderModel with Llama.cpp-specific metadata.
 */
export interface LlamaCppModelInfo extends ProviderModel {
  readonly providerId: "llamacpp"
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

export type ServerStatus =
  | { readonly status: "ready"; readonly endpoint: string }
  | { readonly status: "loading"; readonly endpoint: string }
  | { readonly status: "error"; readonly endpoint: string; readonly message: string }
  | { readonly status: "not_found"; readonly endpoint: string }

export interface ServerProps {
  readonly nCtx?: number
  readonly modelAlias?: string
  readonly modelFtype?: string
  readonly modelPath?: string
  readonly chatTemplate?: string
  readonly modalities?: {
    readonly vision?: boolean
    readonly audio?: boolean
  }
}

export interface LlamaCppDiscoveryResult {
  readonly models: readonly LlamaCppModelInfo[]
  readonly status: "ok" | "loading" | "not_found" | "error"
  readonly endpoint: string
  readonly message?: string
  readonly hint?: string
}

/** Raw model observation decoded by the shared llama.cpp endpoint client. */
export type LlamaCppRawModel = LlamaCppServedModel

/** Metadata decoded by the shared llama.cpp endpoint client. */
export type LlamaCppModelMeta = NonNullable<LlamaCppServedModel["meta"]>

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
