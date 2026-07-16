import {
  NativeChatCompletions,
  Option,
  type ModelSpec,
  type BoundModel,
  type ImagePlaceholderConfig,
  type ProviderModelCapabilities,
  type BaseCallOptions,
} from "@magnitudedev/ai"
import { classifyLlamaCppRejectedResponse } from "./errors"
import type { LlamaCppCallOptions, LlamaCppToolChoice } from "./contract"

export type { LlamaCppCallOptions, LlamaCppToolChoice } from "./contract"
export type { ModelProfile } from "@magnitudedev/ai"
export { toModelProfile } from "@magnitudedev/ai"

export type LlamaCppModelSpec = ModelSpec<LlamaCppCallOptions>

export interface LlamaCppCompatibleSpecConfig {
  readonly modelId: string
  readonly endpoint: string
}

const llamacppOptions = {
  maxTokens: NativeChatCompletions.options.maxTokens,
  toolChoice: Option.define(
    (v: LlamaCppToolChoice) => ({ tool_choice: v }),
  ),
  reasoningEffort: Option.define(
    (v: string) => ({ reasoning_effort: v }),
  ),
  temperature: Option.define(
    (v: number) => ({ temperature: v }),
  ),
  topP: Option.define(
    (v: number) => ({ top_p: v }),
  ),
  stop: Option.define(
    (v: readonly string[]) => ({ stop: [...v] }),
  ),
} as const

export function createLlamaCppCompatibleSpec(config: LlamaCppCompatibleSpecConfig) {
  return NativeChatCompletions.model({
    modelId: config.modelId,
    endpoint: config.endpoint,
    path: "/chat/completions?autoload=false",
    options: llamacppOptions,
    classifyRejectedResponse: classifyLlamaCppRejectedResponse,
  })
}

/**
 * Wrap an internal `BoundModel<LlamaCppCallOptions>` to accept
 * `BaseCallOptions` from the caller. Llama.cpp has no provider-specific
 * baked options, so this is a straightforward pass-through.
 */
export function wrapAsBaseModel(
  internal: BoundModel<LlamaCppCallOptions>,
): BoundModel<BaseCallOptions> {
  return {
    stream: (prompt, tools, options) =>
      internal.stream(prompt, tools, options),
  }
}
