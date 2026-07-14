import {
  NativeChatCompletions,
  Option,
  type ChatCompletionsRequest,
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

type LlamaCppChatCompletionsRequest = Partial<ChatCompletionsRequest> & {
  readonly chat_template_kwargs?: {
    readonly enable_thinking?: boolean
    readonly reasoning_effort?: string
  }
}

/** Map Magnitude's generic effort setting to llama-server's template controls. */
export function composeLlamaCppReasoningRequest(
  wire: Partial<ChatCompletionsRequest>,
): LlamaCppChatCompletionsRequest {
  const withoutProviderDetails = {
    ...wire,
    messages: wire.messages?.map((message) => {
      if (message.role !== "assistant" || message.reasoning_details === undefined) return message
      const { reasoning_details: _reasoningDetails, ...rest } = message
      return rest
    }),
  }
  const effort = withoutProviderDetails.reasoning_effort as string | undefined
  if (effort === undefined) return withoutProviderDetails

  const { reasoning_effort: _reasoningEffort, ...rest } = withoutProviderDetails
  if (effort === "default") return rest
  if (effort === "none") {
    return {
      ...rest,
      chat_template_kwargs: { enable_thinking: false },
    }
  }
  return {
    ...rest,
    chat_template_kwargs: {
      enable_thinking: true,
      reasoning_effort: effort,
    },
  }
}

export function createLlamaCppCompatibleSpec(config: LlamaCppCompatibleSpecConfig) {
  return NativeChatCompletions.model({
    modelId: config.modelId,
    endpoint: config.endpoint,
    options: llamacppOptions,
    compose: composeLlamaCppReasoningRequest,
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
