import {
  NativeChatCompletions,
  Option,
  type ModelSpec,
  type BoundModel,
  type BaseCallOptions,
  type ChatCompletionsRequest,
  type ToolChoice as AiToolChoice,
} from "@magnitudedev/ai"
import { classifyMagnitudeRejectedResponse } from "./errors"
import type { MagnitudeAdditionalOptions } from "./contract"

export type { ModelProfile } from "@magnitudedev/ai"
export { toModelProfile } from "@magnitudedev/ai"

export type MagnitudeModelSpec = ModelSpec<MagnitudeCallOptions>

export interface MagnitudeCompatibleSpecConfig {
  modelId: string
  endpoint: string
}

/**
 * Internal call options for the Magnitude provider.
 * `magnitudeAdditionalOptions` is baked in at bind time — callers only see
 * `BaseCallOptions`.
 */
export type MagnitudeCallOptions = {
  maxTokens?: number
  toolChoice?: AiToolChoice
  magnitudeAdditionalOptions?: MagnitudeAdditionalOptions
  reasoningEffort?: string
}

const magnitudeOptions = {
  maxTokens: NativeChatCompletions.options.maxTokens,
  toolChoice: Option.define(
    (v: AiToolChoice) => ({ tool_choice: v }),
  ),
  magnitudeAdditionalOptions: Option.define(
    (v: MagnitudeAdditionalOptions) => ({ magnitude_additional_options: v }),
  ),
  reasoningEffort: Option.define(
    (v: string) => ({ reasoning_effort: v }),
  ),
} as const

function stripProviderReasoningDetails(
  wire: Partial<ChatCompletionsRequest>,
): Partial<ChatCompletionsRequest> {
  return {
    ...wire,
    messages: wire.messages?.map((message) => {
      if (message.role !== "assistant" || message.reasoning_details === undefined) return message
      const { reasoning_details: _reasoningDetails, ...rest } = message
      return rest
    }),
  }
}

export function createMagnitudeCompatibleSpec(config: MagnitudeCompatibleSpecConfig) {
  return NativeChatCompletions.model({
    modelId: config.modelId,
    endpoint: config.endpoint,
    options: magnitudeOptions,
    compose: stripProviderReasoningDetails,
    classifyRejectedResponse: classifyMagnitudeRejectedResponse,
  })
}

/**
 * Wrap an internal `BoundModel<MagnitudeCallOptions>` to accept
 * `BaseCallOptions` from the caller. `magnitudeAdditionalOptions` is baked
 * in at bind time and invisible to the caller.
 */
export function wrapAsBaseModel(
  internal: BoundModel<MagnitudeCallOptions>,
  bakedOptions: MagnitudeAdditionalOptions,
): BoundModel<BaseCallOptions> {
  return {
    stream: (prompt, tools, options) =>
      internal.stream(prompt, tools, {
        ...options,
        magnitudeAdditionalOptions: {
          ...bakedOptions,
        },
      }),
  }
}
