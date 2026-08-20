import {
  NativeChatCompletions,
  Option,
  type ModelSpec,
  type BoundModel,
  type BaseCallOptions,
  type ToolChoice as AiToolChoice,
} from "@magnitudedev/ai"
import { Schema } from "effect"
import { classifyMagnitudeRejectedResponse } from "./errors"
import {
  MagnitudeAdditionalOptionsSchema,
  type MagnitudeAdditionalOptions,
} from "./contract"


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
  toolChoice: NativeChatCompletions.options.toolChoice,
  magnitudeAdditionalOptions: Option.field(
    "magnitude_additional_options",
    MagnitudeAdditionalOptionsSchema,
  ),
  reasoningEffort: Option.field("reasoning_effort", Schema.String),
} as const

export function createMagnitudeCompatibleSpec(config: MagnitudeCompatibleSpecConfig) {
  return NativeChatCompletions.model({
    modelId: config.modelId,
    endpoint: config.endpoint,
    options: magnitudeOptions,
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
        magnitudeAdditionalOptions: bakedOptions,
      }),
  }
}
